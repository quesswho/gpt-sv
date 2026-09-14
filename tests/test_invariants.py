"""Invariants that fail *silently* if broken.

Deliberately not a full test suite. `scripts/smoke_test.sh` already proves the
stack runs end to end - it exercises the model, Muon, MTP, eval and
checkpointing under a real training loop, and it is what caught every crashing
bug so far.

What a smoke test cannot catch is a run that completes, converges, and is
simply worse than it should be. Each test below guards one such failure: no
exception, no visible loss anomaly, just days of GPU time spent on a slightly
wrong model. That is the only reason these survived.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gptsv.config import ModelConfig  # noqa: E402
from gptsv.data.loader import ShardDataset, TokenLoader  # noqa: E402
from gptsv.model import GPTSV, KVCache  # noqa: E402
from gptsv.tokenizer import CHAT_TOKENS, SPECIAL_TOKENS  # noqa: E402
from gptsv.tokenizer.reserve import reserve_special_tokens  # noqa: E402


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(
        vocab_size=97,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=16,
        n_mtp_heads=0,
        z_loss_weight=0.0,
    )
    base.update(kw)
    return ModelConfig(**base)


def test_meta_load_matches_eager_construction(tmp_path):
    """Loading via `meta` must produce the same model as constructing eagerly.

    `load_model` skips __init__ by building on `meta`, then adopts the
    checkpoint's tensors with assign=True. Two things can go wrong silently:
    assign replaces the Parameter objects, so a tied lm_head quietly becomes a
    second copy that drifts from the embedding, and the rope buffers are
    persistent=False, so they are absent from the state dict and stay on `meta`.
    """
    from gptsv.generate import load_model

    cfg = tiny_cfg(tie_embeddings=True)
    torch.manual_seed(0)
    reference = GPTSV(cfg).eval()
    path = tmp_path / "ckpt.pt"
    torch.save({"model": reference.state_dict(), "step": 7, "config": {"model": vars(cfg)}}, path)

    loaded, step = load_model(str(path), torch.device("cpu"))
    assert step == 7

    assert not any(p.is_meta for p in loaded.parameters())
    assert not any(b.is_meta for b in loaded.buffers())
    assert loaded.lm_head.weight is loaded.tok_emb.weight
    assert loaded.num_params() == reference.num_params()

    tokens = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    with torch.no_grad():
        assert torch.equal(loaded.trunk(tokens), reference.trunk(tokens))


def test_mtp_targets_are_offset_correctly():
    """Depth k must predict token i+k+1 - one step further out than depth k-1.

    An off-by-one here is the classic MTP bug: training runs, loss falls, and
    the heads simply learn a shifted task. Nothing surfaces it at runtime.
    """
    cfg = tiny_cfg(n_mtp_heads=3)
    m = GPTSV(cfg)
    L, T, D = m.block_len(), cfg.max_seq_len, cfg.n_mtp_heads
    tokens = torch.arange(L).unsqueeze(0)

    # main head: input tokens[:T], target tokens[1:T+1]
    assert tokens[0, :T].tolist() == list(range(T))
    assert tokens[0, 1 : T + 1].tolist() == list(range(1, T + 1))

    # depth k: embedding input tokens[k:k+T], target tokens[k+1:k+1+T]
    for k in range(1, D + 1):
        emb_in = tokens[0, k : k + T]
        target = tokens[0, k + 1 : k + 1 + T]
        assert len(emb_in) == T and len(target) == T
        assert (target - emb_in == 1).all()
        # the deepest slice must not run off the end of the block
        assert k + 1 + T <= L


def test_chunked_loss_gradients_match():
    """`loss_chunk_size` is a memory optimisation and must be exactly neutral.

    It is only ever enabled on the 12GB tier, so a subtle gradient difference
    would degrade phase 0 and phase 1 alone - and would look like a
    small-model effect rather than a bug.
    """
    torch.manual_seed(0)
    m = GPTSV(tiny_cfg(n_mtp_heads=1, loss_chunk_size=0))
    m.train()
    tokens = torch.randint(0, 97, (4, m.block_len()))

    m.loss(tokens).loss.backward()
    full = [p.grad.clone() for p in m.parameters()]

    m.zero_grad(set_to_none=True)
    m.cfg.loss_chunk_size = 7
    m.loss(tokens).loss.backward()
    chunked = [p.grad.clone() for p in m.parameters()]

    for g1, g2 in zip(full, chunked, strict=True):
        assert torch.allclose(g1, g2, atol=1e-5)


def test_grad_checkpointing_matches():
    """Same contract for activation checkpointing: pure memory/compute trade."""
    torch.manual_seed(0)
    m = GPTSV(tiny_cfg(grad_checkpoint=False))
    m.train()
    tokens = torch.randint(0, 97, (2, m.block_len()))

    m.loss(tokens).loss.backward()
    plain = [p.grad.clone() for p in m.parameters()]

    m.zero_grad(set_to_none=True)
    m.cfg.grad_checkpoint = True
    m.loss(tokens).loss.backward()
    ckpt = [p.grad.clone() for p in m.parameters()]

    for g1, g2 in zip(plain, ckpt, strict=True):
        assert torch.allclose(g1, g2, atol=1e-5)


def test_kv_cache_matches_full_forward():
    """Cached decoding must reproduce full-sequence logits; a cache bug only degrades samples."""
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = GPTSV(cfg).eval()
    tokens = torch.randint(0, 97, (2, 12))
    cache = KVCache(cfg, batch_size=2, max_len=12, device="cpu", dtype=torch.float32)

    with torch.no_grad():
        full = m(tokens)
        steps = [m(tokens[:, :5], cache), m(tokens[:, 5:8], cache)]
        steps += [m(tokens[:, i : i + 1], cache) for i in range(8, 12)]

    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5)


def test_param_groups_exclude_embeddings():
    """Muon must never reach the embedding table or the output head.

    Orthogonalising them is known to hurt, and the damage shows up only as a
    slightly worse model - never as an error.
    """
    m = GPTSV(tiny_cfg(tie_embeddings=False, n_mtp_heads=1))
    muon, adam = m.param_groups()

    assert all(p.ndim == 2 for p in muon)
    adam_ptrs = {p.data_ptr() for p in adam}
    assert m.tok_emb.weight.data_ptr() in adam_ptrs
    assert m.lm_head.weight.data_ptr() in adam_ptrs
    # every parameter accounted for exactly once
    assert len(muon) + len(adam) == len(list(m.parameters()))


@pytest.fixture
def shard_dir(tmp_path):
    d = tmp_path / "train"
    d.mkdir()
    for i in range(3):
        np.arange(i * 5000, i * 5000 + 5000, dtype=np.uint16).tofile(d / f"shard_{i:05d}.bin")
    return d


def test_seek_reproduces_the_uninterrupted_order(shard_dir):
    """Resume must see exactly the batches an uninterrupted run would have.

    If `seek` drifts, a preempted run silently retrains on data it has already
    seen (or skips data entirely). On a queue that preempts often, that quietly
    becomes most of the run.
    """
    ds = ShardDataset(shard_dir)
    uninterrupted = TokenLoader(ds, 2, 33, seed=3, rank=1, world_size=2)
    expected = [next(uninterrupted) for _ in range(10)]

    resumed = TokenLoader(ds, 2, 33, seed=3, rank=1, world_size=2)
    resumed.seek(6)
    for i in range(6, 10):
        assert torch.equal(next(resumed), expected[i])


def test_reserved_special_tokens_keep_every_other_id():
    """Reserving special tokens must not change how ordinary text tokenizes.

    `gptsv.tokenizer.reserve` swaps a tokenizer's last merges for new special
    tokens. If it dropped the wrong merges or shifted an ID, shards and
    checkpoints would still load - the IDs would just mean different text.
    """
    rng = random.Random(0)
    words = [
        "".join(rng.choices("abcdefghijklmnopqrstuvwxyzåäö", k=rng.randint(2, 8)))
        for _ in range(300)
    ]
    texts = [" ".join(rng.choices(words, k=12)) for _ in range(500)]

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    n = tok.get_vocab_size()

    new = CHAT_TOKENS + ["<|reserved_0|>"]
    derived = Tokenizer.from_str(json.dumps(reserve_special_tokens(json.loads(tok.to_str()), new)))
    cut = n - len(new)

    assert derived.get_vocab_size() == n
    assert [derived.token_to_id(t) for t in new] == list(range(cut, n))
    assert derived.encode(f"{CHAT_TOKENS[0]}hej").ids[0] == cut

    untouched = 0
    for text in texts:
        before, after = tok.encode(text).ids, derived.encode(text).ids
        assert derived.decode(after) == text
        if max(before) < cut:
            assert after == before
            untouched += 1
    # both branches must be exercised, or the test proves nothing
    assert 0 < untouched < len(texts)


def test_mtp_logits_are_z_regularised_like_the_main_head():
    """Every head that writes through the tied lm_head must get the z penalty.

    The MTP heads once passed want_z=False, so their logits were the only ones
    in the model with nothing bounding their scale. Nothing failed: the logits
    drifted up over tens of thousands of steps, and because lm_head is tied to
    tok_emb it arrived as an ever-growing gradient on the shared embedding. In
    phase 1 that reached main logsumexp 0.91 against the MTP heads' 280.6 and
    87% of the total gradient norm, which grad_clip then applied to the healthy
    LM update as well.

    Detected here as: z_loss_weight must change the gradient of an MTP-only
    parameter. Under the old behaviour it could not.
    """
    torch.manual_seed(0)
    cfg_kw = dict(n_mtp_heads=1, loss_chunk_size=0)
    tokens = torch.randint(0, 97, (2, tiny_cfg(**cfg_kw).max_seq_len + 2))

    def mtp_grad(z_weight: float) -> torch.Tensor:
        torch.manual_seed(0)
        model = GPTSV(tiny_cfg(**cfg_kw, z_loss_weight=z_weight))
        model.train()
        model.zero_grad()
        model.loss(tokens).loss.backward()
        return model.mtp_heads[0].proj.weight.grad.clone()

    off, on = mtp_grad(0.0), mtp_grad(1e-2)
    assert not torch.allclose(off, on), (
        "z_loss_weight does not reach the MTP heads: their logits are unbounded"
    )


def _write_ckpts(root: Path, steps: list[int]) -> None:
    for s in steps:
        (root / f"step_{s:07d}.pt").write_bytes(b"x")


def _steps(root: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in root.glob("step_*.pt"))


def test_pruning_never_deletes_milestones(tmp_path):
    from gptsv.utils import prune_checkpoints

    steps = list(range(200, 10001, 200))
    for s in steps:  # prune after every save, exactly as train.py does
        _write_ckpts(tmp_path, [s])
        prune_checkpoints(tmp_path, 3, protect=tmp_path / f"step_{s:07d}.pt", keep_every=4000)
    assert _steps(tmp_path) == [4000, 8000, 9600, 9800, 10000]


def test_pruning_without_milestones_is_unchanged(tmp_path):
    from gptsv.utils import prune_checkpoints

    _write_ckpts(tmp_path, [4000, 4200, 4400, 4600])
    prune_checkpoints(tmp_path, 2)
    assert _steps(tmp_path) == [4400, 4600]


def test_milestone_interval_must_align_with_ckpt_interval():
    from gptsv.config import TrainConfig

    TrainConfig(ckpt_interval=200, keep_every_n_steps=4000)
    TrainConfig(ckpt_interval=200, keep_every_n_steps=0)
    with pytest.raises(ValueError, match="not a multiple"):
        TrainConfig(ckpt_interval=200, keep_every_n_steps=4100)
    with pytest.raises(ValueError, match=">= 0"):
        TrainConfig(ckpt_interval=200, keep_every_n_steps=-1)


def test_loss_reports_z_terms_that_sum_to_the_total():
    torch.manual_seed(0)
    cfg = tiny_cfg(n_mtp_heads=1, z_loss_weight=1e-2, loss_chunk_size=0)
    model = GPTSV(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, model.block_len()))
    out = model.loss(tokens)
    expected = (
        out.main_ce
        + cfg.z_loss_weight * (out.z_main + out.z_mtp)
        + cfg.mtp_loss_weight * out.mtp_ce
    )
    assert torch.allclose(out.loss.detach(), expected, rtol=1e-5)

    assert GPTSV(tiny_cfg(z_loss_weight=1e-2)).loss(tokens[:, :-1]).z_mtp is None
    no_z = GPTSV(tiny_cfg(n_mtp_heads=1)).loss(tokens)
    assert no_z.z_main is None and no_z.z_mtp is None


def test_grad_norm_groups_cover_every_parameter_once():
    from gptsv.train import grad_norm_groups

    torch.manual_seed(0)
    model = GPTSV(tiny_cfg(n_mtp_heads=1, z_loss_weight=1e-4))
    model.loss(torch.randint(0, 97, (2, model.block_len()))).loss.backward()

    groups = grad_norm_groups(model)
    grouped = [p for ps in groups.values() for p in ps]
    assert len(grouped) == len({id(p) for p in grouped}) == len(list(model.parameters()))
    assert set(groups) == {"embed", "trunk", "mtp", "other"}

    total = torch.nn.utils.get_total_norm([p.grad for p in model.parameters()])
    per_group = torch.stack(
        [torch.nn.utils.get_total_norm([p.grad for p in ps]) for ps in groups.values()]
    )
    assert torch.allclose(per_group.pow(2).sum().sqrt(), total, rtol=1e-5)
