"""Muon: momentum + Newton-Schulz orthogonalisation of the update.

Reference: Jordan et al. (modded-nanogpt), scaled up in Moonlight (16B MoE /
5.7T tokens) and Kimi K2 (1T params / 15.5T tokens). Muon is applied only to
2D hidden matrices; embeddings, the output head and all 1D tensors use AdamW.

Distributed notes
-----------------
* DDP: parameters are replicated, so each rank holds the full matrix and
  Newton-Schulz runs locally. No extra communication. This is the phase-0/1
  path and it is exact.
* FSDP2: parameters are DTensors sharded across ranks. Orthogonalisation is
  *not* a per-element op, so the matrix must be whole. We keep the momentum
  buffer sharded (momentum is elementwise, so that part is safe), then
  all-gather only the resulting update, orthogonalise, and re-shard. That is
  one all-gather per matrix per step - correct, but it does not scale to very
  large world sizes. For phase 2, prefer a genuinely distributed Muon:
  Megatron-Core's `Emerging-Optimizers` (Apr 2026) or torchtitan #2494.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

try:  # torch >= 2.4
    from torch.distributed.tensor import DTensor, distribute_tensor

    _HAS_DTENSOR = True
except ImportError:  # pragma: no cover
    DTensor = None  # type: ignore[assignment]
    distribute_tensor = None  # type: ignore[assignment]
    _HAS_DTENSOR = False


# Quintic iteration coefficients tuned to push the singular values of the
# normalised matrix towards 1 in as few steps as possible. The iteration does
# not converge to an exact orthogonalisation - it does not need to; what
# matters is that the spectrum is squashed into a narrow band.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonal factor of G via a quintic Newton-Schulz iteration.

    Runs in bfloat16 by design: the iteration is numerically forgiving and this
    is on the critical path of every step.
    """
    assert G.ndim >= 2, "Muon expects matrices"
    a, b, c = _NS_COEFFS
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


def _gather(t: torch.Tensor):
    """Return (full_tensor, resharding_spec_or_None)."""
    if _HAS_DTENSOR and isinstance(t, DTensor):
        return t.full_tensor(), (t.device_mesh, t.placements)
    return t, None


def _scatter(full: torch.Tensor, spec) -> torch.Tensor:
    if spec is None:
        return full
    mesh, placements = spec
    return distribute_tensor(full, mesh, placements)


class Muon(Optimizer):
    """Muon for 2D parameters. Every param in every group must be 2D."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"invalid momentum: {momentum}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon received a {p.ndim}D parameter of shape {tuple(p.shape)}; "
                        "route embeddings, the output head and norms to AdamW instead"
                    )

    @torch.no_grad()
    def step(self):  # noqa: D102
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]

                # Elementwise, so safe to run directly on a sharded DTensor.
                buf.lerp_(g, 1.0 - momentum)
                update = g.lerp(buf, momentum) if nesterov else buf

                # Orthogonalisation needs the whole matrix.
                full, spec = _gather(update)
                ortho_full = zeropower_via_newtonschulz5(full, steps=ns_steps)
                ortho = _scatter(ortho_full, spec)

                # RMS-matching scale (Moonlight): keeps the update magnitude
                # comparable to AdamW's across very non-square matrices, which
                # is what makes AdamW-tuned learning rates transfer.
                rows, cols = p.shape[-2], p.shape[-1]
                scale = max(1.0, rows / cols) ** 0.5

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(ortho, alpha=-lr * scale)


class OptimizerGroup:
    """Drives several optimizers as one, with a shared LR multiplier.

    Each optimizer keeps its own base LR (Muon and AdamW live on completely
    different scales); the scheduler supplies a single 0..1 multiplier.
    """

    def __init__(self, optimizers: list[Optimizer]):
        self.optimizers = optimizers
        self._base_lrs = [[g["lr"] for g in o.param_groups] for o in optimizers]

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def set_lr_mult(self, mult: float) -> None:
        for opt, bases in zip(self.optimizers, self._base_lrs, strict=True):
            for group, base in zip(opt.param_groups, bases, strict=True):
                group["lr"] = base * mult

    def current_lrs(self) -> list[float]:
        return [g["lr"] for g in self.param_groups]

    def step(self) -> None:
        for o in self.optimizers:
            o.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"optimizers": [o.state_dict() for o in self.optimizers], "base_lrs": self._base_lrs}

    def load_state_dict(self, sd: dict) -> None:
        for o, s in zip(self.optimizers, sd["optimizers"], strict=True):
            o.load_state_dict(s)
        self._base_lrs = sd["base_lrs"]


def build_optimizer(model, cfg) -> OptimizerGroup:
    """Build the Muon+AdamW pair (or AdamW alone) from an OptimConfig."""
    muon_params, adam_params = model.param_groups()

    if not cfg.use_muon:
        adam_params = muon_params + adam_params
        muon_params = []

    optimizers: list[Optimizer] = []
    if muon_params:
        optimizers.append(
            Muon(
                muon_params,
                lr=cfg.muon_lr,
                momentum=cfg.muon_momentum,
                nesterov=cfg.muon_nesterov,
                ns_steps=cfg.muon_ns_steps,
                weight_decay=cfg.muon_weight_decay,
            )
        )
    if adam_params:
        # Weight decay is applied to >=2D tensors only; decaying norm gains and
        # biases is a well-known small regression.
        decay = [p for p in adam_params if p.ndim >= 2]
        no_decay = [p for p in adam_params if p.ndim < 2]
        groups = []
        if decay:
            groups.append({"params": decay, "weight_decay": cfg.adam_weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        optimizers.append(
            torch.optim.AdamW(
                groups,
                lr=cfg.adam_lr,
                betas=tuple(cfg.adam_betas),
                eps=cfg.adam_eps,
                fused=torch.cuda.is_available(),
            )
        )
    return OptimizerGroup(optimizers)
