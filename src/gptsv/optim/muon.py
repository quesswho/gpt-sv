"""Muon: momentum followed by Newton-Schulz orthogonalisation of the update.

From Jordan et al. (modded-nanogpt), with the update scaling from Moonlight.
Muon is applied only to 2D hidden matrices; embeddings, the output head and all
1D tensors use AdamW. Under DDP every rank holds the full matrices, so the
orthogonalisation runs locally with no extra communication.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

# Quintic iteration coefficients that push the singular values towards 1 in few
# steps. The result is only approximately orthogonal, which is enough.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonal factor of G via a quintic Newton-Schulz iteration.

    Runs in bfloat16, which the iteration tolerates well.
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
    def step(self, closure=None):  # noqa: D102
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

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

                buf.lerp_(g, 1.0 - momentum)
                update = g.lerp(buf, momentum) if nesterov else buf
                ortho = zeropower_via_newtonschulz5(update, steps=ns_steps)

                # Scale up tall matrices so their update size stays comparable
                # to wide ones.
                rows, cols = p.shape[-2], p.shape[-1]
                scale = max(1.0, rows / cols) ** 0.5

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(ortho, alpha=-lr * scale)

        return loss


class OptimizerGroup:
    """Drives several optimizers as one, with a shared LR multiplier.

    Each optimizer keeps its own base LR, since Muon and AdamW use very
    different scales. The scheduler supplies a single 0..1 multiplier.
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
        # No weight decay on norm gains and other 1D tensors.
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
