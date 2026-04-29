"""Optimizers for the Anchored Weight Decay project.

Two optimizers, sharing the anchor machinery in awd.anchors:

    AnchoredSGD  : SGD + Nesterov momentum + optional coupled L2 (wd)
                   + optional anchored decay (ed). The CIFAR/SWA-paper
                   recipe baseline. This is the *main* optimizer in the
                   paper.

    AdamE        : Adam with decoupled, anchored decay. Reduces to
                   AdamW exactly when anchor='origin'. Used for the ViT
                   lane (transformers / ImageNet) where AdamW is the
                   standard baseline.

Both call into awd.anchors for anchor state and the pull term, so the
anchor selection axis (origin / init / ema / polyak / window) is
factored out from the optimizer family.

Convention notes
----------------
* "wd"  = standard L2 weight decay, anchor=origin.
* "ed"  = generalized / anchored decay, anchor != origin.
The 4-cell ablation = {wd off/on} × {ed off/on}. With wd=on, ed=on the
update sees both pulls (toward 0 and toward θ_anchor) summed; the
gradient form makes that linear, so the implementation is just two
calls to anchor_pull with different (kind, λ).

Sanity invariants
-----------------
* AnchoredSGD with wd=0, ed=0 ≡ plain SGD+Nesterov (unit-tested).
* AdamE with anchor='origin' is bit-identical to torch AdamW for the
  same (lr, betas, eps, weight_decay).
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim.optimizer import Optimizer

from .anchors import (
    VALID_ANCHORS,
    anchor_pull,
    init_anchor_state,
    update_anchor,
)


# ─────────────────────────────────────────────────────────────────────
#  AnchoredSGD — main lane
# ─────────────────────────────────────────────────────────────────────
class AnchoredSGD(Optimizer):
    """SGD + Nesterov + (optional) L2 wd + (optional) anchored decay.

    Update (per parameter, per step):
        g_t = ∇L(θ_t)
        # decay terms (decoupled, applied as direct param shifts before
        # the SGD step — cleaner than folding into g_t because Nesterov
        # interacts with momentum):
        if wd > 0:  θ_t ← θ_t − lr·wd·θ_t                     # standard L2
        if ed > 0:  θ_t ← θ_t − lr·ed·(θ_t − θ_anchor,t)      # anchored
        # SGD with Nesterov momentum on the bare gradient:
        v_{t+1} = μ·v_t + g_t
        θ_{t+1} = θ_t − lr·(g_t + μ·v_{t+1})  if nesterov
                = θ_t − lr·v_{t+1}             else

    The decay is applied *before* the SGD step. This decoupling matches
    the spirit of AdamW vs Adam-with-L2: the decay is not a source of
    momentum, it's a direct contractive map on θ.

    Parameters
    ----------
    params : iterable
        Parameters or param groups.
    lr : float
        Learning rate (scheduled by main.py per-step).
    momentum : float
        Momentum coefficient (0.9 in the SWA recipe).
    nesterov : bool
        Use Nesterov momentum (True in the SWA recipe).
    weight_decay : float
        λ_wd for the standard L2 term (anchor=origin). 0 disables.
    weight_decay_form : str
        'coupled'   — wd is added to the gradient (g ← g + λ_wd·θ),
                      then flows through Nesterov momentum. Matches
                      torch.optim.SGD(weight_decay=λ_wd) and the
                      cs-giung/swa baseline (which adds λ·½‖θ‖² to
                      the loss).
        'decoupled' — wd is applied as a direct shrink on θ, separate
                      from momentum. Matches AdamW's spirit but for
                      SGD. Cleaner pairing with `ed_lambda` since both
                      then act as direct contractive maps. Default.
        Use 'coupled' to match the SWA-paper / cs-giung baseline acc;
        use 'decoupled' when comparing wd vs ed apples-to-apples.
    ed_lambda : float
        λ_ed for the anchored decay term. 0 disables. Always decoupled
        (applied as a direct anchor pull on θ — that's the framework's
        defining choice). When >0 the anchor specified by ``anchor``
        (default 'ema') is maintained.
    ed_normalize : bool
        If True, the ed pull uses the direction-normalized form
        ``-λ·‖θ‖·(θ−anchor)/‖θ−anchor‖`` so the ed strength is
        comparable to wd at the same λ (per-tensor magnitudes match
        wd's `λθ`). If False, uses the raw ``-λ·(θ−anchor)`` form,
        which is much weaker for ema-like anchors where
        ‖θ−anchor‖ ≪ ‖θ‖.
    anchor : str
        One of awd.anchors.VALID_ANCHORS. Only consulted when ed_lambda
        > 0; ignored otherwise.
    ema_decay : float
        α in θ_anchor ← α·θ_anchor + (1−α)·θ. Used iff anchor='ema'.
    window : int
        Window size for anchor='window'.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.1,
        momentum: float = 0.9,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        weight_decay_form: str = "coupled",
        ed_lambda: float = 0.0,
        ed_normalize: bool = False,
        anchor: str = "ema",
        ema_decay: float = 0.9999,
        window: int = 16,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if weight_decay_form not in ("coupled", "decoupled"):
            raise ValueError(
                f"weight_decay_form must be 'coupled' or 'decoupled', got {weight_decay_form!r}"
            )
        if ed_lambda < 0.0:
            raise ValueError(f"Invalid ed_lambda: {ed_lambda}")
        if anchor not in VALID_ANCHORS:
            raise ValueError(f"anchor must be one of {VALID_ANCHORS}, got {anchor!r}")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            weight_decay_form=weight_decay_form,
            ed_lambda=ed_lambda,
            ed_normalize=ed_normalize,
            anchor=anchor,
            ema_decay=ema_decay,
            window=window,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            wd_form = group["weight_decay_form"]
            ed = group["ed_lambda"]
            ed_norm = group.get("ed_normalize", False)
            anchor_kind = group["anchor"]
            ema_decay = group["ema_decay"]
            window = group["window"]
            need_anchor = ed > 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AnchoredSGD does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0.0:
                        state["momentum_buffer"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    if need_anchor:
                        init_anchor_state(state, p, anchor_kind, window)
                state["step"] += 1
                step = state["step"]

                # 1) Coupled wd flows through the gradient (and thus
                # through momentum). Decoupled wd is applied as a
                # direct shrink later, alongside ed.
                if wd != 0.0 and wd_form == "coupled":
                    grad = grad.add(p, alpha=wd)

                if wd != 0.0 and wd_form == "decoupled":
                    anchor_pull(p, state, "origin", lr=lr, lam=wd)
                if need_anchor:
                    anchor_pull(p, state, anchor_kind, lr=lr, lam=ed,
                                normalize=ed_norm)

                # 2) SGD step on the (possibly wd-augmented) gradient.
                if momentum > 0.0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    if nesterov:
                        # standard Nesterov: g_eff = g + μ·v
                        g_eff = grad.add(buf, alpha=momentum)
                    else:
                        g_eff = buf
                    p.add_(g_eff, alpha=-lr)
                else:
                    p.add_(grad, alpha=-lr)

                # 3) Anchor evolves from the *post-step* parameter.
                if need_anchor:
                    update_anchor(p, state, anchor_kind, ema_decay=ema_decay, step=step)

        return loss


# ─────────────────────────────────────────────────────────────────────
#  AdamE — secondary lane (ViT / ImageNet)
# ─────────────────────────────────────────────────────────────────────
class AdamE(Optimizer):
    """Adam with anchored, decoupled weight decay.

    Update:
        m_t = β1·m_{t-1} + (1-β1)·g_t
        v_t = β2·v_{t-1} + (1-β2)·g_t²
        m̂_t = m_t / (1-β1ᵗ);  v̂_t = v_t / (1-β2ᵗ)
        θ_t ← θ_t − lr·wd·(θ_t − θ_anchor,t)            # decoupled pull
        θ_{t+1} = θ_t − lr · m̂_t / (√v̂_t + ε)

    With anchor='origin', this is bit-identical to PyTorch AdamW for
    the same (lr, betas, eps, weight_decay).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        anchor: str = "origin",
        ema_decay: float = 0.9999,
        window: int = 16,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if anchor not in VALID_ANCHORS:
            raise ValueError(f"anchor must be one of {VALID_ANCHORS}, got {anchor!r}")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            anchor=anchor,
            ema_decay=ema_decay,
            window=window,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            anchor_kind = group["anchor"]
            ema_decay = group["ema_decay"]
            window = group["window"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamE does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if wd != 0.0:
                        init_anchor_state(state, p, anchor_kind, window)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Adam moment updates.
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bc1 = 1.0 - beta1**step
                bc2 = 1.0 - beta2**step
                step_size = lr / bc1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)

                # 1) Decoupled pull (BEFORE the Adam step, matching
                # PyTorch AdamW's convention so anchor='origin' is
                # bit-identical to AdamW).
                if wd != 0.0:
                    anchor_pull(p, state, anchor_kind, lr=lr, lam=wd)

                # 2) Adam adaptive step.
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # 3) Anchor evolution.
                if wd != 0.0:
                    update_anchor(p, state, anchor_kind, ema_decay=ema_decay, step=step)

        return loss


# ─────────────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────────────
def _split_decay_groups(model: torch.nn.Module, no_decay_keys: tuple[str, ...]):
    """Return (decay, no_decay) parameter lists.

    no_decay candidates: any param whose name contains a substring in
    no_decay_keys (typical: 'bias', 'bn', 'norm'), OR has ndim<=1 (a
    bias/scale tensor). Convention from timm/DeiT.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in no_decay_keys) or p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def build_optimizer(model: torch.nn.Module, args) -> Optimizer:
    """Build an optimizer from an argparse namespace.

    Recognized fields:
        optimizer        : 'sgd' or 'adame'
        lr, weight_decay : standard
        ed_lambda        : SGD-only — anchored decay strength
        anchor           : when (ed_lambda>0 for SGD) or (wd>0 for adame)
        ema_decay, window
        opt_betas, opt_eps : AdamE only
        momentum, nesterov : SGD only
        no_decay_keys    : iterable of name substrings for the no-decay
                           param group (default: bias, bn, norm). Both
                           wd and ed skip the no-decay group.

    Note on parameter groups: for SGD we always create two groups so
    that wd and ed both skip BN/bias parameters consistently — same
    convention as cs-giung/swa, which excludes batch-norm/bias from L2.
    """
    no_decay_keys = tuple(getattr(args, "no_decay_keys", ("bias", ".bn", "norm")))
    decay, no_decay = _split_decay_groups(model, no_decay_keys)

    optimizer_name = getattr(args, "optimizer", "sgd").lower()
    lr = float(args.lr)
    wd = float(getattr(args, "weight_decay", 0.0))
    ed = float(getattr(args, "ed_lambda", 0.0))
    ed_normalize = bool(getattr(args, "ed_normalize", False))
    anchor = getattr(args, "anchor", "ema")
    ema_decay = float(getattr(args, "ema_decay", 0.9999))
    window = int(getattr(args, "window", 16))

    if optimizer_name == "sgd":
        momentum = float(getattr(args, "momentum", 0.9))
        nesterov = bool(getattr(args, "nesterov", True))
        wd_form = getattr(args, "weight_decay_form", "coupled")
        param_groups = [
            {
                "params": decay,
                "weight_decay": wd,
                "weight_decay_form": wd_form,
                "ed_lambda": ed,
                "ed_normalize": ed_normalize,
                "anchor": anchor,
                "ema_decay": ema_decay,
                "window": window,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
                "weight_decay_form": wd_form,
                "ed_lambda": 0.0,
                "ed_normalize": ed_normalize,
                "anchor": "origin",
                "ema_decay": ema_decay,
                "window": window,
            },
        ]
        return AnchoredSGD(
            param_groups, lr=lr, momentum=momentum, nesterov=nesterov,
            weight_decay=wd, weight_decay_form=wd_form,
            ed_lambda=ed, ed_normalize=ed_normalize, anchor=anchor,
            ema_decay=ema_decay, window=window,
        )

    if optimizer_name == "adame":
        betas = tuple(getattr(args, "opt_betas", (0.9, 0.999)))
        eps = float(getattr(args, "opt_eps", 1e-8))
        param_groups = [
            {
                "params": decay,
                "weight_decay": wd,
                "anchor": anchor,
                "ema_decay": ema_decay,
                "window": window,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
                "anchor": "origin",
                "ema_decay": ema_decay,
                "window": window,
            },
        ]
        return AdamE(param_groups, lr=lr, betas=betas, eps=eps)

    if optimizer_name == "adamw":
        # Convenience alias — torch built-in for the AdamW=AdamE(origin)
        # baseline. Kept so configs can switch between them by name.
        betas = tuple(getattr(args, "opt_betas", (0.9, 0.999)))
        eps = float(getattr(args, "opt_eps", 1e-8))
        param_groups = [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)

    raise ValueError(f"Unknown optimizer: {optimizer_name}")
