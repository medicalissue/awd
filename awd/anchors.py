"""Per-parameter anchor state, shared by AnchoredSGD and AdamE.

The anchor θ_anchor is the point our generalized weight decay pulls θ
toward. Picking the anchor parametrizes a family that contains:

    origin  → standard L2 weight decay
    init    → L2-SP (anchor frozen at θ_0)
    ema     → AdamE / proposed (running EMA over θ)
    polyak  → SWA-style running mean (used as anchor, not just at eval)
    window  → mean over last W parameter snapshots

This module is optimizer-agnostic: it only knows how to maintain the
anchor and compute the pull term. AnchoredSGD and AdamE both call into
it. Lazy state init means we don't have to know the parameter shapes
at construction time — same idiom as torch.optim's per-parameter state.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import torch


VALID_ANCHORS = ("origin", "init", "ema", "polyak", "window")


def init_anchor_state(state: dict, p: torch.Tensor, kind: str, window: int) -> None:
    """Lazily populate per-parameter anchor state.

    Called once per parameter, on the first step. Cheap for `origin`
    (no state at all); ~1× param memory for the others; `window` adds
    W× param memory in a bounded deque.
    """
    if kind == "origin":
        return
    if kind == "window":
        buf: deque = deque(maxlen=window)
        buf.append(p.detach().clone())
        state["anchor_window"] = buf
        state["anchor"] = p.detach().clone()
        return
    # init / ema / polyak all start at θ_0.
    state["anchor"] = p.detach().clone()


@torch.no_grad()
def anchor_pull(
    p: torch.Tensor,
    state: dict,
    kind: str,
    lr: float,
    lam: float,
    *,
    normalize: bool = False,
    norm_eps: float = 1e-12,
) -> None:
    """Apply the decoupled anchor pull in-place.

    Two modes, controlled by ``normalize``:

    Default (normalize=False), magnitude follows ``θ − anchor``:
        p ← p − lr·λ·(p − θ_anchor)
        For kind='origin' this collapses to p ← (1 − lr·λ)·p (≡ wd).
        For other kinds, ``λ‖θ−anchor‖`` is the effective force; this
        is small whenever θ is near its anchor (typical for EMA where
        ‖θ−θ_EMA‖ ≪ ‖θ‖), so the same λ that produces meaningful wd
        produces near-zero ed.

    normalize=True, magnitude follows ``‖θ‖`` (matched to wd):
        u = (p − θ_anchor) / ‖p − θ_anchor‖
        p ← p − lr·λ·‖p‖·u
        Direction comes from the anchor; magnitude is ‖p‖ — the same
        as wd's `λθ` formulation. Picks anchor selection apart from
        decay strength so a single λ scale is comparable across
        anchors.

    For kind='origin' the normalized form coincides with the default
    (the unit vector toward 0 is θ̂ = θ/‖θ‖, and ‖θ−0‖ = ‖θ‖), so we
    never need a separate origin path.

    Per-tensor norm so layer-wise wd scaling (BN scale ≈ 1, conv
    kernel ≈ 0.1, FC ≈ 0.05, etc.) is preserved — matches torch's wd
    behavior layer for layer.
    """
    if lam == 0.0:
        return
    if kind == "origin":
        # Unchanged regardless of `normalize` — see docstring.
        p.mul_(1.0 - lr * lam)
        return
    anchor = state["anchor"]
    if not normalize:
        p.mul_(1.0 - lr * lam).add_(anchor, alpha=lr * lam)
        return
    # Direction-normalized form.
    direction = p - anchor
    # Per-tensor norms; eps guards the very-early step where direction
    # ≈ 0 (anchor and p coincide at init for ema/polyak/window).
    dir_norm = direction.norm()
    if dir_norm.item() < norm_eps:
        return
    p_norm = p.norm()
    # Effective shrink: λ·‖p‖, applied along (p−anchor) unit vector.
    coef = lr * lam * (p_norm / dir_norm).item()
    p.add_(direction, alpha=-coef)


@torch.no_grad()
def update_anchor(p: torch.Tensor, state: dict, kind: str, *, ema_decay: float, step: int) -> None:
    """Update the anchor *after* the parameter has stepped.

    'init' is frozen, 'origin' has no buffer, 'ema'/'polyak'/'window'
    all advance.
    """
    if kind in ("origin", "init"):
        return
    if kind == "ema":
        anchor = state["anchor"]
        anchor.mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)
        return
    if kind == "polyak":
        anchor = state["anchor"]
        # Running mean: anchor ← (k·anchor + p) / (k+1).
        k = float(step - 1)
        anchor.mul_(k / (k + 1.0)).add_(p.detach(), alpha=1.0 / (k + 1.0))
        return
    if kind == "window":
        buf: deque = state["anchor_window"]
        buf.append(p.detach().clone())
        anchor = state["anchor"]
        anchor.zero_()
        for snap in buf:
            anchor.add_(snap, alpha=1.0 / len(buf))
        return
    raise ValueError(f"Unknown anchor kind: {kind!r}")


def serialize_anchor_state(state: dict) -> dict[str, Any]:
    """Return the anchor-related fields from `state` as a checkpoint dict.

    deque is not directly torch.save-friendly across versions — flatten
    to a list of tensors. Reverse in `deserialize_anchor_state`.
    """
    out = {}
    if "anchor" in state:
        out["anchor"] = state["anchor"]
    if "anchor_window" in state:
        out["anchor_window"] = list(state["anchor_window"])
    return out


def deserialize_anchor_state(state: dict, payload: dict[str, Any], window: int) -> None:
    """Inverse of serialize_anchor_state."""
    if "anchor" in payload:
        state["anchor"] = payload["anchor"]
    if "anchor_window" in payload:
        state["anchor_window"] = deque(payload["anchor_window"], maxlen=window)
