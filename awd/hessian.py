"""Hessian top-eigenvalue estimator via power iteration on Hv products.

Used to characterize sharpness of converged minima — a flatter minimum
has a smaller top eigenvalue. We use Hutchinson-free power iteration
because we only care about the dominant eigenpair, not the trace.

Single iteration cost: one extra backward pass per power-iter step.
20 iterations on ResNet-18 / CIFAR-100 ≈ a few minutes on a GPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def _normalize(vec_list: list[torch.Tensor]) -> float:
    norm_sq = sum((v * v).sum().item() for v in vec_list)
    norm = norm_sq**0.5
    if norm == 0:
        return 0.0
    for v in vec_list:
        v.div_(norm)
    return norm


def hessian_top_eigenvalue(
    model: nn.Module,
    loss_fn,
    data_iter,
    *,
    n_iters: int = 20,
    n_batches: int = 4,
    tol: float = 1e-3,
    device: torch.device | None = None,
) -> float:
    """Estimate the top eigenvalue of the loss Hessian via power iteration.

    Parameters
    ----------
    model : nn.Module
    loss_fn : callable(logits, targets) -> scalar
    data_iter : iterable yielding (input, target) batches
    n_iters : maximum power-iteration steps
    n_batches : how many minibatches to average per Hv (estimator
                noise control; 1 is fine for a quick read)
    tol : early-stop when |λ_k − λ_{k−1}| / |λ_k| < tol
    device : device to place v on (defaults to first param's)

    Returns
    -------
    float : estimated top eigenvalue
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    if device is None:
        device = params[0].device

    # Random unit vector matching params.
    v = [torch.randn_like(p) for p in params]
    _normalize(v)

    # Materialize a fixed list of batches so successive Hv products are
    # against the same loss surface (otherwise power iter diverges).
    batches = []
    it = iter(data_iter)
    for _ in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        batches.append((x.to(device, non_blocking=True), y.to(device, non_blocking=True)))
    if not batches:
        raise RuntimeError("No data batches available for Hessian eval")

    eig_prev = 0.0
    eig = 0.0
    for k in range(n_iters):
        # Hv across the cached batches, averaged.
        Hv = [torch.zeros_like(p) for p in params]
        for x, y in batches:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = model(x)
                loss = loss_fn(logits, y)
                grads = torch.autograd.grad(loss, params, create_graph=True)
                gv = sum((g * vv).sum() for g, vv in zip(grads, v))
                Hv_batch = torch.autograd.grad(gv, params, retain_graph=False)
            for acc, hb in zip(Hv, Hv_batch):
                acc.add_(hb.detach() / len(batches))
        # λ ≈ vᵀ H v.
        eig = sum((vv * hh).sum().item() for vv, hh in zip(v, Hv))
        # New v ← Hv / ‖Hv‖.
        norm = _normalize(Hv)
        if norm == 0:
            break
        v = Hv
        if k > 0 and abs(eig - eig_prev) / (abs(eig) + 1e-12) < tol:
            break
        eig_prev = eig

    model.zero_grad(set_to_none=True)
    return eig
