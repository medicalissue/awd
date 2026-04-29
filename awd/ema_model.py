"""Eval-only EMA model with BN re-estimation.

Maintains a parameter-and-buffer EMA copy of the trainee. Updated every
training step with a single global decay α (default 0.9999). Does NOT
participate in the gradient — its only role is to be a smoother,
flatter version of the trainee that we evaluate alongside the raw
model. When the optimizer's anchored decay (`ed`) is enabled, this
*same* EMA buffer is what the optimizer pulls θ toward — see
awd.optim.AnchoredSGD.

BN running statistics
---------------------
Per-step EMA of BN running_mean / running_var is biased: the trainee's
own running stats are themselves an EMA over batches, so EMA-of-EMA is
ill-defined and tends to misfit. Standard SWA practice — and what
cs-giung/swa does in `update_swa_batch_stats` — is to re-estimate the
EMA model's BN stats with one fresh forward pass over (a subset of)
the training set in eval-time. We expose that as a separate call:

    ema.update_parameters(trainee)        # every step (cheap)
    ema.bn_reestimate(train_loader, ...)  # before evaluating (~1 epoch
                                          # forward, no backward)
"""
from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import torch
import torch.nn as nn


class ModelEma:
    """Per-step parameter EMA with eval-time BN re-estimate.

    Parameters
    ----------
    model : nn.Module
        The trainee. We deepcopy it once at construction to seed the
        EMA copy; the trainee is then never mutated by us.
    decay : float
        α in θ_ema ← α·θ_ema + (1−α)·θ_trainee. SWA-friendly default
        is 0.9999 (effective window ≈ 10 k steps ≈ 25 epochs at batch
        128 / 50 k samples).
    device : torch.device | None
        Device for the EMA copy. Defaults to the trainee's.

    The class also exposes the same nn.Module API as `model` for the
    EMA: `ema.module(...)` runs forward on the EMA model.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999,
                 device: torch.device | None = None):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.module = deepcopy(self._unwrap(model))
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.module.eval()
        if device is not None:
            self.module.to(device)

    @staticmethod
    def _unwrap(m: nn.Module) -> nn.Module:
        # Strip DDP / DataParallel wrapper so we EMA the underlying model.
        if hasattr(m, "module") and isinstance(m.module, nn.Module):
            return m.module
        return m

    @torch.no_grad()
    def update_parameters(self, model: nn.Module) -> None:
        """Pull one EMA step toward the trainee.

        We EMA both `parameters()` and `buffers()`. Buffers include BN
        running_mean/var; we update them in lockstep, then *replace*
        them at eval-time via bn_reestimate (the EMA'd BN stats are a
        reasonable starting point for the re-estimate but not what we
        ultimately use). For non-BN buffers (e.g. ones() in BN
        weight, attention masks) the EMA is a no-op since they're
        constant.
        """
        trainee = self._unwrap(model)
        ema_params = dict(self.module.named_parameters())
        for name, p in trainee.named_parameters():
            ema_params[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

        ema_bufs = dict(self.module.named_buffers())
        for name, b in trainee.named_buffers():
            target = ema_bufs[name]
            if b.dtype.is_floating_point:
                target.mul_(self.decay).add_(b.data, alpha=1.0 - self.decay)
            else:
                # int buffers (BN's num_batches_tracked) — copy directly.
                target.copy_(b.data)

    @torch.no_grad()
    def bn_reestimate(
        self,
        loader: Iterable,
        device: torch.device,
        max_batches: int | None = None,
    ) -> None:
        """Re-estimate BN running stats with a single forward sweep.

        Sets all BatchNorm modules to .train() (so they update
        running_{mean,var}), zeros the running stats (so the new pass
        is the only contribution), forwards the loader, then restores
        .eval(). Equivalent to torch.optim.swa_utils.update_bn — we
        roll our own to avoid a state_dict dance and to keep the
        knob (max_batches) under our control for big datasets.
        """
        had_bn = False
        for m in self.module.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                had_bn = True
                m.reset_running_stats()
                # Use the batch-mean/var only for this sweep; the
                # accumulator is the running stat itself, started from
                # zero, so we set momentum to None which switches BN to
                # cumulative-average mode for the duration of the pass.
                m.momentum = None
                m.train()
        if not had_bn:
            return
        for i, batch in enumerate(loader):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True)
            self.module(x)
            if max_batches is not None and i + 1 >= max_batches:
                break
        for m in self.module.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    def state_dict(self) -> dict:
        return {"decay": self.decay, "module": self.module.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state.get("decay", self.decay)
        self.module.load_state_dict(state["module"])
