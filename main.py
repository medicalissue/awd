"""CIFAR-100 training entrypoint for the Anchored Weight Decay project.

The 4-cell experiment matrix this entrypoint targets:

                 ed=off            ed=on (anchor=ema)
    wd=off       'plain'           'ed'
    wd=on        'wd'              'wd_ed'

Every cell is trained with SGD+Nesterov on the cs-giung/swa recipe
(lr=0.1, momentum=0.9, cosine schedule + 5 epoch warmup, batch=128,
200 epochs). Every cell *also* maintains an eval-only EMA model with
α=0.9999, evaluated alongside the raw model. When ed is on, the same
EMA is what the optimizer's anchored decay pulls θ toward.

Output convention (must stay in sync with scripts/orchestrate.sh):
    $output_dir/
        log.txt              — appended every epoch (json lines)
        args.json            — argparse snapshot
        checkpoint-{N}.pth   — rolling, capped at --save_ckpt_num
        checkpoint.pt        — symlink to the latest (NELU convention)
        complete             — sentinel; presence ⇒ done
"""
from __future__ import annotations

import argparse
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from awd.data import build_loaders
from awd.ema_model import ModelEma
from awd.models import build_model
from awd.optim import build_optimizer
from awd.utils import (
    FileLogger,
    cosine_lr,
    find_latest_checkpoint,
    init_distributed_mode,
    is_main_process,
    save_checkpoint,
    str2bool,
    write_args_json,
    write_complete_marker,
)


# ── Args ──────────────────────────────────────────────────────────────


def get_args():
    p = argparse.ArgumentParser("AWD CIFAR trainer", add_help=False)
    # data
    p.add_argument("--data_path", type=str, default="./data")
    p.add_argument("--data_set", type=str, default="cifar100",
                   choices=["cifar10", "cifar100"])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_mem", type=str2bool, default=True)
    p.add_argument("--valid_size", type=int, default=5000,
                   help="Last `valid_size` train examples become the validation "
                        "split (cs-giung convention: 45000 train / 5000 valid). "
                        "Set to 0 to use full 50k train.")
    # model
    p.add_argument("--model", type=str, default="wrn_28_10")
    p.add_argument("--nb_classes", type=int, default=100)
    # optim
    p.add_argument("--optimizer", type=str, default="sgd",
                   choices=["sgd", "adamw", "adame"])
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--min_lr", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", type=str2bool, default=True)
    # wd: standard L2.
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--weight_decay_form", type=str, default="coupled",
                   choices=["coupled", "decoupled"],
                   help="'coupled' matches torch SGD + cs-giung baseline; "
                        "'decoupled' applies wd as a direct shrink (AdamW-style).")
    # ed: anchored decay.
    p.add_argument("--ed_lambda", type=float, default=0.0,
                   help="Anchored-decay strength λ_ed. 0 disables. With "
                        "anchor=ema this is the AdamE main effect.")
    p.add_argument("--anchor", type=str, default="ema",
                   choices=["origin", "init", "ema", "polyak", "window"])
    p.add_argument("--ema_decay", type=float, default=0.9999,
                   help="α for both the optimizer's EMA anchor and the eval EMA model.")
    p.add_argument("--window", type=int, default=16)
    # AdamE-only (ViT lane).
    p.add_argument("--opt_eps", type=float, default=1e-8)
    p.add_argument("--opt_betas", type=float, nargs="+", default=[0.9, 0.999])
    # schedule
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    # runtime
    p.add_argument("--use_amp", type=str2bool, default=True)
    p.add_argument("--amp_dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16"])
    p.add_argument("--compile", type=str2bool, default=False)
    p.add_argument("--clip_grad", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    # eval EMA
    p.add_argument("--eval_ema", type=str2bool, default=True,
                   help="Maintain a separate eval-only EMA model and evaluate it.")
    p.add_argument("--bn_reestimate_batches", type=int, default=0,
                   help="If >0, run that many train batches through the EMA "
                        "model to refresh BN stats before each EMA eval. "
                        "0 → re-estimate over the full train loader.")
    # checkpointing
    p.add_argument("--output_dir", type=str, default="./runs/default")
    p.add_argument("--auto_resume", type=str2bool, default=True)
    p.add_argument("--save_ckpt", type=str2bool, default=True)
    p.add_argument("--save_ckpt_freq", type=int, default=1)
    p.add_argument("--save_ckpt_num", type=int, default=3)
    p.add_argument("--limit_train_batches", type=int, default=0,
                   help="If >0, stop the train epoch after this many batches "
                        "(smoke-test only).")
    # logging
    p.add_argument("--enable_wandb", type=str2bool, default=False)
    p.add_argument("--wandb_project", type=str, default="awd")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--exp_name", type=str, default="",
                   help="Optional run name for wandb. Defaults to output_dir basename.")
    return p.parse_args()


# ── Seed pinning ──────────────────────────────────────────────────────


def seed_everything(seed: int, rank: int = 0) -> None:
    """Pin all RNGs we touch.

    Per-rank offset so DataLoader workers across DDP ranks see different
    streams (required for non-trivial shuffling). Set deterministic
    cuDNN — slower, but the 4-cell comparison hinges on shared init.
    """
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # Deterministic conv kernels. Skipped for AMP because float16 conv
    # kernels are more limited under deterministic mode.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ── Train / eval ──────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    args,
    *,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    scaler: torch.cuda.amp.GradScaler | None,
    ema: ModelEma | None,
    logger: FileLogger,
) -> int:
    model.train()
    if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    use_scaler = scaler is not None and args.amp_dtype == "float16"

    n = 0
    correct = 0
    loss_sum = 0.0
    t0 = time.time()
    limit = getattr(args, "limit_train_batches", 0) or 0
    for batch_idx, (x, y) in enumerate(train_loader):
        if limit > 0 and batch_idx >= limit:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Per-step LR.
        lr = cosine_lr(global_step, total_steps, warmup_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        if args.use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x)
                loss = criterion(logits, y)
            if use_scaler:
                scaler.scale(loss).backward()
                if args.clip_grad is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if args.clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        # EMA update — after the param step. No-op if --eval_ema=false.
        if ema is not None:
            ema.update_parameters(model)

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            n += y.size(0)
            loss_sum += loss.item() * y.size(0)

        global_step += 1

    elapsed = time.time() - t0
    train_loss = loss_sum / max(n, 1)
    train_acc = correct / max(n, 1)
    logger.log_dict({
        "phase": "train", "epoch": epoch,
        "lr": optimizer.param_groups[0]["lr"],
        "loss": round(train_loss, 6), "acc": round(train_acc, 6),
        "step": global_step, "secs": round(elapsed, 2),
    })
    return global_step


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    n = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss_sum += loss.item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.size(0)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t = torch.tensor([loss_sum, correct, n], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(t)
        loss_sum, correct, n = t[0].item(), int(t[1].item()), int(t[2].item())
    return loss_sum / max(n, 1), correct / max(n, 1)


# ── Main ──────────────────────────────────────────────────────────────


def main():
    args = get_args()
    init_distributed_mode(args)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    seed_everything(args.seed, args.rank)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = FileLogger(args.output_dir)
    if is_main_process():
        write_args_json(args, args.output_dir)
        logger.log(f"args: {vars(args)}")

    # wandb (optional).
    wandb_run = None
    if args.enable_wandb and is_main_process():
        try:
            import wandb  # noqa: WPS433
            run_name = args.exp_name or os.path.basename(args.output_dir.rstrip("/"))
            wandb_kwargs = {
                "project": args.wandb_project,
                "name": run_name,
                "config": vars(args),
                "dir": args.output_dir,
                "resume": "allow",
            }
            if args.wandb_entity:
                wandb_kwargs["entity"] = args.wandb_entity
            wandb_run = wandb.init(**wandb_kwargs)
        except Exception as e:  # noqa: BLE001
            logger.log(f"wandb init failed: {e}")
            wandb_run = None

    # Data.
    train_loader, valid_loader, test_loader = build_loaders(args)
    if is_main_process():
        logger.log(
            f"data: train={len(train_loader.dataset)}  "
            f"valid={len(valid_loader.dataset) if valid_loader else 0}  "
            f"test={len(test_loader.dataset)}"
        )

    # Model.
    model = build_model(args.model, num_classes=args.nb_classes)
    model = model.to(device)
    if args.compile:
        model = torch.compile(model)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    base_model = model.module if hasattr(model, "module") else model
    optimizer = build_optimizer(base_model, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    ema = ModelEma(model, decay=args.ema_decay, device=device) if args.eval_ema else None

    scaler = (
        torch.cuda.amp.GradScaler()
        if (args.use_amp and args.amp_dtype == "float16" and device.type == "cuda")
        else None
    )

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    # Resume.
    start_epoch = 0
    global_step = 0
    if args.auto_resume:
        latest = find_latest_checkpoint(args.output_dir)
        if latest is not None:
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            base_model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt.get("global_step", start_epoch * steps_per_epoch)
            if scaler is not None and "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            if ema is not None and "ema" in ckpt:
                ema.load_state_dict(ckpt["ema"])
            logger.log(f"resumed from {latest} at epoch {start_epoch}")

    # Graceful SIGTERM (preempt) — let the current epoch finish, then save.
    preempt = {"flag": False}

    def _handler(signum, frame):
        preempt["flag"] = True
        logger.log(f"received signal {signum} — will checkpoint at next epoch boundary")

    signal.signal(signal.SIGTERM, _handler)

    # Train.
    best_raw = 0.0
    best_ema = 0.0
    for epoch in range(start_epoch, args.epochs):
        global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args,
            global_step=global_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            scaler=scaler,
            ema=ema,
            logger=logger,
        )

        # Eval — raw model.
        raw_loss, raw_acc = evaluate(model, valid_loader, device)
        best_raw = max(best_raw, raw_acc)
        eval_payload = {
            "phase": "eval", "epoch": epoch, "model": "raw",
            "loss": round(raw_loss, 6), "acc": round(raw_acc, 6),
            "best_acc": round(best_raw, 6),
        }
        logger.log_dict(eval_payload)
        if wandb_run is not None:
            wandb_run.log(
                {"raw/val_loss": raw_loss, "raw/val_acc": raw_acc, "epoch": epoch},
                step=global_step,
            )

        # Eval — EMA model with BN re-estimate.
        if ema is not None:
            max_bn = args.bn_reestimate_batches if args.bn_reestimate_batches > 0 else None
            ema.bn_reestimate(train_loader, device, max_batches=max_bn)
            ema_loss, ema_acc = evaluate(ema.module, valid_loader, device)
            best_ema = max(best_ema, ema_acc)
            ema_payload = {
                "phase": "eval", "epoch": epoch, "model": "ema",
                "loss": round(ema_loss, 6), "acc": round(ema_acc, 6),
                "best_acc": round(best_ema, 6),
            }
            logger.log_dict(ema_payload)
            if wandb_run is not None:
                wandb_run.log(
                    {"ema/val_loss": ema_loss, "ema/val_acc": ema_acc, "epoch": epoch},
                    step=global_step,
                )

        # Checkpoint.
        if (
            is_main_process()
            and args.save_ckpt
            and ((epoch + 1) % args.save_ckpt_freq == 0 or preempt["flag"])
        ):
            state = {
                "model": base_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "args": vars(args),
            }
            if scaler is not None:
                state["scaler"] = scaler.state_dict()
            if ema is not None:
                state["ema"] = ema.state_dict()
            ckpt_path = save_checkpoint(state, args.output_dir, epoch, keep=args.save_ckpt_num)
            # Also write the NELU-style 'checkpoint.pt' symlink so the
            # orchestrator's resume probe finds it via a stable name.
            stable = Path(args.output_dir, "checkpoint.pt")
            try:
                if stable.is_symlink() or stable.exists():
                    stable.unlink()
                stable.symlink_to(ckpt_path.name)
            except OSError:
                pass

        if preempt["flag"]:
            logger.log("exiting on preempt — checkpoint saved")
            return 0

    # Final test-set eval (cs-giung uses the held-out 10000).
    if test_loader is not None:
        raw_test_loss, raw_test_acc = evaluate(model, test_loader, device)
        logger.log_dict({
            "phase": "test", "model": "raw",
            "loss": round(raw_test_loss, 6), "acc": round(raw_test_acc, 6),
        })
        if ema is not None:
            max_bn = args.bn_reestimate_batches if args.bn_reestimate_batches > 0 else None
            ema.bn_reestimate(train_loader, device, max_batches=max_bn)
            ema_test_loss, ema_test_acc = evaluate(ema.module, test_loader, device)
            logger.log_dict({
                "phase": "test", "model": "ema",
                "loss": round(ema_test_loss, 6), "acc": round(ema_test_acc, 6),
            })
        if wandb_run is not None:
            payload = {"raw/test_acc": raw_test_acc, "raw/test_loss": raw_test_loss}
            if ema is not None:
                payload["ema/test_acc"] = ema_test_acc
                payload["ema/test_loss"] = ema_test_loss
            wandb_run.log(payload, step=global_step)

    if is_main_process():
        write_complete_marker(args.output_dir)
        logger.log("training complete")
    if wandb_run is not None:
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
