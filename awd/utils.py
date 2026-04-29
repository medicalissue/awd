"""Distributed init, checkpoint I/O, logging helpers."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist


# ── Distributed ───────────────────────────────────────────────────────


def init_distributed_mode(args) -> None:
    """Init torch.distributed if launched via torchrun, else single-process.

    Mirrors the shape of timm/DeiT helpers so callers don't need to know
    whether they're under torchrun.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = True
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
        )
        dist.barrier()
    else:
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.distributed = False


def is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    val = value.clone()
    dist.all_reduce(val, op=dist.ReduceOp.SUM)
    val /= dist.get_world_size()
    return val


# ── Checkpoint ────────────────────────────────────────────────────────


def save_checkpoint(state: dict, output_dir: str, epoch: int, keep: int = 3) -> Path:
    """Atomic save to checkpoint-<epoch>.pth, prune to last `keep` files.

    Convention matches DyF orchestrate.sh: globbing checkpoint-*.pth in
    output_dir resumes the latest by epoch number.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"checkpoint-{epoch}.pth"
    tmp = out_dir / f"checkpoint-{epoch}.pth.tmp"
    torch.save(state, tmp)
    os.replace(tmp, final)

    # Prune older ones (keep the most recent `keep`).
    ckpts = sorted(
        out_dir.glob("checkpoint-*.pth"),
        key=lambda p: int(p.stem.rsplit("-", 1)[1]),
    )
    for stale in ckpts[:-keep]:
        try:
            stale.unlink()
        except OSError:
            pass
    return final


def find_latest_checkpoint(output_dir: str) -> Path | None:
    out_dir = Path(output_dir)
    if not out_dir.is_dir():
        return None
    ckpts = sorted(
        out_dir.glob("checkpoint-*.pth"),
        key=lambda p: int(p.stem.rsplit("-", 1)[1]),
    )
    return ckpts[-1] if ckpts else None


def write_complete_marker(output_dir: str) -> None:
    """Write a 'complete' sentinel for orchestrate.sh."""
    Path(output_dir, "complete").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\n"))


def write_args_json(args, output_dir: str) -> None:
    safe = {
        k: v
        for k, v in vars(args).items()
        if isinstance(v, (str, int, float, bool, list, tuple)) or v is None
    }
    Path(output_dir, "args.json").write_text(json.dumps(safe, indent=2, sort_keys=True))


# ── Logging ───────────────────────────────────────────────────────────


class FileLogger:
    """Append-only logger that mirrors to stdout on rank 0."""

    def __init__(self, output_dir: str, filename: str = "log.txt"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(output_dir, filename)
        self._is_main = is_main_process()

    def log(self, msg: str) -> None:
        if not self._is_main:
            return
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}"
        print(line, flush=True)
        with self.path.open("a") as f:
            f.write(line + "\n")

    def log_dict(self, d: dict) -> None:
        if not self._is_main:
            return
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] " + json.dumps(d, sort_keys=True)
        print(line, flush=True)
        with self.path.open("a") as f:
            f.write(line + "\n")


# ── Misc ──────────────────────────────────────────────────────────────


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError(f"Cannot parse {v!r} as bool")


def cosine_lr(step: int, total: int, warmup: int, base_lr: float, min_lr: float) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    import math

    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
