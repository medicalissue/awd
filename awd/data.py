"""CIFAR-10 / CIFAR-100 loaders, cs-giung/swa convention.

Splits
------
The full CIFAR train set has 50000 examples; cs-giung carves the last
``valid_size`` (= 5000 by default) into a held-out validation split,
leaving 45000 for training. The standard 10000-image test set is kept
unchanged. This convention matches the cs-giung config:

    DATASETS.CIFAR.TRAIN_INDICES: [0, 45000]
    DATASETS.CIFAR.VALID_INDICES: [45000, 50000]

Setting ``valid_size=0`` falls back to 50000 train + 10000 test (with
the test set masquerading as the validation loader for backwards
compatibility with simpler training scripts).

Augmentation
------------
"STANDARD" (cs-giung): 4-pixel reflective pad → RandomCrop(32) →
RandomHorizontalFlip → ToTensor → mean/std normalize. No mixup, no
RandAug. This matches the SWA paper recipe.
"""
from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


_NORM = {
    "cifar10":  ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def _build_transforms(dataset: str):
    mean, std = _NORM[dataset]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def build_loaders(args):
    """Return (train_loader, valid_loader, test_loader).

    valid_loader is None when valid_size == 0.
    """
    dataset = args.data_set.lower()
    if dataset not in _NORM:
        raise ValueError(f"data_set must be one of {list(_NORM)}, got {args.data_set!r}")

    data_dir = args.data_path
    os.makedirs(data_dir, exist_ok=True)
    train_tf, eval_tf = _build_transforms(dataset)
    cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100

    # Two copies of the train set — one with augmentation (for the
    # train slice) and one without (for the valid slice carved from the
    # tail). Indices are deterministic so the 45000/5000 split is
    # identical across processes / runs / seeds.
    train_full_aug = cls(data_dir, train=True, download=True, transform=train_tf)
    train_full_eval = cls(data_dir, train=True, download=True, transform=eval_tf)
    test_set = cls(data_dir, train=False, download=True, transform=eval_tf)

    n_total = len(train_full_aug)
    valid_size = int(getattr(args, "valid_size", 0))
    if valid_size > 0:
        train_idx = list(range(0, n_total - valid_size))
        valid_idx = list(range(n_total - valid_size, n_total))
        train_set = Subset(train_full_aug, train_idx)
        valid_set = Subset(train_full_eval, valid_idx)
    else:
        train_set = train_full_aug
        valid_set = None

    use_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
    if use_dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, shuffle=True, drop_last=True
        )
        valid_sampler = (
            torch.utils.data.distributed.DistributedSampler(
                valid_set, shuffle=False, drop_last=False
            ) if valid_set is not None else None
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_set, shuffle=False, drop_last=False
        )
    else:
        train_sampler = valid_sampler = test_sampler = None

    eval_bs = max(args.batch_size, 256)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    valid_loader = (
        DataLoader(
            valid_set,
            batch_size=eval_bs,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        ) if valid_set is not None else None
    )
    test_loader = DataLoader(
        test_set,
        batch_size=eval_bs,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    # If no valid split, the per-epoch eval falls back to the test set.
    return train_loader, (valid_loader if valid_loader is not None else test_loader), test_loader
