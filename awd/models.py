"""CIFAR-sized backbones, matching cs-giung/swa configs.

We mirror cs-giung/swa exactly so our SGD baseline reproduces the SWA
paper's reported numbers (WRN-28×10 / CIFAR-100 baseline ≈ 80.6%
test acc, SWA ≈ 81.5%).

Models
------
- ResNet-20 (BN-ReLU)             — cs-giung's C100_R20-BN-ReLU
    BasicBlock, IN_PLANES=16, NUM_BLOCKS=[3,3,3], identity shortcut,
    first-block uses BN+activation+conv3x3-stride1.

- WRN-28×10 (BN-ReLU PreResNet)   — cs-giung's C100_WRN28x10-BN-ReLU
    PreResNet (BN-ReLU-Conv ordering), IN_PLANES=16, NUM_BLOCKS=[4,4,4],
    WIDEN_FACTOR=10, projection shortcut on stride/width changes,
    first conv has no preceding BN.

Param count check (CIFAR-100, num_classes=100):
    ResNet-20  ≈  0.27 M
    WRN-28×10  ≈ 36.5  M
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_c: int, out_c: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)


# ─────────────────────────────────────────────────────────────────────
#  ResNet-20 (post-activation BN-ReLU, identity shortcut)
# ─────────────────────────────────────────────────────────────────────
class _PostResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = _conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = _conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        if stride != 1 or in_c != out_c:
            # Identity-with-zero-pad → cs-giung uses a 1×1 conv instead.
            # We follow cs-giung: 1×1 conv + BN, since "IdentityShortcut"
            # in their config still allows projecting on shape change.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.shortcut(x), inplace=True)


class CifarResNet(nn.Module):
    """Post-activation ResNet for 32×32 inputs (He et al., CIFAR variant).

    Used by cs-giung's C{10,100}_R20-BN-ReLU configs (NUM_BLOCKS=[3,3,3],
    IN_PLANES=16). Layer counts follow the standard 6n+2 formula:
    n=3 → ResNet-20, n=5 → ResNet-32, n=9 → ResNet-56, n=18 → ResNet-110.
    """

    def __init__(self, num_blocks: tuple[int, int, int], in_planes: int = 16,
                 num_classes: int = 100):
        super().__init__()
        self.in_c = in_planes
        self.stem = nn.Sequential(
            _conv3x3(3, in_planes),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(in_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(in_planes * 4, num_blocks[2], stride=2)
        self.fc = nn.Linear(in_planes * 4, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, out_c: int, n: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (n - 1)
        layers = []
        for s in strides:
            layers.append(_PostResBlock(self.in_c, out_c, s))
            self.in_c = out_c
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.fc(h)


# ─────────────────────────────────────────────────────────────────────
#  PreResNet WRN-28×10 (BN-ReLU pre-activation, projection shortcut)
# ─────────────────────────────────────────────────────────────────────
class _PreActBasicBlock(nn.Module):
    """Pre-activation basic block: BN → ReLU → Conv → BN → ReLU → Conv.

    Shortcut is a 1×1 conv on shape change (ProjectionShortcut in
    cs-giung's terminology). The first BN-ReLU shares activations with
    the projection in the matching cs-giung implementation; we tap the
    shortcut after that first BN-ReLU when shapes change, matching the
    standard preact-WRN reference.
    """

    def __init__(self, in_c: int, out_c: int, stride: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = _conv3x3(in_c, out_c, stride)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = _conv3x3(out_c, out_c)
        self.has_proj = (stride != 1) or (in_c != out_c)
        if self.has_proj:
            self.shortcut = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(x), inplace=True)
        sc = self.shortcut(h) if self.has_proj else x
        h = self.conv1(h)
        h = F.relu(self.bn2(h), inplace=True)
        h = self.conv2(h)
        return h + sc


class PreResNetWRN(nn.Module):
    """Pre-activation Wide-ResNet for CIFAR.

    cs-giung's C100_WRN28x10-BN-ReLU config:
        BLOCK = "BasicBlock"   (pre-act here)
        SHORTCUT = "ProjectionShortcut"
        NUM_BLOCKS = [4, 4, 4]
        WIDEN_FACTOR = 10
        IN_PLANES = 16

    Total depth = 4·NUM_BLOCKS·2 + first_conv + final BN-ReLU + FC ≈ 28.
    Channel widths: 16, 16·k, 32·k, 64·k for k=10 → 16, 160, 320, 640.
    """

    def __init__(self, num_blocks: tuple[int, int, int] = (4, 4, 4),
                 widen_factor: int = 10, in_planes: int = 16, num_classes: int = 100):
        super().__init__()
        widths = [in_planes,
                  in_planes * widen_factor,
                  in_planes * 2 * widen_factor,
                  in_planes * 4 * widen_factor]
        # First conv has no preceding BN (USE_NORM_LAYER=False in
        # cs-giung's FIRST_BLOCK). Bias-free, kernel 3, stride 1.
        self.conv1 = _conv3x3(3, widths[0])
        self.stage1 = self._make_stage(widths[0], widths[1], num_blocks[0], stride=1)
        self.stage2 = self._make_stage(widths[1], widths[2], num_blocks[1], stride=2)
        self.stage3 = self._make_stage(widths[2], widths[3], num_blocks[2], stride=2)
        # Final BN-ReLU before the head (standard preact wrap-up).
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_stage(self, in_c: int, out_c: int, n: int, stride: int) -> nn.Sequential:
        layers = [_PreActBasicBlock(in_c, out_c, stride)]
        for _ in range(1, n):
            layers.append(_PreActBasicBlock(out_c, out_c, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = F.relu(self.bn(h), inplace=True)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.fc(h)


# ─────────────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────────────
def build_model(name: str, num_classes: int = 100) -> nn.Module:
    name = name.lower()
    if name in ("resnet20", "r20", "c_resnet20"):
        return CifarResNet((3, 3, 3), num_classes=num_classes)
    if name in ("resnet32", "r32"):
        return CifarResNet((5, 5, 5), num_classes=num_classes)
    if name in ("resnet56", "r56"):
        return CifarResNet((9, 9, 9), num_classes=num_classes)
    if name in ("resnet110", "r110"):
        return CifarResNet((18, 18, 18), num_classes=num_classes)
    if name in ("wrn_28_10", "wrn28x10", "wrn-28-10"):
        return PreResNetWRN((4, 4, 4), widen_factor=10, num_classes=num_classes)
    if name in ("wrn_16_8", "wrn16x8"):
        return PreResNetWRN((2, 2, 2), widen_factor=8, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")
