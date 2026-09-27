"""Adapter for the pinned official SNUNet-CD ECAM implementation."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


DEFAULT_SOURCE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "baseline_sources" / "Siam-NestedUNet"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class OfficialSNUNetCD(nn.Module):
    """Official SNUNet-CD/ECAM adapted to 8-channel binary logits.

    The official topology and ECAM module are unchanged. Only the first-layer
    input channels and final output channels are set to match this experiment.
    """

    def __init__(self, in_channels: int = 8, num_classes: int = 1,
                 source_root: str | os.PathLike[str] | None = None):
        super().__init__()
        root = Path(
            source_root
            or os.environ.get("SNUNET_OFFICIAL_ROOT", DEFAULT_SOURCE_ROOT)
        ).resolve()
        model_path = root / "models" / "Models.py"
        if not model_path.exists():
            raise FileNotFoundError(
                "Pinned official SNUNet-CD source is missing. Run "
                f"prepare_recent_baseline_sources.py first: {model_path}")
        module = _load_module("official_snunet_models", model_path)
        self.network = module.SNUNet_ECAM(
            in_ch=in_channels, out_ch=num_classes)

    def forward(self, image_a: torch.Tensor,
                image_b: torch.Tensor) -> torch.Tensor:
        output = self.network(image_a, image_b)
        return output[0] if isinstance(output, (list, tuple)) else output


class SNUNetCDLiteNoECAM(nn.Module):
    """SNUNet-CD ablation without ECAM and with half-width feature maps.

    This is an explicitly labelled ablation, not a replacement for the pinned
    official SNUNet-CD baseline. The nested Siamese topology is retained,
    while the initial width is reduced from 32 to 16 channels.
    """

    def __init__(self, in_channels: int = 8, num_classes: int = 1,
                 base_channels: int = 16,
                 source_root: str | os.PathLike[str] | None = None):
        super().__init__()
        root = Path(
            source_root
            or os.environ.get("SNUNET_OFFICIAL_ROOT", DEFAULT_SOURCE_ROOT)
        ).resolve()
        model_path = root / "models" / "Models.py"
        if not model_path.exists():
            raise FileNotFoundError(f"Pinned SNUNet-CD source is missing: {model_path}")
        module = _load_module("official_snunet_lite_components", model_path)
        block, up = module.conv_block_nested, module.up
        filters = [base_channels * (2 ** index) for index in range(5)]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0_0 = block(in_channels, filters[0], filters[0])
        self.conv1_0 = block(filters[0], filters[1], filters[1])
        self.up1_0 = up(filters[1])
        self.conv2_0 = block(filters[1], filters[2], filters[2])
        self.up2_0 = up(filters[2])
        self.conv3_0 = block(filters[2], filters[3], filters[3])
        self.up3_0 = up(filters[3])
        self.conv4_0 = block(filters[3], filters[4], filters[4])
        self.up4_0 = up(filters[4])

        self.conv0_1 = block(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_1 = block(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.up1_1 = up(filters[1])
        self.conv2_1 = block(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.up2_1 = up(filters[2])
        self.conv3_1 = block(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.up3_1 = up(filters[3])

        self.conv0_2 = block(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_2 = block(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.up1_2 = up(filters[1])
        self.conv2_2 = block(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.up2_2 = up(filters[2])

        self.conv0_3 = block(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_3 = block(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.up1_3 = up(filters[1])
        self.conv0_4 = block(filters[0] * 5 + filters[1], filters[0], filters[0])

        self.final1 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final4 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.conv_final = nn.Conv2d(num_classes * 4, num_classes, kernel_size=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, image_a: torch.Tensor,
                image_b: torch.Tensor) -> torch.Tensor:
        x0_0a = self.conv0_0(image_a)
        x1_0a = self.conv1_0(self.pool(x0_0a))
        x2_0a = self.conv2_0(self.pool(x1_0a))
        x3_0a = self.conv3_0(self.pool(x2_0a))

        x0_0b = self.conv0_0(image_b)
        x1_0b = self.conv1_0(self.pool(x0_0b))
        x2_0b = self.conv2_0(self.pool(x1_0b))
        x3_0b = self.conv3_0(self.pool(x2_0b))
        x4_0b = self.conv4_0(self.pool(x3_0b))

        x0_1 = self.conv0_1(torch.cat([x0_0a, x0_0b, self.up1_0(x1_0b)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0a, x1_0b, self.up2_0(x2_0b)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0a, x0_0b, x0_1, self.up1_1(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0a, x2_0b, self.up3_0(x3_0b)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0a, x1_0b, x1_1, self.up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(
            torch.cat([x0_0a, x0_0b, x0_1, x0_2, self.up1_2(x1_2)], 1)
        )

        x3_1 = self.conv3_1(torch.cat([x3_0a, x3_0b, self.up4_0(x4_0b)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0a, x2_0b, x2_1, self.up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(
            torch.cat([x1_0a, x1_0b, x1_1, x1_2, self.up2_2(x2_2)], 1)
        )
        x0_4 = self.conv0_4(
            torch.cat([x0_0a, x0_0b, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1)
        )

        outputs = [
            self.final1(x0_1), self.final2(x0_2),
            self.final3(x0_3), self.final4(x0_4),
        ]
        return self.conv_final(torch.cat(outputs, dim=1))
