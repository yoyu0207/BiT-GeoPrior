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
