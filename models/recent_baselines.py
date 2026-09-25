"""Adapters for recent change-detection methods from pinned official sources."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18


DEFAULT_RSCHANGE_ROOT = (
    Path(__file__).resolve().parents[2] / "baseline_sources" / "rschange"
)
DEFAULT_EDGEREFNET_ROOT = (
    Path(__file__).resolve().parents[2] / "baseline_sources" / "EdgeRefNet"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _official_steinformer_decoder(source_root: Path):
    decoder_root = source_root / "rscd" / "models" / "decoderheads"
    required = [
        decoder_root / "stnet.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Pinned rschange source is missing. Run "
            "prepare_recent_baseline_sources.py first. Missing: "
            + ", ".join(missing)
        )

    for package in ("rscd", "rscd.models", "rscd.models.decoderheads"):
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__ = []
            sys.modules[package] = module

    stnet = _load_module(
        "rscd.models.decoderheads.stnet", decoder_root / "stnet.py")
    return stnet.STNet


def _official_edgerefnet(source_root: Path):
    model_path = source_root / "custom_models" / "networks.py"
    if not model_path.exists():
        raise FileNotFoundError(
            "Pinned EdgeRefNet source is missing. Run "
            f"prepare_recent_baseline_sources.py first. Missing: {model_path}"
        )
    source_text = model_path.read_text(encoding="utf-8")
    broken_start = source_text.index("def define_G(")
    broken_end = source_text.index(
        "###############################################################################\n"
        "# main Functions",
        broken_start,
    )
    source_text = (
        source_text[:broken_start]
        + "def define_G(*args, **kwargs):\n"
        + "    raise RuntimeError('Unused by the reproducibility adapter')\n\n"
        + source_text[broken_end:]
    )
    source_text = source_text.replace(
        "            mlp_dim=mlp_dim,  \n        )",
        "            mlp_dim=mlp_dim, dropout=0.0,\n        )",
        1,
    )
    source_text = source_text.replace(
        "if x.shape[1] != 3:", "if x.shape[1] not in (3, 8):")
    source_text = source_text.replace(
        "pretrained=True,", "pretrained=False,")
    source_text = source_text.replace(
        "self.SA4 = HFAB(input_channel=256,input_size=32,ratio=0.5)",
        "self.SA4 = HFAB(input_channel=256,input_size=16,ratio=0.5)",
    )
    unused_imports = (
        "from custom_models.BIT import",
        "from custom_models.EGCTNet import",
        "from custom_models.ChangeFormer import",
        "from custom_models.DTCDSCN import",
    )
    source_text = "\n".join(
        line for line in source_text.splitlines()
        if 'print(f"x' not in line
        and not line.startswith(unused_imports)
    )
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)
    attention_path = source_root / "custom_models" / "attention_block.py"
    attention_source = attention_path.read_text(encoding="utf-8")
    if attention_source.startswith(" import math"):
        attention_source = attention_source[1:]
    attention_module = types.ModuleType("custom_models.attention_block")
    attention_module.__file__ = str(attention_path)
    exec(
        compile(attention_source, str(attention_path), "exec"),
        attention_module.__dict__,
    )
    sys.modules["custom_models.attention_block"] = attention_module
    edge_path = source_root / "custom_models" / "edge_block.py"
    edge_source = edge_path.read_text(encoding="utf-8").replace(
        "nn.ConvTranspose2d(256, 64, kernel_size=4, stride=4)",
        "nn.ConvTranspose2d(256, 64, kernel_size=8, stride=8)",
    )
    edge_module = types.ModuleType("custom_models.edge_block")
    edge_module.__file__ = str(edge_path)
    exec(compile(edge_source, str(edge_path), "exec"), edge_module.__dict__)
    sys.modules["custom_models.edge_block"] = edge_module
    module = types.ModuleType("official_edgerefnet_networks")
    module.__file__ = str(model_path)
    exec(compile(source_text, str(model_path), "exec"), module.__dict__)
    return module.BASE_Transformer


def _adapt_first_conv(module: nn.Module, in_channels: int) -> None:
    original = module.resnet.conv1
    replacement = nn.Conv2d(
        in_channels,
        original.out_channels,
        kernel_size=original.kernel_size,
        stride=original.stride,
        padding=original.padding,
        bias=False,
    )
    module.resnet.conv1 = replacement


class SiameseResNet18(nn.Module):
    """Scratch ResNet-18 adapted to the eight-channel experiment input."""

    def __init__(self, in_channels: int = 8):
        super().__init__()
        backbone = resnet18(weights=None)
        original = backbone.conv1
        replacement = nn.Conv2d(
            in_channels,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=False,
        )
        backbone.conv1 = replacement
        self.backbone = backbone

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        net = self.backbone
        x = net.maxpool(net.relu(net.bn1(net.conv1(image))))
        x1 = net.layer1(x)
        x2 = net.layer2(x1)
        x3 = net.layer3(x2)
        x4 = net.layer4(x3)
        return x1, x2, x3, x4

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor):
        return self.encode(image_a), self.encode(image_b)


class RecentChangeDetector(nn.Module):
    """Expose recent official models through the common one-logit interface."""

    def __init__(self, method: str, in_channels: int = 8,
                 source_root: str | os.PathLike[str] | None = None):
        super().__init__()
        if method == "STeInFormer":
            root = Path(
                source_root
                or os.environ.get("RSCHANGE_ROOT", DEFAULT_RSCHANGE_ROOT)
            ).resolve()
            STNet = _official_steinformer_decoder(root)
            self.backbone = SiameseResNet18(in_channels=in_channels)
            self.decoder = STNet(
                num_class=2,
                channel_list=[64, 128, 256, 512],
                transform_feat=128,
                layer_num=4,
            )
            self.network = None
        elif method == "EdgeRefNet":
            root = Path(
                source_root
                or os.environ.get("EDGEREFNET_ROOT", DEFAULT_EDGEREFNET_ROOT)
            ).resolve()
            EdgeRefNet = _official_edgerefnet(root)
            self.network = EdgeRefNet(
                d_model=32,
                nhead=8,
                num_encoder_layers=1,
                dim_feedforward=64,
                dropout=0.0,
                input_nc=3,
                output_nc=2,
                token_len=4,
                resnet_stages_num=4,
                with_pos="learned",
                enc_depth=1,
                dec_depth=8,
                decoder_dim_head=8,
            )
            _adapt_first_conv(self.network, in_channels)
            self.backbone = None
            self.decoder = None
        else:
            raise ValueError(f"Unsupported recent baseline: {method}")
        self.method = method

    def forward(self, image_a: torch.Tensor,
                image_b: torch.Tensor) -> torch.Tensor:
        if self.method == "EdgeRefNet":
            edge_logits, change_logits = self.network(image_a, image_b)
            self.edge_logits = edge_logits[:, 1:2] - edge_logits[:, 0:1]
            return change_logits[:, 1:2] - change_logits[:, 0:1]

        features = self.backbone(image_a, image_b)
        logits = self.decoder(features)
        if logits.shape[1] != 2:
            raise RuntimeError(
                f"{self.method} decoder returned {logits.shape[1]} channels")
        return logits[:, 1:2] - logits[:, 0:1]

    def compute_training_loss(
        self,
        change_logits: torch.Tensor,
        target: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        if self.method != "EdgeRefNet":
            return criterion(change_logits, target)
        dilated = nn.functional.max_pool2d(target, 3, stride=1, padding=1)
        eroded = -nn.functional.max_pool2d(-target, 3, stride=1, padding=1)
        edge_target = (dilated - eroded).clamp(0.0, 1.0)
        edge_loss = nn.functional.binary_cross_entropy_with_logits(
            self.edge_logits, edge_target)
        return criterion(change_logits, target) + 5.0 * edge_loss
