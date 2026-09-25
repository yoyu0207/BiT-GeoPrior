"""Check recent baseline forward and backward interfaces before long runs."""

import gc

import torch

from losses import BCEHybridLoss
from models.recent_baselines import RecentChangeDetector


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = BCEHybridLoss()
    for name in ("STeInFormer", "EdgeRefNet"):
        model = RecentChangeDetector(name).to(device).train()
        image_a = torch.randn(1, 8, 256, 256, device=device)
        image_b = torch.randn(1, 8, 256, 256, device=device)
        target = torch.randint(
            0, 2, (1, 1, 256, 256), device=device, dtype=torch.int64).float()
        output = model(image_a, image_b)
        if output.shape != target.shape:
            raise AssertionError(
                f"{name} output shape {tuple(output.shape)} != {tuple(target.shape)}")
        loss = model.compute_training_loss(output, target, criterion)
        if not torch.isfinite(loss):
            raise AssertionError(f"{name} produced non-finite loss")
        loss.backward()
        print(f"{name}: forward/backward OK; loss={loss.item():.6f}")
        del model, image_a, image_b, target, output, loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
