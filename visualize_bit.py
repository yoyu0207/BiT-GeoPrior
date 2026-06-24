"""
BiT Change Detection Model - Comprehensive Visualization Script
================================================================
4 types of visualizations:
  1. Prediction results (input images + predicted change map + ground truth)
  2. Feature map heatmaps (intermediate backbone & transformer features)
  3. Attention map visualization (Transformer self-attention)
  4. Confusion matrix & metric statistics
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# ─── Import your project modules ───
from dataset import CDDataset
from models.bit import BiT


# =====================================================================
# 1. Hook-enabled BiT wrapper (captures features & attention)
# =====================================================================
class BiT_Hooked(BiT):
    """BiT with hooks to capture intermediate features and attention maps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = {}
        self.attention_maps = {}
        self._register_hooks()

    def _register_hooks(self):
        # Backbone feature hooks
        # layer1 = backbone[4], layer2 = backbone[5], layer3 = backbone[6]
        self.backbone[4].register_forward_hook(self._make_hook('backbone_layer1'))
        self.backbone[5].register_forward_hook(self._make_hook('backbone_layer2'))
        self.backbone[6].register_forward_hook(self._make_hook('backbone_layer3'))

        # Replace transformer attention to capture weights
        original_atn = self.transformer.atn
        self.transformer.atn = AttentionWithCapture(original_atn, self.attention_maps)

    def _make_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach().cpu()
        return hook

    def forward(self, x1, x2):
        self.features.clear()
        self.attention_maps.clear()
        return super().forward(x1, x2)


class AttentionWithCapture(nn.Module):
    """Wrapper around MultiheadAttention that captures attention weights."""
    def __init__(self, original_atn, storage):
        super().__init__()
        self.atn = original_atn
        self.storage = storage
        self._call_count = 0

    def forward(self, query, key, value):
        attn_out, attn_weights = self.atn(query, key, value, need_weights=True,
                                           average_attn_weights=False)
        label = 'attn_t1' if self._call_count % 2 == 0 else 'attn_t2'
        self.storage[label] = attn_weights.detach().cpu()
        self._call_count += 1
        return attn_out, attn_weights


# =====================================================================
# 2. Visualization Functions
# =====================================================================

def denormalize_image(img_tensor, bands=(0, 1, 32)):
    """Convert a normalized image tensor to RGB for display.
    
    Args:
        img_tensor: [C, H, W] tensor (8-channel multispectral)
        bands: which bands to use as RGB (default: bands 4,3,2 → index 3,2,1)
    """
    img = img_tensor[list(bands), :, :].numpy()
    # Per-channel min-max stretch
    for i in range(3):
        ch = img[i]
        vmin, vmax = np.percentile(ch, [2, 98])
        if vmax - vmin > 1e-6:
            img[i] = np.clip((ch - vmin) / (vmax - vmin), 0, 1)
        else:
            img[i] = 0
    return np.transpose(img, (1, 2, 0))  # [H, W, 3]


# ── 2.1 Prediction Result Visualization ──
def visualize_predictions(model, dataset, device, save_dir, num_samples=6):
    """Show input images, ground truth, and predicted change maps."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        imgA, imgB, label, spatial_prior = dataset[idx]
        imgA_dev = imgA.unsqueeze(0).to(device)
        imgB_dev = imgB.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(imgA_dev, imgB_dev)
            if isinstance(pred, list):
                pred = pred[-1]
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()

        pred_binary = (pred > 0.5).astype(np.float32)
        label_np = label.squeeze().numpy()

        # Display
        axes[row, 0].imshow(denormalize_image(imgA, bands=bands))
        axes[row, 0].set_title(f'Image T1 (Sample {idx})', fontsize=10)

        axes[row, 1].imshow(denormalize_image(imgB, bands=bands))
        axes[row, 1].set_title('Image T2', fontsize=10)

        axes[row, 2].imshow(label_np, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[row, 2].set_title('Ground Truth', fontsize=10)

        axes[row, 3].imshow(pred_binary, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[row, 3].set_title(f'Prediction (thr=0.5)', fontsize=10)

        for ax in axes[row]:
            ax.axis('off')

    plt.suptitle('Prediction Results: T1 | T2 | Ground Truth | Prediction', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Prediction visualization saved to {save_dir}/predictions.png")


# ── 2.2 Feature Map Heatmap Visualization ──
def visualize_feature_maps(model, dataset, device, save_dir, sample_idx=0):
    """Visualize backbone intermediate feature maps as heatmaps."""
    assert isinstance(model, BiT_Hooked), "Model must be BiT_Hooked for feature capture."
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    imgA, imgB, label, spatial_prior = dataset[sample_idx]
    imgA_dev = imgA.unsqueeze(0).to(device)
    imgB_dev = imgB.unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(imgA_dev, imgB_dev)

    layer_names = ['backbone_layer1', 'backbone_layer2', 'backbone_layer3']
    display_names = ['Layer1 (1/4)', 'Layer2 (1/8)', 'Layer3 (1/16)']

    fig, axes = plt.subplots(2, len(layer_names) + 1, figsize=(5 * (len(layer_names) + 1), 10))

    # Show input images in column 0
    axes[0, 0].imshow(denormalize_image(imgA, bands=bands))
    axes[0, 0].set_title('Input T1', fontsize=11)
    axes[1, 0].imshow(denormalize_image(imgB, bands=bands))
    axes[1, 0].set_title('Input T2', fontsize=11)

    for col, (lname, dname) in enumerate(zip(layer_names, display_names), start=1):
        feat = model.features.get(lname)
        if feat is None:
            continue
        # The hook fires twice (once for T1, once for T2), but since backbone
        # is shared and hooks capture last call only, we show the available one.
        # To get both, we run them separately below.
        feat_mean = feat[0].mean(dim=0).numpy()  # average over channels
        axes[0, col].imshow(feat_mean, cmap='jet')
        axes[0, col].set_title(f'{dname} (T1 pass)', fontsize=10)
        axes[0, col].axis('off')

    # Run T2 separately to capture its features
    model.features.clear()
    with torch.no_grad():
        _ = model.backbone(imgB_dev)

    for col, (lname, dname) in enumerate(zip(layer_names, display_names), start=1):
        feat = model.features.get(lname)
        if feat is None:
            continue
        feat_mean = feat[0].mean(dim=0).numpy()
        axes[1, col].imshow(feat_mean, cmap='jet')
        axes[1, col].set_title(f'{dname} (T2 pass)', fontsize=10)
        axes[1, col].axis('off')

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    plt.suptitle('Backbone Feature Maps (channel-averaged heatmaps)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_maps.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Feature map visualization saved to {save_dir}/feature_maps.png")


# ── 2.3 Attention Map Visualization ──
def visualize_attention(model, dataset, device, save_dir, sample_idx=0):
    """Visualize Transformer self-attention maps."""
    assert isinstance(model, BiT_Hooked), "Model must be BiT_Hooked for attention capture."
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    imgA, imgB, label, spatial_prior = dataset[sample_idx]
    imgA_dev = imgA.unsqueeze(0).to(device)
    imgB_dev = imgB.unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(imgA_dev, imgB_dev)

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    for row, key in enumerate(['attn_t1', 'attn_t2']):
        attn = model.attention_maps.get(key)  # [B, num_heads, L, L]
        if attn is None:
            print(f"  Warning: {key} not captured")
            continue

        attn = attn[0]  # first sample: [num_heads, L, L]
        num_heads = attn.shape[0]
        L = attn.shape[-1]
        side = int(np.sqrt(L))

        # Show first 4 heads + mean
        for h in range(min(4, num_heads)):
            attn_map = attn[h].numpy()
            # Average attention received by each token
            attn_avg = attn_map.mean(axis=0).reshape(side, side)
            axes[row, h].imshow(attn_avg, cmap='inferno')
            axes[row, h].set_title(f'{"T1" if row==0 else "T2"} Head {h}', fontsize=10)
            axes[row, h].axis('off')

        # Mean over all heads
        attn_mean = attn.mean(dim=0).numpy().mean(axis=0).reshape(side, side)
        axes[row, 4].imshow(attn_mean, cmap='inferno')
        axes[row, 4].set_title(f'{"T1" if row==0 else "T2"} Mean Attn', fontsize=11)
        axes[row, 4].axis('off')

    plt.suptitle('Transformer Self-Attention Maps (per-head & mean)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'attention_maps.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Attention map visualization saved to {save_dir}/attention_maps.png")


# ── 2.4 Confusion Matrix & Metrics ──
def visualize_metrics(model, loader, device, save_dir, model_name='BiT'):
    """Compute pixel-wise confusion matrix and metric bar chart on validation set."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgA, imgB, label, spatial_prior in tqdm(loader, desc="Computing metrics"):
            imgA, imgB = imgA.to(device), imgB.to(device)
            pred = model(imgA, imgB)
            if isinstance(pred, list):
                pred = pred[-1]
            pred = torch.sigmoid(pred).cpu().numpy().flatten()
            label_np = label.numpy().flatten()

            all_preds.append((pred > 0.5).astype(int))
            all_labels.append(label_np.astype(int))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # ── Confusion Matrix ──
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Confusion matrix heatmap
    im = axes[0].imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['No Change', 'Change'], fontsize=11)
    axes[0].set_yticklabels(['No Change', 'Change'], fontsize=11)
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Actual', fontsize=12)
    axes[0].set_title('Normalized Confusion Matrix', fontsize=13)

    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            axes[0].text(j, i, f'{cm[i,j]:,}\n({cm_norm[i,j]:.2%})',
                        ha='center', va='center', fontsize=12, color=text_color,
                        fontweight='bold')

    plt.colorbar(im, ax=axes[0], fraction=0.046)

    # Right: Metric bar chart
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    # IoU for change class
    intersection = np.logical_and(all_preds == 1, all_labels == 1).sum()
    union = np.logical_or(all_preds == 1, all_labels == 1).sum()
    iou = intersection / union if union > 0 else 0

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'IoU']
    metric_values = [acc, prec, rec, f1, iou]
    colors = ["#6FB7E7", "#83C59C", "#A5D093", "#AA95EC", "#F3C290"]

    bars = axes[1].bar(metric_names, metric_values, color=colors, width=0.6, edgecolor='white')
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title(f'{model_name} - Validation Metrics', fontsize=13)
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

    for bar, val in zip(bars, metric_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[✓] Metrics visualization saved to {save_dir}/metrics.png")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    IoU:       {iou:.4f}")


# =====================================================================
# 3. Main Entry
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="BiT Visualization Script")
    parser.add_argument('--checkpoint', type=str, default='checkpoints_BiT_0333/best_model.pth',
                        help='Path to best_model.pth')
    parser.add_argument('--data_root', type=str,
                        default=r"D:/yoyu/SA_Identification/dataset_patches_2020_2024",
                        help='Dataset root directory')
    parser.add_argument('--save_dir', type=str, default='visualizations_BiT',
                        help='Output directory for visualization images')
    parser.add_argument('--num_samples', type=int, default=6,
                        help='Number of samples for prediction visualization')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Sample index for feature/attention maps')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--bands', type=str, default='0,1,2',
                        help='Band indices for RGB display, e.g. "3,2,1"')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bands = tuple(map(int, args.bands.split(',')))
    print(f"Device: {device}")

    # ── Load model with hooks ──
    model = BiT_Hooked(in_channels=8, num_classes=1).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    
    # Handle potential key mismatches from the hook wrapper
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Note: Missing keys (expected for hook wrapper): {missing}")
    if unexpected:
        print(f"  Note: Unexpected keys: {unexpected}")
    print(f"[✓] Loaded checkpoint: {args.checkpoint}")

    # ── Dataset ──
    val_ds = CDDataset(args.data_root, split='val', split_ratio=0.85, transform=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[✓] Validation set: {len(val_ds)} samples")

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # ── Run all 4 visualizations ──
    print("\n" + "=" * 50)
    print("1/4  Prediction Results")
    print("=" * 50)
    visualize_predictions(model, val_ds, device, save_dir, num_samples=args.num_samples)

    print("\n" + "=" * 50)
    print("2/4  Feature Map Heatmaps")
    print("=" * 50)
    visualize_feature_maps(model, val_ds, device, save_dir, sample_idx=args.sample_idx)

    print("\n" + "=" * 50)
    print("3/4  Attention Maps")
    print("=" * 50)
    visualize_attention(model, val_ds, device, save_dir, sample_idx=args.sample_idx)

    print("\n" + "=" * 50)
    print("4/4  Confusion Matrix & Metrics")
    print("=" * 50)
    visualize_metrics(model, val_loader, device, save_dir, model_name='BiT')

    print("\n" + "=" * 50)
    print(f"All visualizations saved to: {save_dir}/")
    print("  - predictions.png")
    print("  - feature_maps.png")
    print("  - attention_maps.png")
    print("  - metrics.png")
    print("=" * 50)
