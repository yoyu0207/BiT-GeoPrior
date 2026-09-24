# BiT-GeoPrior

<div align="center">

**Bi-temporal Transformer with Ecologically-informed Geographic Prior for Coastal Wetland Change Detection**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

</div>

## Abstract

Accurate monitoring of *Spartina alterniflora* (cordgrass) dynamics in coastal wetlands is critical for ecological conservation, yet remains challenging due to spectral confusion and tidal complexity. We propose **BiT-GeoPrior**, a bi-temporal change detection framework that integrates geographic prior knowledge into deep neural networks via a lightweight, zero-initialized **Spatial Prior Gate (SPG)**. The framework supports both **static priors** (GWR-based local *R*², GWDA posterior probability) and an **online ecological prior encoder** that estimates prior maps end-to-end from satellite imagery—eliminating the need for precomputed auxiliary data.

## Architecture

<div align="center">
  <img src="assets/architecture.png" alt="BiT-GeoPrior Architecture" width="90%">
  <p><em>Overall architecture: Bi-temporal Transformer with Spatial Prior Gate injection.</em></p>
</div>

### Spatial Prior Gate (SPG)

The SPG is a zero-initialized residual attention gate that injects prior knowledge into intermediate feature maps:

<div align="center">
  <img src="assets/spg_module.svg" alt="Spatial Prior Gate" width="50%">
  <p><em>SPG: residual channel attention with zero-initialized learnable gain γ. At initialization, the model is strictly equivalent to a prior-free baseline.</em></p>
</div>

$$
\mathbf{F}_{out} = \mathbf{F} + \gamma \cdot (\mathbf{F} \odot \sigma(\text{Conv}_{1\times1}(\mathbf{P})))
$$

Key properties:
- **Identity at initialization** — γ starts at 0, ensuring training begins identically to the baseline
- **Plug-and-play** — can be inserted into any Siamese change detection network
- **Interpretable γ** — the learned gain reveals how strongly the model relies on prior knowledge

## Models

| Model | Prior Type | Description |
|-------|-----------|-------------|
| `SNUNet` | None | Siamese Nested U-Net baseline (ECCV 2020) |
| `SNUNet_GeoAware` | Static | SNUNet + SPG at shallow decoder layers |
| `FCSiamDiff_Aligned` | None | Fully Convolutional Siamese Difference |
| `BiT` | None | Bi-temporal Transformer (ResNet-18 + Transformer Encoder) |
| `BiT_GWR` | Static GWR | BiT + GWR local-*R*² prior |
| `BiT_GWDA` | Static GWDA | BiT + GWDA posterior probability prior |
| `BiT_Online` | **Online** | BiT + learnable ecological prior encoder (no static files) |
| `ChangeFormer` | None | Hierarchical Transformer |

## Results

### Qualitative Comparison

<div align="center">
  <img src="assets/qualitative_results.png" alt="Qualitative Comparison" width="100%">
  <p><em>Qualitative comparison across representative coastal wetland scenes. BiT_Online consistently reduces false positives in spectrally ambiguous regions.</em></p>
</div>

### Ablation Study

<div align="center">
  <img src="assets/ablation_comparison.png" alt="Ablation Study" width="70%">
  <p><em>Ablation results: prior injection consistently improves F1 across all base architectures.</em></p>
</div>

### Training Dynamics

<div align="center">
  <img src="assets/training_curves.png" alt="Training Curves" width="70%">
  <p><em>Training and validation curves. Left: loss convergence. Right: F1 progression.</em></p>
</div>

### Estuary Transfer Performance

<div align="center">
  <img src="assets/transfer_results.png" alt="Transfer Results" width="70%">
  <p><em>Spatial transfer across different estuaries demonstrates generalization capability.</em></p>
</div>

## Installation

```bash
git clone https://github.com/yoyu0207/BiT-GeoPrior.git
cd BiT-GeoPrior
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision ≥ 0.15

## Dataset Structure

```
data_root/
├── A/                     # T1 Sentinel-2 patches (.npy, [8, H, W])
├── B/                     # T2 Sentinel-2 patches (.npy, [8, H, W])
├── label/                 # Binary change labels (.npy or .png)
├── spatial_prior_gwr/     # (optional) GWR prior patches [0, 1]
├── spatial_prior_gwda/    # (optional) GWDA prior patches [0, 1]
└── spatial_split_manifest.csv
```

8-channel composition: **B8, B4, B3, B2, NDVI, EVI, SAVI, GNDVI**

### Spatially independent split

Overlapping image patches must not be randomly divided between training and
validation sets. Generate an explicit spatial manifest before training:

```bash
python make_spatial_split.py \
  --data_root /path/to/data_root \
  --block_size 2048 \
  --patch_size 256 \
  --buffer 256 \
  --seed 42
```

For 10 m Sentinel-2 imagery, this configuration uses approximately 20.48 km
spatial blocks and a 2.56 km exclusion buffer between train, validation, and
test patches. Samples in the buffer are retained in the manifest as
`excluded` for auditability. The validation set is used for checkpoint
selection; the test set is evaluated only after training.

### Leakage-safe GWDA teacher

The revised GWDA workflow fits the teacher exclusively with samples from the
training spatial blocks. Validation and test labels are not used for feature
standardisation, adaptive-bandwidth selection, model fitting, or probability
calibration:

```bash
python build_gwda_prior.py \
  --data_root /path/to/data_root \
  --output_dir /path/to/data_root/spatial_prior_gwda_train_only
```

The command records the fitting sample provenance, spatial-block
cross-validation results, selected adaptive Gaussian bandwidth, isotonic
calibration parameters, and teacher-only diagnostics in
`gwda_metadata.json`. The new revision experiments do not use GWR priors.

### Multi-seed revision experiments

Run the complete five-seed baseline and control matrix with:

```bash
python run_revision_experiments.py \
  --python /path/to/python \
  --data_root /path/to/data_root \
  --epochs 200 \
  --patience 30
```

The matrix includes FC-SiamDiff, SNUNet, ChangeFormer, BiT, train-only
BiT-GWDA, OEP-BiT, COAST, shuffled-prior, random-prior, no-gating, and
distillation-weight sensitivity controls. It writes per-run metadata,
mean/standard deviation summaries, paired bootstrap confidence intervals,
paired t-tests, and Wilcoxon tests. Distillation-weight sensitivity runs use
the validation set only and do not access the test set.

## Quick Start

### Training

```bash
# Baseline (no prior)
python train.py --model SNUNet --epochs 200
python train.py --model BiT --lr 6e-5 --epochs 200
python train.py --model ChangeFormer --epochs 200

# With static prior (requires spatial_prior_gwr/ or spatial_prior_gwda/)
python train.py --model BiT_GWR --lr 6e-5 --epochs 200
python train.py --model BiT_GWDA --lr 6e-5 --epochs 200

# With online prior (no static files needed)
python train.py --model BiT_Online --lr 6e-5 --epochs 200
```

Training uses `data_root/spatial_split_manifest.csv` by default. Set
`--split_manifest` to use another manifest and `--seed` for repeated runs.
The legacy random patch split is available only through the explicit
`--allow_random_patch_split` flag and should not be used for independent
spatial validation.

### Evaluation

```bash
python evaluate.py \
  --model BiT_GWR \
  --pth checkpoints/BiT_GWR/best_model.pth \
  --split test
```

## Project Structure

```
├── models/
│   ├── __init__.py
│   ├── snunet.py                  # SNUNet & SNUNet_GeoAware
│   ├── FC_Siam_diff.py            # FC-Siam-Diff
│   ├── bit.py                     # BiT baseline
│   ├── bit_gwr.py                 # BiT + GWR prior
│   ├── bit_gwda.py                # BiT + GWDA prior
│   ├── bit_online.py              # BiT + online prior
│   ├── changeformer.py            # ChangeFormer
│   ├── SPGmodule.py               # Spatial Prior Gate
│   ├── ecological_prior.py        # Online prior encoder
│   └── transformer_block.py       # Shared Transformer block
├── train.py                       # Training entry point
├── dataset.py                     # Data loader with augmentation
├── make_spatial_split.py          # Leakage-safe spatial split generator
├── losses.py                      # BCE + Dice hybrid loss
├── utils.py                       # Metric tracker (IoU, F1, etc.)
├── evaluate.py                    # Standalone evaluation
├── splits/                        # Versioned split manifests and summaries
├── assets/                        # README figures
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

```bibtex
@article{...,
  title     = {...},
  author    = {...},
  journal   = {...},
  year      = {2025}
}
```

## License

This project is released under the [MIT License](LICENSE).
