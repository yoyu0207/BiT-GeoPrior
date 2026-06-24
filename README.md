# BiT-GeoPrior

**Bi-temporal Transformer with Geographic Prior for Spartina alterniflora Change Detection**

PyTorch implementation of the models described in our paper on ecologically-informed remote sensing change detection.

## Overview

This repository provides deep learning models for detecting Spartina alterniflora (cordgrass) expansion and eradication using dual-temporal Sentinel-2 satellite imagery. The core contribution is a **Spatial Prior Gate (SPG)** module that injects geographic/ecological prior knowledge into deep change detection networks, improving predictions in ecologically complex coastal wetlands.

## Models

| Model | Description |
|-------|-------------|
| `SNUNet` | Siamese Nested U-Net baseline (ECCV 2020) |
| `SNUNet_GeoAware` | SNUNet with SPG spatial prior injection at shallow decoder layers |
| `FCSiamDiff_Aligned` | Fully Convolutional Siamese Difference network |
| `BiT` | Bi-temporal Transformer baseline (ResNet-18 + Transformer Encoder) |
| `BiT_GWR` | BiT with static GWR spatial prior |
| `BiT_GWDA` | BiT with static GWDA spatial prior |
| `BiT_Online` | BiT with online ecological prior encoder (no static prior files needed) |
| `ChangeFormer` | Hierarchical Transformer for change detection |

### Key Modules

| Module | Description |
|--------|-------------|
| `SPGmodule.py` | Spatial Prior Gate: zero-initialized residual channel attention |
| `ecological_prior.py` | EcologicalPriorEncoder: online prior estimation from T1 imagery |
| `transformer_block.py` | Pre-LayerNorm Transformer Encoder Block |

## Installation

```bash
git clone https://github.com/your-username/BiT-GeoPrior.git
cd BiT-GeoPrior
pip install -r requirements.txt
```

## Dataset Structure

The dataset should be organized as:

```
data_root/
  A/                  # T1 image patches (.npy, shape [C, H, W])
  B/                  # T2 image patches (.npy)
  label/              # Binary change labels (.npy or .png)
  spatial_prior_gwr/  # (optional) GWR spatial prior patches
  spatial_prior_gwda/ # (optional) GWDA spatial prior patches
```

Each patch contains 8 Sentinel-2 bands: B8, B4, B3, B2, NDVI, EVI, SAVI, GNDVI.

## Training

```bash
# Baseline models (no prior)
python train.py --model SNUNet --epochs 200
python train.py --model BiT --lr 6e-5 --epochs 200
python train.py --model ChangeFormer --epochs 200
python train.py --model FCSiamDiff --epochs 200

# Static prior models (requires spatial_prior_gwr/ or spatial_prior_gwda/)
python train.py --model SNUNet_GeoAware --epochs 200
python train.py --model BiT_GWR --lr 6e-5 --epochs 200
python train.py --model BiT_GWDA --lr 6e-5 --epochs 200

# Online prior model (no static prior files needed)
python train.py --model BiT_Online --lr 6e-5 --epochs 200
```

## Evaluation

```bash
python evaluate.py --model BiT_GWR --pth checkpoints_BiT_GWR_xxx/best_model.pth
```

## Loss Functions

- `BCEHybridLoss` — BCE + Dice hybrid loss (main training loss)
- `HybridLoss` — BCEWithLogitsLoss with positive class weighting
- `EdgeAwareHybridLoss` — Spatially-aware BCE + Dice with dampened edge weighting

## Citation

If you use this code in your research, please cite our paper:

```
@article{...,
  title={...},
  author={...},
  journal={...},
  year={2025}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
