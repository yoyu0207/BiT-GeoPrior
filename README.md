# BiT-GeoPrior

**Bi-temporal Transformer with Geographic Prior for Spartina alterniflora Change Detection**

PyTorch implementation of the models described in our paper.

## Overview

Deep learning models for detecting Spartina alterniflora (cordgrass) expansion and eradication using dual-temporal Sentinel-2 satellite imagery. The core contribution is a **Spatial Prior Gate (SPG)** module that injects geographic/ecological prior knowledge into deep change detection networks.

## Models

| Model | File | Description |
|-------|------|-------------|
| `SNUNet` | `models/snunet.py` | Siamese Nested U-Net baseline |
| `SNUNet_GeoAware` | `models/snunet.py` | SNUNet with SPG spatial prior injection |
| `FCSiamDiff_Aligned` | `models/FC_Siam_diff.py` | Fully Convolutional Siamese Difference |
| `BiT` | `models/bit.py` | Bi-temporal Transformer (ResNet-18 + Transformer) |
| `BiT_GWR` | `models/bit_gwr.py` | BiT with static GWR spatial prior |
| `BiT_GWDA` | `models/bit_gwda.py` | BiT with static GWDA spatial prior |
| `BiT_Online` | `models/bit_online.py` | BiT with online ecological prior encoder |
| `ChangeFormer` | `models/changeformer.py` | Hierarchical Transformer |

### Key Modules

| Module | File | Description |
|--------|------|-------------|
| `SpatialPriorGate` | `models/SPGmodule.py` | Zero-init residual channel attention gate |
| `EcologicalPriorEncoder` | `models/ecological_prior.py` | Online prior estimation from T1 imagery |
| `TransformerBlock` | `models/transformer_block.py` | Pre-LayerNorm Transformer Encoder Block |

## Installation

```bash
git clone https://github.com/your-username/BiT-GeoPrior.git
cd BiT-GeoPrior
pip install -r requirements.txt
```

## Dataset Structure

```
data_root/
  A/                  # T1 image patches (.npy, [C, H, W])
  B/                  # T2 image patches (.npy)
  label/              # Binary change labels (.npy or .png)
  spatial_prior_gwr/  # (optional) GWR spatial prior patches
  spatial_prior_gwda/ # (optional) GWDA spatial prior patches
```

Each patch contains 8 Sentinel-2 bands: B8, B4, B3, B2, NDVI, EVI, SAVI, GNDVI.

## Training

```bash
# Baseline (no prior)
python train.py --model SNUNet --epochs 200
python train.py --model BiT --lr 6e-5 --epochs 200
python train.py --model ChangeFormer --epochs 200
python train.py --model FCSiamDiff --epochs 200

# Static prior
python train.py --model SNUNet_GeoAware --epochs 200
python train.py --model BiT_GWR --lr 6e-5 --epochs 200
python train.py --model BiT_GWDA --lr 6e-5 --epochs 200

# Online prior
python train.py --model BiT_Online --lr 6e-5 --epochs 200
```

## Evaluation

```bash
python evaluate.py --model BiT_GWR --pth checkpoints/BiT_GWR/best_model.pth
```

## Citation

```
@article{...,
  title={...},
  author={...},
  journal={...},
  year={2025}
}
```

## License

MIT License — see [LICENSE](LICENSE).
