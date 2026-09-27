# Official SNUNet-CD versus COAST

All accuracy results use the same spatial split, five paired seeds, validation-best checkpoints, and a fixed test threshold of 0.5. Neither model uses pretrained weights.

| Metric | COAST | Official SNUNet-CD | COAST - SNUNet-CD | paired t p | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| pr_auc | 0.9741 +/- 0.0058 | 0.9849 +/- 0.0062 | -0.0108 | 0.09802 | 0.125 |
| brier_score | 0.0188 +/- 0.0032 | 0.0136 +/- 0.0039 | 0.0052 | 0.1445 | 0.1875 |
| ece_10bin | 0.0141 +/- 0.0047 | 0.0121 +/- 0.0049 | 0.0019 | 0.6489 | 0.8125 |
| fixed_0p5_f1 | 0.9128 +/- 0.0119 | 0.9387 +/- 0.0170 | -0.0259 | 0.08948 | 0.125 |
| fixed_0p5_iou | 0.8398 +/- 0.0201 | 0.8849 +/- 0.0297 | -0.0451 | 0.08778 | 0.125 |
| fixed_0p5_boundary_f1 | 0.5394 +/- 0.0335 | 0.7614 +/- 0.0463 | -0.2220 | 0.002624 | 0.0625 |
| fixed_0p5_patch_area_fraction_mae_pp | 1.4581 +/- 0.3485 | 1.2379 +/- 0.5317 | 0.2201 | 0.5541 | 0.8125 |
| fixed_0p5_patch_area_fraction_median_ae_pp | 0.8438 +/- 0.2695 | 0.3818 +/- 0.0646 | 0.4620 | 0.03487 | 0.0625 |
| fixed_0p5_patch_sampled_area_relative_bias | 0.0533 +/- 0.0486 | 0.0722 +/- 0.0418 | -0.0189 | 0.6069 | 0.625 |
| fixed_0p5_patch_area_fraction_signed_error_pp | 0.7031 +/- 0.6415 | 0.9528 +/- 0.5514 | -0.2497 | 0.6069 | 0.625 |

## Complexity

| Metric | COAST | Official SNUNet-CD |
|---|---:|---:|
| Parameters (M) | 3.114 | 12.036 |
| Profiler-counted FLOPs (G) | 28.267 | 88.227 |
| FP32 batch-1 latency (ms) | 22.53 | 42.28 |
| Throughput (images/s) | 44.38 | 23.65 |
| Peak allocated GPU memory (MB) | 496.9 | 234.6 |

## Spatial-block robustness

| Metric | COAST | Official SNUNet-CD |
|---|---:|---:|
| Positive-block macro F1 | 0.8847 | 0.9018 |
| Positive-block F1 SD across blocks | 0.0753 | 0.1084 |
| Worst positive-block F1 | 0.7990 | 0.7771 |

Spatial-block robustness is descriptive because the test set contains only three positive blocks. Profiler-counted FLOPs exclude operations unsupported by PyTorch's FLOP counter.
