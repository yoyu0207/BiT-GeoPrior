# Application-facing metrics

Fixed-threshold 0.5 results, mean +/- sample SD over five seeds.

| Model | F1 | PR-AUC | Boundary F1 | Patch area MAE (pp) | Brier | Params (M) | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCSiamDiff | 0.9111 +/- 0.0216 | 0.9702 +/- 0.0100 | 0.6612 +/- 0.0536 | 1.517 +/- 0.432 | 0.0663 +/- 0.0289 | 7.85 | 11.85 |
| SiameseNestedUNet_local | 0.9226 +/- 0.0163 | 0.9780 +/- 0.0047 | 0.6826 +/- 0.0574 | 1.481 +/- 0.545 | 0.0580 +/- 0.0210 | 9.16 | 22.06 |
| BiT | 0.8864 +/- 0.0572 | 0.9567 +/- 0.0337 | 0.5281 +/- 0.0534 | 2.254 +/- 1.786 | 0.0249 +/- 0.0144 | 3.11 | 18.86 |
| ChangeFormer | 0.8518 +/- 0.0092 | 0.9215 +/- 0.0049 | 0.5557 +/- 0.0354 | 3.149 +/- 0.311 | 0.0307 +/- 0.0020 | 0.53 | 10.58 |
| STeInFormer | 0.9097 +/- 0.0388 | 0.9772 +/- 0.0108 | 0.7450 +/- 0.0864 | 2.024 +/- 1.217 | 0.0192 +/- 0.0083 | 15.14 | 21.60 |
| EdgeRefNet | 0.7037 +/- 0.2298 | 0.7773 +/- 0.2607 | 0.4947 +/- 0.2316 | 13.036 +/- 14.255 | 0.0805 +/- 0.0710 | 47.10 | 63.97 |
| COAST | 0.9128 +/- 0.0119 | 0.9741 +/- 0.0058 | 0.5394 +/- 0.0335 | 1.458 +/- 0.349 | 0.0188 +/- 0.0032 | 3.11 | 22.53 |

Interpretation: boundary F1 measures delineation quality; patch area MAE measures absolute error in changed-area fraction per overlapping test patch; Brier measures probability calibration (lower is better). Patch area values are not unique-area hectares.
