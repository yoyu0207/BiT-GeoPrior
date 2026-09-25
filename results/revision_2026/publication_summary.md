# Leakage-safe revision experiments

All hyperparameters were selected using validation data only. Final values are mean +/- sample SD over five fixed random seeds on the spatially independent test split.

| Experiment | Year | Test F1 | Test IoU | n |
|---|---:|---:|---:|---:|
| OEP_BiT_SPGopt | - | 0.9053 +/- 0.0145 | 0.8272 +/- 0.0238 | 5 |
| COAST_SPGopt | - | 0.9128 +/- 0.0119 | 0.8398 +/- 0.0201 | 5 |
| COAST_shuffled_SPGopt | - | 0.9074 +/- 0.0190 | 0.8309 +/- 0.0315 | 5 |
| COAST_random_SPGopt | - | 0.9147 +/- 0.0156 | 0.8432 +/- 0.0260 | 5 |
| COAST_no_gating_SPGopt | - | 0.9076 +/- 0.0159 | 0.8311 +/- 0.0263 | 5 |
