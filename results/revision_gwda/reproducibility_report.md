# Leakage-safe GWDA revision experiment audit

## Completion and reproducibility

- Completed runs: 70/70 (14 experiments x 5 seeds: 42, 1337, 2025, 3407, 9001).
- Spatial split: 313 train, 59 validation, and 71 test patches; 49 buffer patches excluded.
- Buffered cross-split conflicts: 0.
- GWDA teacher fit: training split only, using 313 source patches and 43904 unique sampled pixels.
- Revision prior experiments: GWDA only; no GWR experiment is present.
- Alpha sensitivity: alpha = 0.05 and 0.20 used validation only; test metrics and test-loader counts are null. Alpha = 0.10 is the prespecified main COAST setting.

## Main result

Across five seeds, COAST achieved test F1 = 0.9104 +/- 0.0135 and IoU = 0.8358 +/- 0.0224. Full baseline, control, sensitivity, and paired-comparison results are provided in the adjacent CSV and JSON files.

## Interpretation guardrails

- COAST and OEP-BiT were statistically indistinguishable in the paired five-seed tests.
- COAST exceeded the shuffled-prior control on mean F1 and IoU, but the paired t and Wilcoxon tests did not reach p < 0.05 with five seeds.
- The random-prior and no-gating controls had higher mean scores than COAST in this run. The manuscript must report this result directly and avoid claiming that the learned gate or GWDA map alone improves peak segmentation accuracy.
- SNUNet had the highest mean F1 and IoU among the listed main baselines. The revised claim should focus on scalable online-prior inference and comparable accuracy, not universal state-of-the-art accuracy.
