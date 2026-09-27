# Experiment Fairness Audit

## Verdict

The comparison is broadly fair at the data, budget, seed, and checkpoint-selection levels. It is not an identical-optimizer study: method-specific optimisation and loss differences are present and must be reported. The main high-risk issue is the identity of the local SNUNet baseline.

## Findings

| Severity | Topic | Finding |
|---|---|---|
| HIGH | baseline identity | The local implementation is a custom Siamese Nested U-Net-like baseline, not a faithful implementation of the official SNUNet-CD/ECAM architecture. Rename it in the manuscript or rerun the official implementation before making SNUNet-CD claims. |
| MEDIUM | training objective | EdgeRefNet uses its method-specific auxiliary boundary loss (BCE+Dice change loss plus 5x edge BCE), while other baselines use BCE+Dice. This is defensible as architecture-specific training, but must be disclosed. |
| MEDIUM | optimisation protocol | Learning rates and schedulers are not identical: ChangeFormer uses 1e-4 with cosine annealing; recent baselines use a validation-selected 5e-4; COAST uses separate SPG rates. Describe the protocol as validation-tuned, not identical. |
| LOW | precision mode | Most runs use FP16 AMP, while EdgeRefNet uses BF16 because FP16 was unstable. Evaluation should use FP32 for every model. |
| PASS | core comparison | All formal runs use the same 8-channel inputs, canonical 313/59/71 spatial split, batch size 8, 200-epoch budget, patience 30, validation-selected checkpoint, and five seeds. |
| PASS | initialisation | Formal models are trained from scratch without pretrained weights. |

## Formal Run Checks

| Model | Runs | Canonical split | 313/59/71 | Budget | Scratch |
|---|---:|---:|---:|---:|---:|
| FCSiamDiff | 5/5 | yes | yes | yes | yes |
| Siamese Nested U-Net (local) | 5/5 | yes | yes | yes | yes |
| BiT | 5/5 | yes | yes | yes | yes |
| ChangeFormer | 5/5 | yes | yes | yes | yes |
| STeInFormer | 5/5 | yes | yes | yes | yes |
| EdgeRefNet | 5/5 | yes | yes | yes | yes |
| COAST | 5/5 | yes | yes | yes | yes |

## Required Manuscript Wording

- Call the local baseline `Siamese Nested U-Net (local implementation)`, not official `SNUNet-CD`, unless the official ECAM implementation is rerun.
- State that learning rates were selected on the validation set and the test set was not loaded during tuning.
- State the architecture-specific scheduler and EdgeRefNet auxiliary edge loss explicitly.
- Keep fixed-threshold 0.5 results as the primary comparison; validation-selected-threshold results are supplementary.
- Report every metric as mean +/- sample SD across the same five seeds.

## Provenance

- Canonical manifest SHA-256: `6d41dcd222305928e06fc503d83fe2904c329aa1b06b2c5e8ef40283881f8af7`
- Audit generated from: `D:\yoyu\SA_Identification\project`
- Official SNUNet source checked: https://github.com/likyoo/Siam-NestedUNet
