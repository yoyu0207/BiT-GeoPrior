# Application-facing evaluation protocol

- All models are loaded from their validation-best checkpoints and evaluated in FP32.
- The fixed probability threshold of 0.5 is the primary comparison.
- PR-AUC is threshold-free and is computed over all independent-test pixels.
- A supplementary threshold is selected from 0.05 to 0.95 (step 0.01) using validation F1 only, then frozen before test evaluation.
- Boundary F1 uses a symmetric tolerance of 2 pixels. At 10 m Sentinel-2 resolution this corresponds to approximately 20 m.
- Area metrics are patch-sampled area-fraction errors. They are not hectares and must not be interpreted as unique mapped area because neighbouring patches overlap.
- Per-region metrics aggregate pixels within the manifest's region field and are descriptive because region sample sizes differ.
- Efficiency uses FP32, batch size 1, two 8-channel 256 x 256 inputs, 5 warm-up iterations and 20 timed iterations on the same device.
- Results are reported as mean +/- sample SD across the same five random seeds.
- Recomputed legacy F1/IoU values must agree with stored values within an absolute tolerance of 0.0002; this tolerance covers sub-threshold CUDA-kernel numerical variation and is far below the reported precision.
