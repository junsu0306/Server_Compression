# Patch Slimming experiment results

## 2026-08-04: DeiT-Tiny end-to-end smoke experiment

This experiment verifies the complete static Patch Slimming pipeline before the
full 2,048-sample search. It is a pipeline result, not a final accuracy result.

### Environment

- GPU: NVIDIA GeForce RTX 2080 Ti (single GPU)
- PyTorch: 2.4.0+cu118
- timm: 1.0.14
- Dataset: ImageNet-1K train/validation
- Cache/temp root: repository `.cache/` and `.tmp/` on the workspace volume

### Search configuration

- Model: `deit_tiny_patch16_224` (pretrained)
- Calibration: 32 deterministic training samples, seed 2022
- Score: path-energy DP
- Error metric / epsilon: MSE / 0.02
- Search step: 20 patches
- Candidate reconstruction: AdamW, LR 1e-5, one epoch
- Config: `configs/reference/deit_tiny_ps_smoke.yaml`

### Results

- Token schedule: `[197, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 1]`
- Baseline ImageNet val: Top-1 71.36%, Top-5 90.71%
- Searched compact model before final fine-tuning: Top-1 59.28%, Top-5 82.54%
- Accuracy delta: -12.08 percentage points Top-1
- Estimated transformer MAC reduction: 64.04% (1.225G -> 0.440G)
- RTX 2080 Ti batch-1 latency: full 8.63 ms, current Python compact 19.98 ms
  - The unfused GPU reference is slower despite fewer MACs because each fused QKV
    is split into separate projections and small operators dominate. This is not
    an NPU latency result.

Layer 10 required 81 output tokens to pass epsilon (MSE 0.01092). Layers 9 through
0 accepted the same nested mask in this small calibration run. The final block
outputs CLS only.

### ONNX export

- Path: `output/ps_deit_tiny_smoke/ps_compact_npu.onnx`
- Input: static NHWC `[1, 224, 224, 3]`
- PyTorch vs ONNX Runtime max absolute error: 1.97e-06
- Forbidden operators absent: Gather, GatherElements, ScatterElements, TopK, Equal
- Operator set includes rectangular-attention MatMul and Softmax.

The initial export still contained Gather and Equal from the classifier CLS row
selection and timm's dynamic CLS expansion. A fixed-batch NPU export path now
uses direct CLS concatenation and constant-axis squeeze, removing those operators.

### Interpretation and next experiment

The full pipeline is operational and produces a genuinely compact static graph.
The 32-sample/one-epoch search loses too much accuracy for a final model. The next
quality run should use `configs/reference/deit_tiny_ps.yaml` (2,048 samples, step 10, three
reconstruction epochs), followed by final classification fine-tuning and NPU
latency measurement.
