# Patch Slimming main experiment log

This is the live record for the full DeiT-Tiny Patch Slimming experiment. It is
updated at each completed phase and whenever an implementation or runtime issue
changes the experiment.

## 1. Context and completed preflight work

Before starting the main run, the complete pipeline was exercised with the
32-sample smoke configuration. Detailed numbers are in
[`01_SMOKE_EXPERIMENT.md`](01_SMOKE_EXPERIMENT.md).

Preflight conclusions:

- Search, compact evaluation, and ONNX Runtime verification work end to end.
- The smoke schedule was `[197, 81, ..., 81, 1]`.
- Smoke Top-1 was 59.28% before final fine-tuning versus a 71.36% baseline.
- Estimated transformer MAC reduction was 64.04%.
- NPU export initially contained `Gather` and `Equal`; the static batch-one path
  was corrected and the final smoke ONNX contained none of the forbidden ops.
- PyTorch/ORT maximum absolute difference was 1.97e-06.
- Thirty-one unit/integration tests pass, including GPU candidate-device handling,
  reconstruction gradient routing, resume, compact shapes, and export-path
  equivalence.

## 2. Main experiment definition

Start time: 2026-08-04 14:46:50 UTC

- Git base revision: `fbe934c2724dd898cad1e8ee0db8597c39b8fc35`
- Configuration: `configs/reference/deit_tiny_ps.yaml`
- Model: pretrained `deit_tiny_patch16_224`
- Dataset: ImageNet-1K
- Calibration samples: 2,048 deterministic train images
- Calibration seed: 2022
- Calibration batch size: 8
- Score mode: `paper_path_energy_dp`
- Acceptance metric: MSE
- Epsilon: 0.02
- Search step: 10 patches
- Candidate reconstruction: AdamW, LR 1e-5, weight decay 0, 3 epochs
- Candidate policy: cumulative fine-tuning
- Output directory: `output/ps_deit_tiny`
- Search GPU: physical GPU 3 (RTX 2080 Ti)
- Cache/temp: repository `.cache/` and `.tmp/` on the workspace volume
- Installed runtime: PyTorch 2.4.0+cu118, timm 1.0.14
- Final fine-tuning GPUs: physical GPUs 3 through 7, batch/GPU 102, global batch 510

The installed PyTorch/timm versions differ from the versions recorded in the
older root `IMPLEMENTATION.md` (2.9.1/1.0.27). The tested Patch Slimming gates
pass in the installed environment, so this run records the actual versions above.

## 3. Phase status

| Phase | Status | Result |
|---|---|---|
| Preflight tests | Complete | 31/31 passed |
| Full top-down search | Complete | Schedule `[197 x10, 91, 41, 1]` including input |
| Searched compact evaluation | Complete | Top-1 61.19%, Top-5 84.48% |
| Final classification fine-tuning | Complete | 30 epochs, 5 GPUs; best epoch 27 |
| Final compact evaluation | Complete | Top-1 72.82%, Top-5 91.26% |
| NPU-safe ONNX export | Complete | ORT max error 3.10e-06; forbidden ops absent |

### Final fine-tuning validation history

| Epoch | Learning rate at start | Val Top-1 | Val Top-5 | Val loss | Notes |
|---:|---:|---:|---:|---:|---|
| 0 | 2.00e-7 | 66.69% | 88.05% | 1.4216 | Training ~701 s; first best checkpoint |
| 1 | 6.80e-6 | 70.29% | 89.95% | 1.2701 | New best; baseline gap reduced to 1.07 pp |
| 2 | 1.34e-5 | 70.79% | 90.23% | 1.2458 | New best; baseline gap reduced to 0.57 pp |
| 3 | 2.00e-5 | 71.06% | 90.40% | 1.2325 | New best; baseline gap reduced to 0.30 pp |
| 4 | 2.00e-5 | 71.31% | 90.46% | 1.2262 | New best; only 0.05 pp below baseline |
| 5 | 1.97e-5 | 71.48% | 90.46% | 1.2176 | New best; exceeds baseline by 0.12 pp |
| 6 | 1.94e-5 | 71.59% | 90.65% | 1.2109 | New best; exceeds baseline by 0.23 pp |
| 7 | 1.90e-5 | 71.70% | 90.66% | 1.2068 | New best; exceeds baseline by 0.34 pp |
| 8 | 1.84e-5 | 71.81% | 90.84% | 1.1982 | New best; exceeds baseline by 0.45 pp |
| 9 | 1.78e-5 | 71.88% | 90.86% | 1.1927 | New best; exceeds baseline by 0.52 pp |
| 10 | 1.70e-5 | 71.91% | 90.79% | 1.1942 | New Top-1 best |
| 11 | 1.62e-5 | 72.05% | 90.88% | 1.1894 | New best |
| 12 | 1.53e-5 | 72.12% | 90.91% | 1.1864 | New best; exceeds baseline by 0.76 pp |
| 13 | 1.43e-5 | 72.24% | 90.95% | 1.1821 | New best |
| 14 | 1.32e-5 | 72.42% | 90.97% | 1.1770 | New Top-1 best; +1.06 pp vs baseline |
| 15 | 1.21e-5 | 72.33% | 91.02% | 1.1747 | Top-1 below epoch 14; best Top-5 so far |
| 16 | 1.11e-5 | 72.37% | 91.09% | 1.1738 | Below epoch-14 Top-1, higher Top-5 |
| 17 | 9.95e-6 | 72.55% | 91.08% | 1.1710 | New Top-1 best |
| 18 | 8.85e-6 | 72.58% | 91.11% | 1.1661 | New best; +1.22 pp vs baseline |
| 19 | 7.78e-6 | 72.61% | 91.10% | 1.1686 | New Top-1 best |
| 20 | 6.74e-6 | 72.69% | 91.10% | 1.1667 | New Top-1 best; +1.33 pp vs baseline |
| 21 | 5.75e-6 | 72.66% | 91.18% | 1.1631 | Top-1 below epoch 20; best Top-5 so far |
| 22 | 4.83e-6 | 72.72% | 91.13% | 1.1640 | New Top-1 best |
| 23 | 3.98e-6 | 72.81% | 91.23% | 1.1619 | New best; +1.45 pp vs baseline |
| 24 | 3.22e-6 | 72.77% | 91.17% | 1.1596 | Below epoch-23 best |
| 25 | 2.56e-6 | 72.77% | 91.21% | 1.1585 | Stable Top-1, improved Top-5 |
| 26 | 2.01e-6 | 72.83% | 91.23% | 1.1598 | New Top-1 best |
| 27 | 1.57e-6 | 72.83% | 91.26% | 1.1569 | Best checkpoint; best Top-5 |
| 28 | 1.26e-6 | 72.81% | 91.21% | 1.1576 | Below epoch-27 best |
| 29 | 1.06e-6 | 72.81% | 91.22% | 1.1563 | Final epoch |

## 4. Full-search layer results

Results will be appended after each completed layer checkpoint. The authoritative
machine-readable records are `output/ps_deit_tiny/search_records.jsonl` and
`search_last.pt`.

| Layer | Kept tokens | Accepted MSE | Candidate count/notes |
|---:|---:|---:|---|
| 10 | 41 | 0.01500 | r=0/10/20/30/40; MSE 0.22130/0.07926/0.03378/0.02176/0.01500 |
| 9 | 91 | 0.01900 | Base 41 + r=50; candidate MSE fell 0.07459 -> 0.01900 |
| 8 | 197 | 0.02603 | **All-token fallback**; base 91, all 106 candidates added, epsilon not reached |
| 7 | 197 | 0.11942 | **All-token fallback**; base already all tokens, epsilon not reached |
| 6 | 197 | 0.01944 | Base already all tokens; epsilon passed |
| 5 | 197 | 0.00862 | Base already all tokens; epsilon passed |
| 4 | 197 | 0.00410 | Base already all tokens; epsilon passed |
| 3 | 197 | 0.00315 | Base already all tokens; epsilon passed |
| 2 | 197 | 0.00225 | Base already all tokens; epsilon passed |
| 1 | 197 | 0.00208 | Base already all tokens; epsilon passed |
| 0 | 197 | 0.00217 | Base already all tokens; epsilon passed |

## 5. Runtime issues and decisions

- No pre-existing `output/ps_deit_tiny` directory was present at start, so this
  run does not overwrite a previous main experiment.
- Large files are kept on the workspace mount because the container overlay is full.
- Layer 10 completed successfully. Compared with the smoke run (81 tokens), the
  full calibration and three reconstruction epochs accepted 41 tokens, showing
  that the smoke architecture was not a stable estimate of the final schedule.
- Layer 8 could not meet epsilon even with all 197 tokens. MSE decreased
  monotonically from 0.06073 at 91 tokens to 0.02603 at 197 tokens. The search
  followed SPEC §8.7 and selected the all-token fallback. This is not recorded as
  an epsilon pass. A likely contributor is cumulative fine-tuning of deeper
  student blocks: the current block must compensate for already-modified block 9
  while matching the original teacher's block-9 output.
- Layer 7 also failed epsilon with all tokens (MSE 0.11942). From layer 6 onward
  the errors passed epsilon, but the nested-mask invariant prevents removing a
  token once layer 8 has fallen back to all tokens. The final search schedule is
  therefore `[197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 91, 41, 1]`
  including the model input followed by twelve block outputs.
- A two-GPU run initially started at 2026-08-04 15:04 UTC with global batch 512,
  but early throughput projected roughly nine hours for 30 epochs. It was stopped
  during epoch 0 before any checkpoint was produced. The definitive run uses five
  available GPUs with batch/GPU 102 (global batch 510). This preserves the global
  batch and approximately 2,500 optimizer steps per epoch while reducing wall time.
  `finetune_compact.py` now exposes `--batch-size` for this explicit override.
- The definitive five-GPU epoch 0 completed successfully. Validation Top-1
  improved from the pre-fine-tuning 61.19% to 66.69% (+5.50 percentage points).
- Epoch 1 reached 70.29% Top-1, only 1.07 percentage points below the 71.36%
  pretrained baseline while retaining the searched static token schedule.
- Epoch 2 reached 70.79% Top-1. Warmup then completed and epoch 3 started at the
  configured 2e-5 learning rate. PyTorch emitted the known scheduler epoch-API
  deprecation warning; no training error or numerical issue accompanied it.
- Epoch 3 reached 71.06% Top-1 at the full learning rate, recovering to within
  0.30 percentage points of the original pretrained model.
- Epoch 4 reached 71.31% Top-1, effectively recovering the baseline within
  0.05 percentage points. Cosine decay begins from epoch 5.
- Epoch 5 reached 71.48% Top-1, exceeding the measured pretrained baseline by
  0.12 percentage points for the first time.
- Epoch 6 reached 71.59% Top-1 and 90.65% Top-5, improving the best result again.
- Epoch 7 reached 71.70% Top-1 and 90.66% Top-5, continuing the gradual gain
  during cosine decay.
- Epoch 8 reached 71.81% Top-1 and 90.84% Top-5, setting another best checkpoint.
- Epoch 9 reached 71.88% Top-1 and 90.86% Top-5, setting another best checkpoint.
- Epochs 10 through 12 continued improving Top-1 to 71.91%, 72.05%, and 72.12%.
  Epoch 12 exceeds the measured pretrained baseline by 0.76 percentage points.
- Epochs 13 and 14 improved Top-1 to 72.24% and 72.42%. Epoch 15 had a slightly
  lower 72.33% Top-1 but the highest Top-5 so far at 91.02%. The best checkpoint
  remains epoch 14 at this point.
- Epochs 16 through 18 yielded 72.37%, 72.55%, and 72.58% Top-1. Epoch 18 is the
  new best and exceeds the measured baseline by 1.22 percentage points.

- Epochs 19 and 20 improved Top-1 to 72.61% and 72.69%. Epoch 21 reached 72.66%
  Top-1 and the highest Top-5 so far, 91.18%. The best checkpoint remains epoch 20.

- Epochs 22 and 23 improved Top-1 to 72.72% and 72.81%. Epoch 24 reached 72.77%.
  The best checkpoint is epoch 23 at this point, with 91.23% Top-5.
- Epochs 25 through 29 completed normally. Epoch 27 produced the retained best
  checkpoint (training record Top-1 72.83%, Top-5 91.26%). An independent full
  validation pass of that checkpoint measured Top-1 72.82% and Top-5 91.26%.

## 6. Final results

### Before final classification fine-tuning

- Baseline pretrained: Top-1 71.36%, Top-5 90.71%
- Main searched compact: Top-1 61.19%, Top-5 84.48%, loss 1.7882
- Pre-fine-tuning Top-1 delta: -10.17 percentage points

### Completed main experiment

End time: 2026-08-04 22:34 UTC

- Final static schedule (input plus twelve block outputs):
  `[197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 91, 41, 1]`
- Independently evaluated best compact checkpoint (epoch 27): Top-1 72.82%,
  Top-5 91.26%, loss 1.1569.
- Accuracy relative to the measured pretrained baseline: **+1.46 percentage
  points Top-1** and **+0.55 percentage points Top-5**.
- Estimated transformer MACs: 1,224,589,824 -> 1,000,060,416, a **18.34%
  reduction**.
- RTX 2080 Ti batch-one timing was 8.08 ms for the original model and 19.55 ms
  for the current unfused Python compact implementation. This is not an NPU
  latency result; Python indexing/module overhead dominates this GPU comparison.
- The final static-NHWC batch-one ONNX is 25.9 MB. ONNX checker and ONNX Runtime
  passed, with PyTorch/ORT maximum absolute error 3.10e-06. `Gather`,
  `GatherElements`, `ScatterElements`, `TopK`, and `Equal` are absent. The target
  NPU compiler itself was not available in this environment, so this establishes
  the specified NPU-safe graph gate rather than device latency or compilation.
- All 31 unit/integration tests pass.

Authoritative artifacts:

- `output/ps_deit_tiny/architecture.json`: searched token IDs and errors
- `output/ps_deit_tiny/search_records.jsonl`: per-candidate search history
- `output/ps_deit_tiny/compact_finetuned_best.pt`: final selected checkpoint
- `output/ps_deit_tiny/compact_best_eval.log`: independent final validation
- `output/ps_deit_tiny/efficiency.json`: MAC and GPU timing report
- `output/ps_deit_tiny/ps_compact_npu.onnx`: verified static NPU-safe graph
- `output/ps_deit_tiny/export.log`: export and numerical verification record
