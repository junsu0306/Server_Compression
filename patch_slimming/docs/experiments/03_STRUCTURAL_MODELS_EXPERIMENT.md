# Structural-pruned ViT Patch Slimming experiment log

This is the live record for the production-target Patch Slimming study on the
physically reduced ViT checkpoints under `output/vit_*_final/reduced.pt`.

## 1. Motivation and scope

The earlier main experiment used the original pretrained DeiT-Tiny. Its token
schedule cannot be treated as authoritative for models whose FFN widths and
weights changed through structural pruning. This study therefore recalculates
static token masks for all four production candidates:

- `vit_tiny_30_final/reduced.pt`
- `vit_tiny_50_final/reduced.pt`
- `vit_small_30_final/reduced.pt`
- `vit_small_50_final/reduced.pt`

## 2. Dataset and tracking

- Dataset: ImageNet-1K
- Train split: 1,280,297 images across 1,000 classes
- Validation split: 50,000 images across 1,000 classes
- Mask calibration: deterministic subset of the ImageNet train split
- Candidate and final accuracy: ImageNet validation split
- W&B project: `vit-structural-patch-slimming` (new project)
- W&B account verified: `junsu000306`
- Large caches, temporary data, checkpoints, and W&B local files are kept on
  the workspace volume rather than the container overlay.

## 3. Experiment design

Token-count changes are restricted to evenly spaced 1-based layers 3, 6, and
9 (zero-based block outputs 2, 5, and 8). The last block emits only CLS. Masks
remain static and nested. Aggressive candidates are retained even when their
pre-fine-tuning Top-1 drop is large; selection is based on recovery after short
fine-tuning together with the compute/accuracy Pareto trade-off.

Initial candidate families (CLS included):

| Policy | After layer 3 | After layer 6 | After layer 9 | Purpose |
|---|---:|---:|---:|---|
| mild | 177 | 157 | 137 | accuracy-oriented control |
| medium | 157 | 118 | 79 | balanced candidate |
| aggressive | 138 | 89 | 40 | compute-oriented candidate |
| extra-aggressive | 118 | 69 | 30 | recovery-limit probe |

Token identities are recalculated from each reduced checkpoint using the
Patch Slimming significance score; only the token counts above are shared.

## 4. Initial audit

| Model | Structural compression | Per-block reduced MLP dimensions |
|---|---:|---|
| ViT-Tiny 30 | 29.98% | 461, 598, 571, 537, 572, 494, 505, 435, 335, 154, 64, 38 |
| ViT-Tiny 50 | 49.94% | 248, 293, 272, 232, 247, 168, 114, 74, 38, 38, 38, 38 |
| ViT-Small 30 | 30.01% | 738, 984, 841, 852, 859, 1079, 1092, 1080, 984, 818, 424, 77 |
| ViT-Small 50 | 50.01% | 459, 457, 440, 389, 380, 483, 439, 324, 276, 202, 166, 77 |

The existing Patch Slimming entry points assumed a standard pretrained timm
model. Reduced-checkpoint loading and metadata propagation are being added
before candidate generation so the physical FFN widths are never expanded back
to their original sizes.

## 5. Results

### Full ImageNet validation baselines

| Model | Top-1 | Top-5 | Loss | W&B run |
|---|---:|---:|---:|---|
| ViT-Tiny 30 | 73.53% | 91.86% | 1.0503 | `prgdpglk` |
| ViT-Tiny 50 | 66.14% | 87.66% | 1.3763 | `aey3i81g` |
| ViT-Small 30 | 81.05% | 95.86% | 0.7113 | `o54j64zo` |
| ViT-Small 50 | 77.41% | 94.13% | 0.8696 | `3ur6kdwc` |

The first three baselines use all 50,000 validation images. Candidate mask
generation is now running with 1,024 deterministic ImageNet train images per
model. The earlier 8-calibration/64-validation Tiny-30 run (`d1sx5u0u`) was a
pipeline-only smoke test and is excluded from comparisons.

### Stage 3/6/9 candidate results before fine-tuning

| Model | Policy | Token counts (L3/L6/L9) | Top-1 | Drop | Transformer MAC reduction |
|---|---|---|---:|---:|---:|
| Tiny 30 | mild | 177 / 157 / 137 | 70.91% | -2.61 pp | 19.97% |
| Tiny 30 | medium | 157 / 118 / 79 | 66.08% | -7.45 pp | 34.75% |
| Tiny 30 | aggressive | 138 / 89 / 40 | 59.42% | -14.10 pp | 45.29% |
| Tiny 30 | extra-aggressive | 118 / 69 / 30 | 54.63% | -18.89 pp | 51.90% |
| Tiny 50 | mild | 177 / 157 / 137 | 63.37% | -2.77 pp | 21.63% |
| Tiny 50 | medium | 157 / 118 / 79 | 57.72% | -8.43 pp | 36.42% |
| Tiny 50 | aggressive | 138 / 89 / 40 | 49.41% | -16.74 pp | 46.69% |
| Tiny 50 | extra-aggressive | 118 / 69 / 30 | 43.98% | -22.16 pp | 52.97% |
| Small 30 | mild | 177 / 157 / 137 | 79.16% | -1.89 pp | 19.82% |
| Small 30 | medium | 157 / 118 / 79 | 72.44% | -8.61 pp | 35.51% |
| Small 30 | aggressive | 138 / 89 / 40 | 57.35% | -23.70 pp | 46.85% |
| Small 30 | extra-aggressive | 118 / 69 / 30 | 49.29% | -31.76 pp | 53.52% |
| Small 50 | mild | 177 / 157 / 137 | 75.26% | -2.15 pp | 20.53% |
| Small 50 | medium | 157 / 118 / 79 | 65.10% | -12.30 pp | 35.59% |
| Small 50 | aggressive | 138 / 89 / 40 | 54.31% | -23.09 pp | 46.37% |
| Small 50 | extra-aggressive | 118 / 69 / 30 | 46.20% | -31.20 pp | 52.80% |

All values above use the full 50,000-image validation split. Mild is the clear
accuracy-oriented choice. Medium is the main recovery candidate because it
removes about 35--36% of the remaining transformer MACs while deliberately
allowing a larger pre-fine-tuning loss. Aggressive policies remain useful as a
recovery-limit probe, but Small models show substantially greater immediate
damage than Tiny models at the same token counts.

### Medium-policy one-epoch recovery screening

The purpose of this phase is candidate selection, not final convergence. Each
model receives one complete ImageNet train epoch at LR 2e-5, global batch 512,
with no warmup, followed by the full 50,000-image validation pass.

| Model | Pre-FT Top-1 | After 1 epoch | Recovery | Gap to structural baseline | W&B run |
|---|---:|---:|---:|---:|---|
| Tiny 30 | 66.08% | 68.42% | +2.34 pp | -5.11 pp | `jwxlqt2u` |
| Tiny 50 | 57.72% | 61.19% | +3.47 pp | -4.95 pp | `9ldnorbw` |
| Small 30 | 72.44% | 75.34% | +2.90 pp | -5.71 pp | `mjm9pvuj` |
| Small 50 | 65.10% | 70.94% | +5.84 pp | -6.47 pp | `8peltby6` |

All four medium candidates recover materially in only one epoch. Small-50 has
the largest initial damage among medium candidates but also the strongest early
recovery. The initially attempted low-LR warmup runs were stopped before epoch
completion and are excluded from results.

### Combined structural and token compute reduction

The candidate MAC reductions above are relative to each already-reduced model.
Relative to the original full-width ViT transformer, the selected medium policy
produces the following combined reductions:

| Model | Structural-only transformer MAC reduction | Structural + medium token reduction |
|---|---:|---:|
| Tiny 30 | 27.50% | **52.70%** |
| Tiny 50 | 45.81% | **65.55%** |
| Small 30 | 28.67% | **54.00%** |
| Small 50 | 47.78% | **66.37%** |

These analytical figures cover the twelve transformer blocks, including
rectangular attention and each checkpoint's actual reduced FFN widths. Patch
embedding and classifier cost are not included.

## 6. Current selection

The **medium** stage policy `(157, 118, 79)` is selected for long fine-tuning on
all four production candidates. It reduces the remaining transformer's MACs by
about 35--36%, shows clear recovery after a single epoch, and yields a combined
52.70--66.37% transformer MAC reduction versus the original full-width models.

Mild remains the conservative fallback when the final accuracy requirement is
strict. Aggressive and extra-aggressive are not selected for the main run:
their pre-fine-tuning loss is disproportionately large, especially for Small,
relative to the additional 10--18 percentage points of MAC reduction beyond
medium.

The one-epoch checkpoints are screening artifacts, not final deployable models.
The next phase is a resume-capable 30-epoch fine-tuning run for the four selected
medium candidates, followed by independent validation and NPU-safe ONNX export.

Final implementation audit for this selection phase:

- All four saved architectures validate and have the expected schedule
  `[197, 197, 197, 157, 157, 157, 118, 118, 118, 79, 79, 79, 1]`.
- All four fine-tuned checkpoints retain exactly their source `mlp_dims` and
  load strictly into the physically reduced topology.
- 33/33 Patch Slimming unit and integration tests pass.
- `git diff --check` reports no whitespace errors.
