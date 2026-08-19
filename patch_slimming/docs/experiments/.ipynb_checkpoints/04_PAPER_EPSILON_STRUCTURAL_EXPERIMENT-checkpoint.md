# Paper-style relaxed-epsilon search on structural-pruned ViTs

This is the live record for the Algorithm-1 comparison requested after the
fixed mild/medium/aggressive ablation.

## 1. Question

The previous production study manually fixed four token-count policies and used
the paper-derived significance score only to choose token identities. This
experiment instead lets the paper's top-down reconstruction criterion determine
the token count at every layer.

## 2. Protocol

- Targets: the four `output/vit_*_{30,50}_final/reduced.pt` production models
- Dataset: ImageNet-1K
- Calibration: 1,024 deterministic train images, seed 2022
- Search direction: block 10 down to block 0; block 11 emits CLS only
- Candidate ordering: Patch Slimming path-energy significance score
- Candidate increment: 10 tokens
- Acceptance: next-layer reconstruction MSE <= epsilon
- Candidate reconstruction: AdamW, LR 1e-5, 3 epochs
- Nested static mask: required
- Epsilon: **0.05**
- GPUs: **physical GPU 4 and physical GPU 6 only**
- W&B project: `vit-structural-patch-slimming-paper-search`
- Output root: `output/ps_structural_paper_eps005/`

## 3. Epsilon rationale

The implementation plan records paper experiments at epsilon 0.01 and 0.02.
Here epsilon 0.05 is intentionally 2.5 times looser than 0.02. This is not
claimed as a paper hyperparameter: it is an explicit recovery-aware extension,
motivated by the prior observation that classification fine-tuning recovered
2.34--5.84 Top-1 percentage points after only one epoch and more than 10 points
in the original DeiT-Tiny 30-epoch experiment.

## 4. Results

Machine-readable candidate records are saved in each model's
`search_records.jsonl`, and every candidate is also logged to W&B.

### Search batch 1: Tiny models

| Model | Final schedule | Layer-10 accepted MSE | Layer-0 accepted MSE | W&B run |
|---|---|---:|---:|---|
| Tiny 30 | `[197, 11 x11, 1]` | 0.04794 | 0.03588 | `6qumwf2a` |
| Tiny 50 | `[197, 11 x11, 1]` | 0.04388 | 0.03110 | `lvl41wee` |

At block 10, both models required ten added patches beyond CLS and accepted 11
tokens. Every shallower block then passed epsilon without adding another token.
This is much more aggressive than the manual medium schedule and makes full
classification validation essential; local reconstruction acceptance alone is
not treated as evidence of usable accuracy.

### Search batch 2: Small models

| Model | Final schedule | Layer-10 tokens / MSE | Layer-9 tokens / MSE | W&B run |
|---|---|---:|---:|---|
| Small 30 | `[197, 71 x10, 61, 1]` | 61 / 0.03400 | 71 / 0.04783 | `dovg6k8p` |
| Small 50 | `[197, 51 x10, 41, 1]` | 41 / 0.04869 | 51 / 0.04265 | `m8m7uw4o` |

Unlike Tiny, the Small models required materially more tokens. Once block 9
was accepted, every shallower layer passed without adding another token.

### Full ImageNet validation before classification fine-tuning

| Model | Structural baseline Top-1 | Paper-search Top-1 | Top-5 | Loss | Top-1 drop |
|---|---:|---:|---:|---:|---:|
| Tiny 30 | 73.53% | 5.80% | 14.75% | 6.0263 | -67.73 pp |
| Tiny 50 | 66.14% | 3.24% | 9.10% | 6.7333 | -62.90 pp |
| Small 30 | 81.05% | 55.57% | 78.50% | 1.9617 | -25.48 pp |
| Small 50 | 77.41% | 33.56% | 56.82% | 3.3195 | -43.85 pp |

Epsilon 0.05 is clearly too permissive for Tiny under the current local MSE
criterion. It also causes a large immediate loss for Small, although Small-30
remains trainable. One full classification epoch is still run for all four to
measure, rather than assume, the recovery limit.

### One-epoch recovery: Tiny models

Recovery used the full ImageNet-1K train split, LR `2e-5`, batch size 256,
weight decay 0.05, and no warm-up. Validation is on all 50,000 ImageNet-1K
validation images.

| Model | Pre-FT Top-1 | 1-epoch Top-1 | Recovery | Top-5 | Remaining gap to structural baseline | W&B run |
|---|---:|---:|---:|---:|---:|---|
| Tiny 30 | 5.80% | 24.53% | +18.73 pp | 46.95% | -49.00 pp | `bk1sa1mc` |
| Tiny 50 | 3.24% | 20.18% | +16.94 pp | 40.80% | -45.96 pp | `bwcek6cg` |

The large recovery confirms that pre-fine-tuning accuracy alone is too strict
for schedule selection. Nevertheless, one epoch does not make the 11-token
Tiny schedules usable: both remain more than 45 percentage points below their
structural baselines. For Tiny, epsilon 0.05 therefore crosses the useful
recovery-aware range under this local reconstruction objective.

An earlier launch of these two jobs was manually stopped near step 450 because
buffered stdout made progress appear stalled. It produced no epoch checkpoint
and is excluded (`3ir1ouyf`, `2fucr2dt`). The table reports only the clean,
unbuffered reruns shown above.

### Analytical transformer MACs

MACs use the same rectangular-attention and structurally reduced FFN accounting
as the prior 3/6/9 ablation. They exclude patch embedding and classifier costs.

| Model | Paper schedule MACs | Reduction vs structural baseline | Reduction vs original full-width ViT | Extra reduction vs medium |
|---|---:|---:|---:|---:|
| Tiny 30 | 53,718,528 | 93.95% | 95.61% | 90.73% |
| Tiny 50 | 41,198,592 | 93.79% | 96.64% | 90.24% |
| Small 30 | 1,090,610,688 | 66.33% | 75.98% | 47.79% |
| Small 50 | 566,760,960 | 76.10% | 87.52% | 62.89% |

These very large reductions explain why direct accuracy is much lower than the
fixed medium policy: relaxed epsilon did not merely allow a modest additional
cut, especially for Tiny where it removed about 90% of medium's remaining
transformer MACs.

### One-epoch recovery: Small models

Recovery used the full ImageNet-1K train split, LR `2e-5`, batch size 128,
weight decay 0.05, and no warm-up. Validation is on all 50,000 images.

| Model | Pre-FT Top-1 | 1-epoch Top-1 | Recovery | Top-5 | Remaining gap to structural baseline | W&B run |
|---|---:|---:|---:|---:|---:|---|
| Small 30 | 55.57% | 64.71% | +9.14 pp | 86.29% | -16.34 pp | `v1dxni7v` |
| Small 50 | 33.56% | 50.55% | +16.99 pp | 75.15% | -26.86 pp | `qq5rej3f` |

### Final comparison and decision

| Model | Paper ε=.05 after 1 epoch | Manual medium after 1 epoch | Difference | Paper extra MAC reduction vs medium |
|---|---:|---:|---:|---:|
| Tiny 30 | 24.53% | 68.42% | -43.89 pp | 90.73% |
| Tiny 50 | 20.18% | 61.19% | -41.01 pp | 90.24% |
| Small 30 | 64.71% | 75.34% | -10.63 pp | 47.79% |
| Small 50 | 50.55% | 70.94% | -20.39 pp | 62.89% |

Fine-tuning recovery is indeed large (+9.14 to +18.73 pp), so allowing more
pre-fine-tuning degradation is justified. However, epsilon 0.05 is beyond the
usable range for all four models, especially Tiny. Local next-layer MSE does
not preserve classification semantics tightly enough at this threshold.

The production choice therefore remains the 3/6/9 medium schedule. The paper
search result is retained as a measured extreme-compression point, not a
deployment candidate. A follow-up search should bracket the boundary rather
than repeat 0.05: start with epsilon 0.02 (the larger paper-reported setting),
then test 0.03 only when the pre-FT or short-recovery guard passes. Recommended
guards are model-relative: after one recovery epoch, remain within 8 pp of the
structural baseline, and reject schedules that collapse Tiny below 40 tokens
at every hidden block.

## 5. Reproducibility artifacts

- Search and recovery configs: `configs/structural/paper_eps005_*.yaml`
- Checkpoints and JSONL candidate traces: `output/ps_structural_paper_eps005/`
- W&B project: <https://wandb.ai/junsu000306/vit-structural-patch-slimming-paper-search>
- Search runs: `6qumwf2a`, `lvl41wee`, `dovg6k8p`, `m8m7uw4o`
- Included recovery runs: `bk1sa1mc`, `bwcek6cg`, `v1dxni7v`, `qq5rej3f`
- Compute restriction: physical GPUs 4 and 6 only

Final verification: all four best checkpoints strict-load with their preserved
structural `mlp_dims`; their runtime token schedules exactly match
`architecture.json`. The 33 Patch Slimming regression tests pass and
`git diff --check` reports no whitespace errors.
