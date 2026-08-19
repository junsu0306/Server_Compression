"""Generate and evaluate 3/6/9-stage static masks for structurally reduced ViTs.

The token counts are policy inputs, while token identities are recalculated for
each checkpoint with the Patch Slimming path-energy score.  This is a staged
ablation, not Algorithm-1's free per-layer epsilon search.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
import timm
import timm.data
from torchvision import datasets

_HERE = Path(__file__).resolve().parent
_PS_ROOT = _HERE.parent
_ROOT = _PS_ROOT.parent
for p in (str(_PS_ROOT), str(_ROOT), str(_ROOT / ".deps")):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import evaluate
from psvit.architecture import build_spec
from psvit.compact import CompactPSViT
from psvit.data import build_calibration_loader
from psvit.ids import validate_nested_masks
from psvit.model_utils import get_shape, load_reduced_vit
from psvit.scoring import estimate_mean_scores


POLICIES = {
    "mild": (177, 157, 137),
    "medium": (157, 118, 79),
    "aggressive": (138, 89, 40),
    "extra_aggressive": (118, 69, 30),
}
BOUNDARY_BLOCKS = (2, 5, 8)  # 1-based layers 3, 6, 9


def _rank_ids(scores: torch.Tensor, required: torch.Tensor, count: int) -> torch.Tensor:
    required = torch.unique(required.cpu().long(), sorted=True)
    if count < required.numel():
        raise ValueError(f"count={count} is smaller than downstream requirement={required.numel()}")
    candidates = torch.tensor(
        [i for i in range(scores.numel()) if i not in set(required.tolist())], dtype=torch.long
    )
    order = torch.argsort(scores[candidates], descending=True)
    chosen = candidates[order[: count - required.numel()]]
    return torch.unique(torch.cat([required, chosen]), sorted=True)


def build_stage_masks(model, loader, device, counts, max_batches=None):
    """Score deepest-to-shallowest and enforce a nested, piecewise schedule."""
    n = get_shape(model).num_global_tokens
    keep = [torch.arange(n) for _ in range(12)]
    keep[-1] = torch.tensor([0])
    scores_by_layer = {}

    # At a boundary, rank using the current downstream stage. After selection,
    # propagate that fixed mask through the three-block stage.
    for t, count in reversed(list(zip(BOUNDARY_BLOCKS, counts))):
        scores = estimate_mean_scores(model, t, keep, loader, device,
                                      max_batches=max_batches, log_every=32)
        scores_by_layer[t] = scores
        required = keep[t + 3] if t + 3 < len(keep) else keep[-1]
        selected = _rank_ids(scores, required, count)
        for block in range(t, min(t + 3, len(keep) - 1)):
            keep[block] = selected.clone()

    validate_nested_masks(keep)
    return keep, scores_by_layer


def transformer_macs(model, schedule):
    """Analytical transformer MACs with rectangular attention and reduced FFNs."""
    d = int(model.embed_dim)
    total = 0
    for i, block in enumerate(model.blocks):
        nin, nout = schedule[i], schedule[i + 1]
        f = int(block.mlp.fc1.out_features)
        # Q projection at Nout; K/V at Nin; output projection at Nout;
        # QK and AV rectangular attention; two FFN projections at Nout.
        total += nout * d * d + 2 * nin * d * d + nout * d * d
        total += 2 * nout * nin * d
        total += 2 * nout * d * f
    return int(total)


def val_loader(model_name, data_path, batch_size, workers, max_samples=None):
    ref = timm.create_model(model_name, pretrained=False)
    cfg = timm.data.resolve_model_data_config(ref)
    tf = timm.data.create_transform(**cfg, is_training=False)
    ds = datasets.ImageFolder(os.path.join(data_path, "val"), transform=tf)
    if max_samples is not None:
        ds = torch.utils.data.Subset(ds, range(min(max_samples, len(ds))))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                       num_workers=workers, pin_memory=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reduced", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--data-path", default=str(_ROOT / "data/imagenet"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--calibration-samples", type=int, default=1024)
    ap.add_argument("--calibration-batch-size", type=int, default=8)
    ap.add_argument("--score-max-batches", type=int, default=None)
    ap.add_argument("--val-batch-size", type=int, default=256)
    ap.add_argument("--val-max-samples", type=int, default=None,
                    help="Debug-only deterministic validation prefix; omit for full 50k")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="vit-structural-patch-slimming")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    base, meta = load_reduced_vit(args.reduced, device=device)
    base.eval(); shape = get_shape(base)
    loader, _, ids = build_calibration_loader(
        meta["model_name"], args.data_path, args.calibration_samples,
        args.calibration_batch_size, seed=2022, split="train", num_workers=args.workers,
        sample_id_file=str(out / "calibration_ids.txt"),
    )
    vl = val_loader(meta["model_name"], args.data_path, args.val_batch_size, args.workers,
                    args.val_max_samples)

    run = None
    if args.wandb:
        import wandb
        name = f"{Path(args.reduced).parent.name}-stage369-ablation"
        run = wandb.init(project=args.wandb_project, name=name, config={
            "source": args.reduced, "model": meta["model_name"],
            "structural_compression_pct": meta.get("compression_rate"),
            "mlp_dims": meta["mlp_dims"], "boundaries_1based": [3, 6, 9],
            "calibration_samples": len(ids), "policies": POLICIES,
        })

    results = []
    baseline = evaluate(vl, base, device, amp=True)
    full_schedule = [shape.num_global_tokens] * (shape.num_blocks + 1)
    full_macs = transformer_macs(base, full_schedule)
    results.append({"policy": "baseline", "counts": [197, 197, 197],
                    "schedule": full_schedule, "macs": full_macs, **baseline})

    for policy in args.policies:
        counts = POLICIES[policy]
        print(f"\n[policy] {policy}: {counts}")
        model_for_score = copy.deepcopy(base).eval()
        keep, scores = build_stage_masks(model_for_score, loader, device, counts,
                                         args.score_max_batches)
        del model_for_score
        spec = build_spec(meta["model_name"], shape, keep, [-1.0] * 12,
                          epsilon=-1.0, error_metric="stage_ablation",
                          search_step=0, score_mode="paper_path_energy_dp_stage369",
                          method="patch_slimming_static_stage369")
        pdir = out / policy; pdir.mkdir(exist_ok=True)
        spec.to_json(str(pdir / "architecture.json"))
        torch.save({"model": base.state_dict(), "model_name": meta["model_name"],
                    "mlp_dims": meta["mlp_dims"], "source_reduced_checkpoint": args.reduced,
                    "structural_compression_rate": meta.get("compression_rate")},
                   pdir / "searched_student.pt")
        torch.save({str(k): v for k, v in scores.items()}, pdir / "mean_scores.pt")
        compact = CompactPSViT(base, keep, select_mode="index_select").to(device).eval()
        metrics = evaluate(vl, compact, device, amp=True)
        schedule = compact.token_schedule(); macs = transformer_macs(base, schedule)
        row = {"policy": policy, "counts": list(counts), "schedule": schedule,
               "macs": macs, "mac_reduction_pct": 100 * (1 - macs / full_macs),
               "top1_drop": baseline["acc1"] - metrics["acc1"], **metrics}
        results.append(row)
        (pdir / "metrics.json").write_text(json.dumps(row, indent=2))
        if run:
            run.log({f"candidate/{policy}/{k}": v for k, v in row.items()
                     if isinstance(v, (int, float))})

    (out / "stage_ablation_results.json").write_text(json.dumps(results, indent=2))
    if run:
        import wandb
        table = wandb.Table(columns=["policy", "n3", "n6", "n9", "top1", "top5",
                                     "loss", "top1_drop", "mac_reduction_pct"])
        for r in results:
            table.add_data(r["policy"], *r["counts"], r["acc1"], r["acc5"], r["loss"],
                           r.get("top1_drop", 0.0), r.get("mac_reduction_pct", 0.0))
        run.log({"stage369_ablation": table}); run.finish()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
