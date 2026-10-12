"""Full ImageNet validation, FP32, for the four saved stages of a final run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "patch_slimming")]
import timm
import torch
import yaml
from torchvision.datasets import ImageFolder
from pruning.vit_reducing import transfer_pruning_mask
from pruning.engine import evaluate
from scripts.profiling.profile_reduced_memory import load_candidate
from psvit.architecture import ArchitectureSpec
from psvit.compact import CompactPSViT
from psvit.model_utils import apply_checkpoint_mlp_dims


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default="configs/final.yaml")
    ap.add_argument("--key", required=True)
    ap.add_argument("--kind", choices=["baseline", "pruned", "reduced", "patch_slimming"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    torch.set_num_threads(4)
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    name = matrix["models"][args.key]["model"]
    root = ROOT / matrix["output_root"] / args.key
    files = {"baseline": root / "00_budget/baseline.pt",
             "pruned": root / "01_structural/checkpoint_final.pt",
             "reduced": root / "02_reduced/reduced.pt",
             "patch_slimming": root / "03_patch_slimming_paper/compact_finetuned_best.pt"}
    checkpoint = torch.load(files[args.kind], map_location="cpu", weights_only=False)
    if args.kind == "pruned" and checkpoint["epoch"] + 1 != matrix["structural"]["epochs"]:
        raise ValueError("Pruned evaluation requires the exact structural terminal epoch")
    if args.kind in ("baseline", "reduced"):
        model, _ = load_candidate(checkpoint)
    elif args.kind == "pruned":
        raw = timm.create_model(name, pretrained=False)
        raw.load_state_dict(checkpoint["model"], strict=True)
        model = timm.create_model(name, pretrained=False)
        model.load_state_dict(checkpoint["model_ema"], strict=True)
        transfer_pruning_mask(raw, model)
        del raw
    else:
        model = timm.create_model(name, pretrained=False)
        apply_checkpoint_mlp_dims(model, checkpoint)
        model.load_state_dict(checkpoint["model"], strict=True)
        arch = ArchitectureSpec.from_json(str(root / "03_patch_slimming_paper/architecture.json"))
        model = CompactPSViT(model, arch.keep_ids_list(), select_mode="index_select")
    base = model.model if args.kind == "patch_slimming" else model
    data_config = timm.data.resolve_model_data_config(base)
    dataset = ImageFolder(str(Path(matrix["data_path"]) / "val"),
                          timm.data.create_transform(**data_config, is_training=False))
    if len(dataset) != 50000 or len(dataset.classes) != 1000:
        raise ValueError(f"Expected ImageNet-1k 50000-image validation, got {len(dataset)}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=8, pin_memory=True)
    model = model.cuda().eval()
    start = time.perf_counter()
    metrics = evaluate(loader, model, torch.device("cuda:0"), amp=False)
    result = {"key": args.key, "kind": args.kind, "model": name,
              "source": str(files[args.kind]), "dataset": str(Path(matrix["data_path"]) / "val"),
              "samples": len(dataset), "dtype": "float32", "amp": False,
              "batch_size": args.batch_size, "data_config": data_config,
              "parameters": sum(p.numel() for p in model.parameters()),
              "metrics": metrics, "seconds": time.perf_counter() - start}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    # Only write completion after evaluation and remote logging succeed.
    import wandb
    run = wandb.init(project=matrix["project"], name=f"{args.key}__{args.kind}__fp32_eval",
                     id=f"{matrix['run_prefix']}-{args.key}-{args.kind}-eval", resume="allow",
                     config={"key": args.key, "stage": args.kind, "dtype": "float32"})
    run.log({"eval/top1": metrics["acc1"], "eval/top5": metrics["acc5"], "eval/loss": metrics["loss"]})
    run.finish()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
