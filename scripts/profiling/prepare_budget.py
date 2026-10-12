"""Build an un-fine-tuned target structure and measure its isolated CUDA peak."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import timm
import torch
import yaml
from pruning.vit_pruning import ViTPruner
from pruning.vit_reducing import reduce_vit_model, get_reduced_config


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_target(baseline, compression, max_sparsity):
    target = copy.deepcopy(baseline).eval()
    pruner = ViTPruner(target, compression, max_sparsity=max_sparsity)
    pruner.apply(target)
    reduce_vit_model(target)
    return target


def save_model(path, model, name):
    torch.save({"model_name": name, "state_dict": model.state_dict(),
                "mlp_dims": get_reduced_config(model), "trained_steps": 0}, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--key", required=True)
    args = ap.parse_args()
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    entry = matrix["models"][args.key]; settings = matrix["structural"]
    folder = ROOT / matrix["output_root"] / args.key / "00_budget"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "budget.json"
    if output.exists():
        raise FileExistsError(f"Budget is immutable: {output}")
    torch.set_num_threads(4)
    torch.manual_seed(matrix["seed"])
    baseline_path = folder / "baseline.pt"
    target_path = folder / "target_untrained.pt"
    if baseline_path.exists():
        checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)
        if checkpoint["model_name"] != entry["model"]:
            raise ValueError("Baseline identity mismatch")
        baseline = timm.create_model(entry["model"], pretrained=False).eval()
        baseline.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        baseline = timm.create_model(entry["model"], pretrained=True).eval()
        save_model(baseline_path, baseline, entry["model"])
    target = build_target(baseline, entry["target_compression"], settings["pruning_max_sparsity"])
    save_model(target_path, target, entry["model"])
    profiles = {}
    for label, checkpoint in (("baseline", baseline_path), ("target", target_path)):
        profile = folder / f"{label}_profile.json"
        subprocess.run([sys.executable, str(ROOT / "scripts/profiling/profile_reduced_memory.py"),
                        "--checkpoint", str(checkpoint), "--output", str(profile), "--details",
                        "--device", "cuda:0", "--batch-size", str(settings["memory_batch_size"]),
                        "--warmups", str(settings["memory_warmups"]),
                        "--runs", str(settings["memory_runs"])], check=True)
        profiles[label] = json.loads(profile.read_text())
    result = {"method": matrix["method"], "model": entry["model"],
              "target_compression": entry["target_compression"], "trained_steps": 0,
              "metric": "cuda_peak_allocated_bytes", "max_mib": profiles["target"]["peak_allocated_bytes"] / 2**20,
              "baseline_checkpoint": str(baseline_path), "baseline_sha256": sha256(baseline_path),
              "target_checkpoint": str(target_path), "target_sha256": sha256(target_path),
              "baseline_profile": profiles["baseline"], "target_profile": profiles["target"],
              "pruning": {"method": "global_l2", "max_sparsity": settings["pruning_max_sparsity"]}}
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    print(f"{args.key}: M_max={result['max_mib']:.6f} MiB; no fine-tuning performed", flush=True)


if __name__ == "__main__":
    main()
