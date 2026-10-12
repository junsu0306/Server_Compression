"""Preflight the actual exporter with temporary untrained fixtures, not final results."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "patch_slimming")]
import timm
import torch
from scripts.profiling.prepare_budget import build_target, save_model
from psvit.architecture import build_spec
from psvit.model_utils import get_shape


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2022)
    (ROOT / ".cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="export_preflight_", dir=ROOT / ".cache") as tmp:
        folder = Path(tmp)
        for name in ("vit_tiny_patch16_224", "vit_small_patch16_224"):
            baseline = timm.create_model(name, pretrained=False).eval()
            save_model(folder / "baseline.pt", baseline, name)
            reduced = build_target(baseline, 0.3, 0.95)
            save_model(folder / "reduced.pt", reduced, name)
            keep = [torch.arange(n) for n in [151, 151, 131, 131, 111, 111, 91, 91, 81, 81, 81, 1]]
            spec = build_spec(name, get_shape(reduced), keep, [0.0] * 12,
                              epsilon=0.02, error_metric="sample_relative_frobenius_sq_mean",
                              search_step=10, score_mode="paper_path_energy_dp")
            spec.to_json(str(folder / "architecture.json"))
            torch.save({"model": reduced.state_dict(), "mlp_dims": [b.mlp.fc1.out_features for b in reduced.blocks]},
                       folder / "patch.pt")
            stages = [("baseline", ["--model", name, "--baseline-checkpoint", str(folder / "baseline.pt")]),
                      ("reduced", ["--reduced", str(folder / "reduced.pt")]),
                      ("patch", ["--arch", str(folder / "architecture.json"), "--weights", str(folder / "patch.pt")])]
            for kind, flags in stages:
                output = folder / f"{name}_{kind}.onnx"
                subprocess.run([sys.executable, str(ROOT / "scripts/export_final_onnx.py"),
                                "--kind", kind, *flags, "--output", str(output), "--npu-safe"], check=True)
                record = json.loads(output.with_suffix(".verified.json").read_text())
                print(f"PASS {name}/{kind}: {record['pytorch_ort_max_abs']:.8g}", flush=True)
    print("All six ONNX preflight cases passed; temporary random-weight fixtures removed.")


if __name__ == "__main__":
    main()
