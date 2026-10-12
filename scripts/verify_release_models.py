"""CPU smoke/parity checks for packaged weights and all 12 released ONNX files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "patch_slimming")]

import numpy as np
import onnxruntime as ort
import timm
import torch

from pruning.vit_reducing import reduce_vit_model, transfer_pruning_mask
from scripts.export_final_onnx import load
from scripts.package_release import DEFAULT_RELEASE, KEYS, STAGES, verify_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    args = parser.parse_args()
    print(json.dumps(verify_release(args.release)), flush=True)
    torch.set_num_threads(4)
    torch.manual_seed(2022)
    images = [torch.randn(1, 224, 224, 3) for _ in range(3)]
    results = []
    for key in KEYS:
        folder = args.release / key
        checkpoint = torch.load(folder / "01_structural/checkpoint_final.pt", map_location="cpu", weights_only=False)
        name = checkpoint["args"]["model"]
        if checkpoint["epoch"] != 29 or not checkpoint["release_inference_only"]:
            raise ValueError(f"Wrong structural selection: {key}")
        raw = timm.create_model(name, pretrained=False)
        raw.load_state_dict(checkpoint["model"], strict=True)
        ema = timm.create_model(name, pretrained=False)
        ema.load_state_dict(checkpoint["model_ema"], strict=True)
        transfer_pruning_mask(raw, ema)
        reduce_vit_model(ema)
        reduced = torch.load(folder / "02_reduced/reduced.pt", map_location="cpu", weights_only=False)
        for field, tensor in ema.state_dict().items():
            torch.testing.assert_close(tensor, reduced["state_dict"][field], rtol=0, atol=0)
        del raw, ema, checkpoint, reduced
        for stage in STAGES:
            options = SimpleNamespace(
                kind="patch" if stage == "patch_slimming" else stage,
                model=name, baseline_checkpoint=str(folder / "00_budget/baseline.pt"),
                reduced=str(folder / "02_reduced/reduced.pt"),
                arch=str(folder / "03_patch_slimming_paper/architecture.json"),
                weights=str(folder / "03_patch_slimming_paper/compact_finetuned_best.pt"), npu_safe=True)
            model, _ = load(options)
            path = folder / "04_onnx" / f"{key}__{stage}__nhwc_b1_npusafe.onnx"
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 4
            session_options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(path), session_options, providers=["CPUExecutionProvider"])
            differences = []
            for image in images:
                with torch.inference_mode():
                    expected = model(image).numpy()
                actual = session.run(["output"], {"input": image.numpy()})[0]
                differences.append(float(np.max(np.abs(actual - expected))))
            maximum = max(differences)
            if not np.isfinite(maximum) or maximum >= 1e-4:
                raise ValueError(f"Released weight/ONNX parity failed: {key}/{stage}: {maximum}")
            results.append({"key": key, "stage": stage, "max_abs": maximum, "samples": len(images)})
            print(json.dumps(results[-1]), flush=True)
            del session, model
    print(json.dumps({"state": "passed", "pruned_to_reduced_exact_checks": len(KEYS), "onnx_checks": results}, indent=2))


if __name__ == "__main__":
    main()
