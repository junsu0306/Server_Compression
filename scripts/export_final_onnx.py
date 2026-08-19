#!/usr/bin/env python3
"""Export baseline, structurally reduced, or Patch-Slimming ViT to NHWC ONNX.

All artifacts use fixed batch 1 and NHWC input [1,224,224,3], matching the
Mobilint calibration/runtime path. PyTorch vs ORT is always checked by default.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PS_ROOT = ROOT / "patch_slimming"
for path in (str(ROOT), str(PS_ROOT)):
    if path not in sys.path: sys.path.insert(0, path)

import numpy as np
import torch
import torch.nn as nn
import timm

from psvit.architecture import ArchitectureSpec
from psvit.compact import CompactPSViT
from psvit.model_utils import apply_checkpoint_mlp_dims, load_reduced_vit


class NHWCWrapper(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, image):
        return self.model(image.permute(0, 3, 1, 2))


def load(args):
    if args.kind == "baseline":
        if not args.model: raise ValueError("--model is required for baseline")
        model = timm.create_model(args.model, pretrained=True).eval()
        return NHWCWrapper(model).eval(), {"kind": "baseline", "model": args.model}
    if args.kind == "reduced":
        if not args.reduced: raise ValueError("--reduced is required")
        model, meta = load_reduced_vit(args.reduced)
        return NHWCWrapper(model.eval()).eval(), {
            "kind": "reduced", "model": meta["model_name"],
            "compression_rate": meta.get("compression_rate"), "mlp_dims": meta["mlp_dims"]}
    if not args.arch or not args.weights:
        raise ValueError("--arch and --weights are required for patch")
    spec = ArchitectureSpec.from_json(args.arch); spec.validate()
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    base = timm.create_model(spec.model_name, pretrained=False,
                             num_classes=state["head.weight"].shape[0])
    apply_checkpoint_mlp_dims(base, checkpoint)
    base.load_state_dict(state, strict=True); base.eval()
    compact = CompactPSViT(base, spec.keep_ids_list(), select_mode="matmul",
                           nhwc_input=True, npu_static_batch_one=True).eval()
    compact.bake_for_export(torch.device("cpu"), torch.float32)
    return compact, {"kind": "patch", "model": spec.model_name,
                     "token_schedule": spec.token_schedule()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=["baseline", "reduced", "patch"])
    parser.add_argument("--model", default="")
    parser.add_argument("--reduced", default="")
    parser.add_argument("--arch", default="")
    parser.add_argument("--weights", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    model, metadata = load(args)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 224, 224, 3)
    with torch.inference_mode(): reference = model(dummy).numpy()
    torch.onnx.export(model, dummy, output, opset_version=args.opset,
                      input_names=["input"], output_names=["output"],
                      do_constant_folding=True,
                      training=torch.onnx.TrainingMode.EVAL)

    import onnx
    graph = onnx.load(str(output)); onnx.checker.check_model(graph)
    ops = sorted({node.op_type for node in graph.graph.node})
    shape = [d.dim_value or d.dim_param for d in graph.graph.input[0].type.tensor_type.shape.dim]
    print("metadata=%s" % metadata)
    print("input=%s nodes=%d size=%.2f MiB" % (shape, len(graph.graph.node), output.stat().st_size / 2**20))
    print("ops=%s" % ops)
    if not args.no_verify:
        import onnxruntime as ort
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        actual = session.run(["output"], {"input": dummy.numpy()})[0]
        diff = float(np.max(np.abs(reference - actual)))
        print("PyTorch vs ORT max_abs_diff=%.6e" % diff)
        if diff >= 1e-4: raise RuntimeError("ONNX verification failed")


if __name__ == "__main__":
    main()
