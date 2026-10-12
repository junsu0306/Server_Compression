"""Isolated FP32 inference peak measurement for reduced/pruned ViT checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import timm
import torch

from pruning.vit_reducing import apply_reduced_config, reduce_vit_model, transfer_pruning_mask


def load_candidate(checkpoint):
    if "mlp_dims" in checkpoint:
        name = checkpoint["model_name"]
        state = checkpoint["state_dict"]
        model = timm.create_model(name, pretrained=False, num_classes=state["head.weight"].shape[0])
        apply_reduced_config(model, checkpoint["mlp_dims"])
        model.load_state_dict(state, strict=True)
    else:
        name = checkpoint["args"]["model"]
        state = checkpoint["model"]
        model = timm.create_model(name, pretrained=False, num_classes=state["head.weight"].shape[0])
        model.load_state_dict(state, strict=True)
        if checkpoint.get("model_ema") is not None:
            ema = timm.create_model(name, pretrained=False, num_classes=state["head.weight"].shape[0])
            ema.load_state_dict(checkpoint["model_ema"], strict=True)
            transfer_pruning_mask(model, ema)
            model = ema
        reduce_vit_model(model)
    return model.eval(), name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--model", help="baseline timm model")
    source.add_argument("--arch", help="Patch Slimming architecture.json")
    parser.add_argument("--weights", help="Patch Slimming fine-tuned checkpoint")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--input-npy", help="shared FP32 NHWC [1,224,224,3] input")
    parser.add_argument("--details", action="store_true", help="host memory and separate activation inspection")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--expected-model", default="")
    args = parser.parse_args()
    if min(args.batch_size, args.input_size, args.runs) < 1 or args.warmups < 0:
        parser.error("batch-size/input-size/runs must be positive; warmups must be nonnegative")
    if args.arch and not args.weights:
        parser.error("--weights is required with --arch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("CUDA is required for peak allocated memory measurement")
    torch.cuda.set_device(device)
    torch.manual_seed(2022)
    sampler = None
    if args.details or args.arch:
        sys.path.insert(0, str(ROOT / "patch_slimming"))
    if args.details:
        from scripts.profiling.activation import ProcessSampler, describe_tensors, analytical_blocks
        sampler = ProcessSampler(interval=0.02)
        sampler.start()
    if args.model:
        name = args.model
        model = timm.create_model(name, pretrained=args.pretrained).eval()
    elif args.arch:
        from psvit.architecture import ArchitectureSpec
        from psvit.compact import CompactPSViT
        from psvit.model_utils import apply_checkpoint_mlp_dims
        spec = ArchitectureSpec.from_json(args.arch)
        spec.validate()
        name = spec.model_name
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
        state = checkpoint["model"]
        base = timm.create_model(name, pretrained=False, num_classes=state["head.weight"].shape[0])
        apply_checkpoint_mlp_dims(base, checkpoint)
        base.load_state_dict(state, strict=True)
        model = CompactPSViT(base, spec.keep_ids_list(), select_mode="matmul").eval()
        del checkpoint, state
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model, name = load_candidate(checkpoint)
        del checkpoint
    if args.expected_model and name != args.expected_model:
        parser.error(f"reference model {name} does not match {args.expected_model}")
    model = model.to(device=device, dtype=torch.float32)
    if args.arch:
        model.bake_for_export(device, torch.float32)
    input_sha256 = None
    if args.input_npy:
        import numpy as np
        array = np.load(args.input_npy, allow_pickle=False)
        if array.shape != (args.batch_size, args.input_size, args.input_size, 3) or array.dtype != np.float32:
            parser.error("--input-npy must be FP32 NHWC with matching batch/input size")
        input_sha256 = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        image = torch.from_numpy(array.transpose(0, 3, 1, 2).copy()).to(device)
    else:
        image = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)
    base = model.model if args.arch else model
    schedule = model.token_schedule() if args.arch else [int(base.patch_embed.num_patches) + 1] * (len(base.blocks) + 1)
    with torch.inference_mode():
        for _ in range(args.warmups):
            model(image)
        torch.cuda.synchronize(device)
        resident_bytes = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(args.runs):
            # Do not retain the previous iteration's logits during the next forward.
            model(image)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    result = {
        "model": name, "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__, "timm_version": timm.__version__,
        "dtype": "float32", "input_shape": list(image.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "warmups": args.warmups, "runs": args.runs,
        "source": args.checkpoint or args.weights or args.model,
        "architecture": args.arch, "pretrained": args.pretrained,
        "input_sha256": input_sha256, "token_schedule": schedule,
        "mlp_dims": [block.mlp.fc1.out_features for block in base.blocks],
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "buffer_bytes": sum(b.numel() * b.element_size() for b in model.buffers()),
        "resident_allocated_bytes": resident_bytes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "transient_peak_bytes": torch.cuda.max_memory_allocated(device) - resident_bytes,
        "latency_mean_ms": elapsed * 1000 / args.runs,
    }
    if sampler:
        result["process_memory"] = sampler.stop()
        # Hooks run separately, after peak/latency capture, and retain only metadata.
        rows, handles = [], []
        def hook(name):
            def record(module, inputs, output):
                rows.append({"name": name, "type": type(module).__name__,
                             "output": describe_tensors(output)})
            return record
        for module_name, module in model.named_modules():
            if module_name and not list(module.children()):
                handles.append(module.register_forward_hook(hook(module_name)))
        try:
            with torch.inference_mode():
                model(image)
        finally:
            for handle in handles:
                handle.remove()
        result["activation"] = {
            "module_output_sum_bytes": sum(row["output"]["bytes"] for row in rows),
            "note": "sum of produced module outputs in one separate forward; not live peak",
            "modules": rows,
            "blocks": analytical_blocks(model, schedule, args.batch_size, 4),
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
