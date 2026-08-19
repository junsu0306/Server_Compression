#!/usr/bin/env python3
"""Detailed pre-ONNX memory and activation profiler for CompactPSViT.

Reports measured CUDA/process peaks and two activation views:
1) forward-hook tensor bytes actually emitted by PyTorch modules;
2) analytical block working-set tensors, including functional attention
   intermediates that module hooks cannot observe.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_PS_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_PS_ROOT)
for _path in (_PS_ROOT, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import psutil
import torch
import torch.nn as nn
import timm

from psvit.architecture import ArchitectureSpec
from psvit.compact import CompactPSViT
from psvit.model_utils import apply_checkpoint_mlp_dims, load_reduced_vit


def tensor_info(value):
    tensors = []
    if torch.is_tensor(value):
        tensors.append(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            tensors.extend(tensor_info(item))
    elif isinstance(value, dict):
        for item in value.values():
            tensors.extend(tensor_info(item))
    elif hasattr(value, "x") and torch.is_tensor(value.x):
        tensors.append(value.x)
    return tensors


def describe_tensors(value):
    tensors = tensor_info(value)
    return {
        "bytes": sum(t.numel() * t.element_size() for t in tensors),
        "numel": sum(t.numel() for t in tensors),
        "shapes": [list(t.shape) for t in tensors],
        "dtypes": sorted({str(t.dtype).replace("torch.", "") for t in tensors}),
    }


class ProcessSampler:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def snapshot(self):
        info = self.process.memory_info()
        full = self.process.memory_full_info()
        vm = psutil.virtual_memory()
        return {
            "time": time.time(), "rss": info.rss, "vms": info.vms,
            "uss": getattr(full, "uss", 0), "pss": getattr(full, "pss", 0),
            "system_used": vm.used, "system_available": vm.available,
        }

    def start(self):
        self.samples = [self.snapshot()]
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.samples.append(self.snapshot())

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.samples.append(self.snapshot())
        return {
            "sample_count": len(self.samples),
            "rss_start_bytes": self.samples[0]["rss"],
            "rss_peak_bytes": max(x["rss"] for x in self.samples),
            "uss_peak_bytes": max(x["uss"] for x in self.samples),
            "pss_peak_bytes": max(x["pss"] for x in self.samples),
            "system_used_peak_bytes": max(x["system_used"] for x in self.samples),
            "system_available_min_bytes": min(x["system_available"] for x in self.samples),
        }


def load_model(args, device):
    if args.arch:
        spec = ArchitectureSpec.from_json(args.arch)
        spec.validate()
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        base = timm.create_model(
            spec.model_name, pretrained=False,
            num_classes=state["head.weight"].shape[0],
        )
        apply_checkpoint_mlp_dims(base, checkpoint)
        base.load_state_dict(state, strict=True)
        model = CompactPSViT(
            base, spec.keep_ids_list(), select_mode=args.select_mode,
        )
        schedule = model.token_schedule()
        model_name = spec.model_name
    elif args.reduced:
        model, checkpoint = load_reduced_vit(args.reduced, device=None)
        patches = int(model.patch_embed.num_patches) + int(model.num_prefix_tokens)
        schedule = [patches] * (len(model.blocks) + 1)
        model_name = checkpoint["model_name"]
    else:
        model = timm.create_model(args.model, pretrained=args.pretrained)
        patches = int(model.patch_embed.num_patches) + int(model.num_prefix_tokens)
        schedule = [patches] * (len(model.blocks) + 1)
        model_name = args.model
    return model.to(device).eval(), model_name, schedule


def analytical_blocks(model, schedule, batch, element_size):
    base = model.model if isinstance(model, CompactPSViT) else model
    rows = []
    for index, block in enumerate(base.blocks):
        n_in, n_out = int(schedule[index]), int(schedule[index + 1])
        dim = int(block.attn.qkv.in_features)
        heads = int(block.attn.num_heads)
        hidden = int(block.mlp.fc1.out_features)
        counts = {
            "input": batch * n_in * dim,
            "selected_residual": batch * n_out * dim,
            "q": batch * heads * n_out * (dim // heads),
            "k": batch * heads * n_in * (dim // heads),
            "v": batch * heads * n_in * (dim // heads),
            "attention_scores": batch * heads * n_out * n_in,
            "attention_probabilities": batch * heads * n_out * n_in,
            "attention_context": batch * n_out * dim,
            "mlp_hidden": batch * n_out * hidden,
            "block_output": batch * n_out * dim,
        }
        if isinstance(model, CompactPSViT) and args_global_select_mode(model) == "matmul":
            counts["selection_matrix"] = n_out * n_in
        bytes_by_tensor = {name: count * element_size for name, count in counts.items()}
        # This is a transparent upper-bound sum, not allocator liveness peak.
        rows.append({
            "block": index, "tokens_in": n_in, "tokens_out": n_out,
            "embed_dim": dim, "heads": heads, "mlp_hidden": hidden,
            "bytes_by_tensor": bytes_by_tensor,
            "sum_tensor_bytes_upper_bound": sum(bytes_by_tensor.values()),
            "attention_matrix_bytes": bytes_by_tensor["attention_scores"] + bytes_by_tensor["attention_probabilities"],
        })
    return rows


def args_global_select_mode(model):
    return getattr(model, "select_mode", "none")


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--arch", help="Patch Slimming architecture.json")
    source.add_argument("--reduced", help="structurally reduced checkpoint (reduced.pt)")
    source.add_argument("--model", help="plain timm model name")
    parser.add_argument("--weights", help="required with --arch")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--select-mode", choices=["index_select", "matmul"], default="index_select")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--sample-interval-ms", type=float, default=10.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.arch and not args.weights:
        parser.error("--weights is required with --arch")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    sampler = ProcessSampler(args.sample_interval_ms / 1000.0)
    sampler.start()
    model, model_name, schedule = load_model(args, device)
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    image = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device, dtype=dtype)

    module_rows = []
    call_count = defaultdict(int)
    handles = []

    def make_hook(name, module):
        def hook(_module, inputs, output):
            call_count[name] += 1
            inp, out = describe_tensors(inputs), describe_tensors(output)
            module_rows.append({
                "name": name, "type": module.__class__.__name__,
                "call": call_count[name], "input": inp, "output": out,
            })
        return hook

    for name, module in model.named_modules():
        if name and len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(make_hook(name, module)))

    cuda_before = {}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        cuda_before = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
        }
    with torch.inference_mode():
        for _ in range(args.warmups):
            model(image)
        # Hooks should describe one measured forward only.
        module_rows.clear(); call_count.clear()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(args.runs):
            output = model(image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

    for handle in handles:
        handle.remove()
    process_memory = sampler.stop()
    cuda = {}
    if device.type == "cuda":
        cuda = {
            **cuda_before,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "allocated_after_bytes": torch.cuda.memory_allocated(device),
            "reserved_after_bytes": torch.cuda.memory_reserved(device),
            "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        }

    # Aggregate repeated measured runs to per-forward module values.
    per_module = {}
    for row in module_rows:
        key = row["name"]
        if key not in per_module:
            per_module[key] = row
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    analytical = analytical_blocks(model, schedule, args.batch_size, image.element_size())
    payload = {
        "model": model_name, "architecture": args.arch, "weights": args.weights,
        "reduced_checkpoint": args.reduced,
        "device": str(device), "dtype": args.dtype, "batch_size": args.batch_size,
        "input_shape": list(image.shape), "output": describe_tensors(output),
        "token_schedule": schedule, "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "checkpoint_file_bytes": os.path.getsize(args.weights) if args.weights else (
            os.path.getsize(args.reduced) if args.reduced else None
        ),
        "latency": {"runs": args.runs, "total_seconds": elapsed,
                    "average_ms": elapsed * 1000.0 / args.runs,
                    "images_per_second": args.batch_size * args.runs / elapsed},
        "cuda": cuda, "process_memory": process_memory,
        "hook_activation": {
            "modules": list(per_module.values()),
            "sum_module_output_bytes": sum(x["output"]["bytes"] for x in per_module.values()),
            "largest_module_output_bytes": max((x["output"]["bytes"] for x in per_module.values()), default=0),
            "note": "sum counts produced tensors, not simultaneous live-memory peak",
        },
        "analytical_block_activation": analytical,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2)
    csv_path = os.path.splitext(args.output)[0] + "_blocks.csv"
    with open(csv_path, "w", newline="") as handle:
        fields = ["block", "tokens_in", "tokens_out", "embed_dim", "heads", "mlp_hidden",
                  "attention_matrix_bytes", "sum_tensor_bytes_upper_bound"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(analytical)
    print("model=%s schedule=%s" % (model_name, schedule))
    print("parameters=%.2f MiB buffers=%.2f MiB" % (parameter_bytes / 2**20, buffer_bytes / 2**20))
    if cuda:
        print("CUDA peak allocated=%.2f MiB reserved=%.2f MiB" % (
            cuda["peak_allocated_bytes"] / 2**20, cuda["peak_reserved_bytes"] / 2**20))
    print("process RSS peak=%.2f MiB USS peak=%.2f MiB" % (
        process_memory["rss_peak_bytes"] / 2**20, process_memory["uss_peak_bytes"] / 2**20))
    print("saved=%s\nblocks=%s" % (args.output, csv_path))


if __name__ == "__main__":
    main()
