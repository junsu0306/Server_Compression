"""
[ARCHIVED] Token Pruned 모델 → ONNX 변환 (EViT 실험 전용).

이 파일은 token_pruning_archive 실험 재현용이다. 루트의 export_onnx.py는
Stage 1(channel pruning, reduced.pt) 전용으로 축소되었고, token pruning(EViT)
산출물(token_pruned_*.pt)과 NPU 우회(--npu-safe/--npu-mode)를 다루는 로직은
여기로 옮겼다. 배경/실패 분석은 같은 폴더의 TOKEN_PRUNING.md 참고.

사용법 (repo 루트에서 실행):
    python token_pruning_archive/export_onnx.py \\
        --input ./output/vit_tiny_30_final/token_prune70/token_pruned_last.pt --verify

    # NPU 우회 모드 (미검증 — TOKEN_PRUNING.md §9.2)
    python token_pruning_archive/export_onnx.py \\
        --input .../token_pruned_best.pt --npu-mode onehot_matmul --verify

주의: TopK + 런타임 인덱스 gather 때문에 Aries2/qbcompiler 컴파일은 실패로
확인됐다 (TOKEN_PRUNING.md §9.2). 이 스크립트는 GPU/CPU(onnxruntime) 검증 및
기록 보존용이다.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

# 아카이브 경로 처리: 루트(engine/pruning) + 같은 폴더(token_pruning*) 둘 다 import 가능하게
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import timm
from pruning.vit_reducing import apply_reduced_config
from token_pruning import apply_token_pruning       # sibling


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("[ARCHIVED] Token Pruned ViT → ONNX")
    p.add_argument("--input",      required=True,
                   help="token_pruned_*.pt 경로 ('token_pruning' 키 필수)")
    p.add_argument("--output",     default="",
                   help="저장 경로. 생략 시 체크포인트 메타데이터로 자동 네이밍")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--opset",      type=int, default=17)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--dynamic",    action="store_true", default=True)
    p.add_argument("--verify",     action="store_true")
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--npu-mode",   default="index_select",
                   choices=["index_select", "onehot_matmul"],
                   help="--npu-safe 체크포인트에서 gather 표현 방식 "
                        "(index_select/onehot_matmul — 둘 다 실측 실패, §9.2)")
    return p.parse_args()


def load_model(path: str, npu_mode: str = "index_select") -> tuple[torch.nn.Module, dict, bool, bool]:
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    model = timm.create_model(ckpt["model_name"], pretrained=False)
    apply_reduced_config(model, ckpt["mlp_dims"])

    has_token_pruning = "token_pruning" in ckpt
    npu_safe = bool(ckpt.get("npu_safe", False))
    if has_token_pruning:
        tp_cfg = ckpt["token_pruning"]
        if npu_safe:
            from token_pruning_npu import apply_token_pruning_npu, set_npu_export_mode
            apply_token_pruning_npu(
                model,
                prune_layers=tp_cfg["prune_layers"],
                base_keep_rate=tp_cfg["base_keep_rate"],
                fuse_token=tp_cfg["fuse_token"],
            )
            set_npu_export_mode(model, True, mode=npu_mode)
            print(f"[NPU-SAFE] token_pruning_npu.py forward + export mode(batch=1, npu_mode={npu_mode}) 적용")
        else:
            apply_token_pruning(
                model,
                prune_layers=tp_cfg["prune_layers"],
                base_keep_rate=tp_cfg["base_keep_rate"],
                fuse_token=tp_cfg["fuse_token"],
            )

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt, has_token_pruning, npu_safe


def _short_model_name(name: str) -> str:
    return name.replace("_patch16_224", "")


def default_output_path(input_path, ckpt, has_token_pruning, npu_safe=False, npu_mode="index_select"):
    model = _short_model_name(ckpt["model_name"])
    if has_token_pruning:
        n_before = ckpt.get("n_params_before", 0)
        n_after  = ckpt.get("n_params_after", 0)
        c_pct = round(100 * (1 - n_after / n_before)) if n_before else 0
        keep_pct = round(ckpt["token_pruning"]["base_keep_rate"] * 100)
        suffix = f"_npusafe_{npu_mode}" if npu_safe else ""
        name = f"{model}_c{c_pct}_token{keep_pct}{suffix}.onnx"
    else:
        c_pct = round(ckpt.get("compression_rate", 0))
        name = f"{model}_c{c_pct}_reduced.onnx"
    return str(Path(input_path).parent / name)


def main():
    args  = get_args()
    model, ckpt, has_token_pruning, npu_safe = load_model(args.input, npu_mode=args.npu_mode)
    output_path = args.output or default_output_path(
        args.input, ckpt, has_token_pruning, npu_safe, args.npu_mode
    )

    dynamic = args.dynamic
    if npu_safe:
        if args.batch_size != 1:
            print(f"[NPU-SAFE] --batch-size {args.batch_size} → 1로 강제 (batch=1 전용 경로 필요)")
            args.batch_size = 1
        if dynamic:
            print("[NPU-SAFE] --dynamic 무시 → batch 고정")
            dynamic = False

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Export] model={ckpt['model_name']}")
    if has_token_pruning:
        tp_cfg = ckpt["token_pruning"]
        print(f"         params={n_params:,}  (channel+token pruning{'  [NPU-SAFE]' if npu_safe else ''})")
        print(f"         token_pruning: prune_layers={tp_cfg['prune_layers']}  "
              f"base_keep_rate={tp_cfg['base_keep_rate']}  fuse_token={tp_cfg['fuse_token']}")
    else:
        print(f"         params={n_params:,}")
    print(f"         opset={args.opset}  dynamic={dynamic}")

    dummy = torch.zeros(args.batch_size, 3, args.input_size, args.input_size)
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic else None

    export_kwargs = dict(
        opset_version=args.opset,
        input_names=["input"], output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )
    if npu_safe:
        export_kwargs["dynamo"] = False

    print(f"\nExporting → {output_path} ...")
    torch.onnx.export(model, dummy, output_path, **export_kwargs)
    print("Export 완료.")

    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX graph check: OK")
        out_names = [o.name for o in onnx_model.graph.output]
        print(f"  graph outputs ({len(out_names)}): {out_names}")
        if npu_safe:
            op_types = {n.op_type for n in onnx_model.graph.node}
            always_bad = ["ScatterElements", "GatherElements"]
            mode_bad = ["Gather"] if args.npu_mode == "onehot_matmul" else []
            for bad_op in always_bad + mode_bad:
                if bad_op in op_types:
                    print(f"  ⚠ {bad_op} 여전히 존재 (npu_mode={args.npu_mode})")
    except ImportError:
        print("(onnx 미설치 — graph check 생략)")

    if args.verify:
        try:
            import onnxruntime as ort
            import numpy as np
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = args.num_threads
            sess_opts.inter_op_num_threads = args.num_threads
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(output_path, sess_options=sess_opts,
                                        providers=["CPUExecutionProvider"])
            with torch.no_grad():
                pt_out = model(dummy).numpy()
            ort_out = sess.run(["output"], {"input": dummy.numpy()})[0]
            max_diff = float(np.abs(pt_out - ort_out).max())
            print(f"\n[Verify] PyTorch vs ONNX Runtime 최대 차이: {max_diff:.2e}"
                  f"  {'✓ 일치' if max_diff < 1e-4 else '⚠ 차이 큼'}")
        except ImportError:
            print("(onnxruntime 미설치 — 검증 생략)")

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\n출력 파일: {output_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
