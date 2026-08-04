"""
Phase 7 — Compact PS-ViT → NPU-safe ONNX (SPEC §14, §17 Phase 7).

§14.0 실측 규칙 반영:
    - 토큰 선택은 상수 선택행렬 MatMul (Gather 금지) → bake_for_export()
    - 입력은 NHWC [1,224,224,3] (Mobilint calibration 레이아웃)
    - rectangular attention 그대로 (Aries2 지원 확인됨)

사용법:
    python patch_slimming/export_ps_onnx.py \
        --arch ./output/ps_deit_tiny/architecture.json \
        --weights ./output/ps_deit_tiny/compact_finetuned_best.pt \
        --output ./output/ps_deit_tiny/ps_compact_npu.onnx --verify
"""

from __future__ import annotations

import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import timm

from psvit.architecture import ArchitectureSpec
from psvit.compact import CompactPSViT


def main():
    ap = argparse.ArgumentParser("Export CompactPSViT → NPU-safe ONNX")
    ap.add_argument("--arch", required=True)
    ap.add_argument("--weights", required=True, help="searched_student.pt 또는 compact_finetuned_*.pt")
    ap.add_argument("--output", default="")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    spec = ArchitectureSpec.from_json(args.arch); spec.validate()
    state = torch.load(args.weights, map_location="cpu", weights_only=False)["model"]
    base = timm.create_model(spec.model_name, pretrained=False,
                             num_classes=state["head.weight"].shape[0])
    base.load_state_dict(state, strict=True); base.eval()

    keep_ids = spec.keep_ids_list()
    # NHWC 입력 + matmul 선택행렬 (NPU-safe)
    compact = CompactPSViT(base, keep_ids, select_mode="matmul", nhwc_input=True).eval()
    compact.bake_for_export(device=torch.device("cpu"), dtype=torch.float32)

    out_path = args.output or os.path.join(os.path.dirname(args.arch), "ps_compact_npu.onnx")
    dummy = torch.zeros(1, args.img_size, args.img_size, 3)   # NHWC

    # index_select vs matmul 수치 동일성 자체 확인 (bake 전/후 비교)
    with torch.no_grad():
        ref = CompactPSViT(base, keep_ids, select_mode="index_select", nhwc_input=True).eval()(dummy)
        got = compact(dummy)
        max_diff = (ref - got).abs().max().item()
    print(f"[check] index_select vs matmul 최대차이 = {max_diff:.2e}  (동일해야 정상)")
    print(f"[export] token schedule = {compact.token_schedule()}")

    print(f"\nExporting → {out_path} (opset {args.opset}, NHWC 입력 고정) ...")
    torch.onnx.export(
        compact, dummy, out_path,
        opset_version=args.opset,
        input_names=["input"], output_names=["output"],
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )
    print("Export 완료.")

    try:
        import onnx
        m = onnx.load(out_path)
        onnx.checker.check_model(m)
        ops = sorted({n.op_type for n in m.graph.node})
        print(f"ONNX graph check OK. op types: {ops}")
        for banned in ("Gather", "GatherElements", "ScatterElements", "TopK", "Equal"):
            if banned in ops:
                print(f"  ⚠ {banned} 존재 — NPU 컴파일 실패 위험 (§14.0: 상수 선택행렬 matmul이어야 함)")
    except ImportError:
        print("(onnx 미설치 — graph check 생략)")

    if args.verify:
        try:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
            with torch.no_grad():
                pt = compact(dummy).numpy()
            ortout = sess.run(["output"], {"input": dummy.numpy()})[0]
            d = float(np.abs(pt - ortout).max())
            print(f"[verify] PyTorch vs ORT 최대차이 = {d:.2e}  {'✓' if d < 1e-4 else '⚠'}")
        except ImportError:
            print("(onnxruntime 미설치 — verify 생략)")

    print(f"\n출력: {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print("→ 이 .onnx를 qbcompiler에 넣으면 된다 (NHWC 입력, 상수 선택행렬 matmul, rectangular attention).")


if __name__ == "__main__":
    main()
