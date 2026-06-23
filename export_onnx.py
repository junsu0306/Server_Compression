"""
Reduced 모델 → ONNX 변환 CLI

사용법:
    python export_onnx.py \\
        --reduced  ./output/vit_small_prune50/reduced.pt \\
        --output   ./output/vit_small_prune50/reduced.onnx

변환 후 검증:
    pip install onnx onnxruntime
    python export_onnx.py --reduced reduced.pt --output reduced.onnx --verify
"""

from __future__ import annotations

import argparse
import torch
import timm
from pruning.vit_reducing import apply_reduced_config


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Reduced ViT → ONNX")
    p.add_argument("--reduced",    required=True, help="reduce.py 출력 파일 (reduced.pt)")
    p.add_argument("--output",     required=True, help="저장할 .onnx 파일 경로")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--opset",      type=int, default=17,
                   help="ONNX opset 버전 (기본 17)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="고정 배치 크기. --dynamic 사용 시 이 값은 검증용으로만 쓰임")
    p.add_argument("--dynamic",    action="store_true", default=True,
                   help="배치 차원을 dynamic으로 export (추론 시 임의 배치 가능)")
    p.add_argument("--verify",     action="store_true",
                   help="onnxruntime 으로 출력값 일치 검증")
    return p.parse_args()


def load_reduced_model(reduced_path: str) -> tuple[torch.nn.Module, dict]:
    ckpt  = torch.load(reduced_path, map_location="cpu", weights_only=False)
    model = timm.create_model(ckpt["model_name"], pretrained=False)
    apply_reduced_config(model, ckpt["mlp_dims"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def main():
    args  = get_args()
    model, ckpt = load_reduced_model(args.reduced)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Export] model={ckpt['model_name']}")
    print(f"         params={n_params:,}  (compression={ckpt.get('compression_rate', '?'):.2f}%)")
    print(f"         opset={args.opset}  dynamic={args.dynamic}")

    dummy = torch.zeros(args.batch_size, 3, args.input_size, args.input_size)

    # dynamic_axes: batch 차원을 가변으로 설정
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input":  {0: "batch"},
            "output": {0: "batch"},
        }

    print(f"\nExporting → {args.output} ...")
    torch.onnx.export(
        model,
        dummy,
        args.output,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    print("Export 완료.")

    # ── ONNX 구조 검증 ─────────────────────────────────────────────────────────
    try:
        import onnx
        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        print("ONNX graph check: OK")
    except ImportError:
        print("(onnx 미설치 — graph check 생략. pip install onnx)")

    # ── onnxruntime 출력값 일치 검증 ───────────────────────────────────────────
    if args.verify:
        try:
            import onnxruntime as ort
            import numpy as np

            sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
            inp  = dummy.numpy()

            with torch.no_grad():
                pt_out = model(dummy).numpy()
            ort_out = sess.run(["output"], {"input": inp})[0]

            max_diff = float(np.abs(pt_out - ort_out).max())
            print(f"\n[Verify] PyTorch vs ONNX Runtime 최대 차이: {max_diff:.2e}")
            if max_diff < 1e-4:
                print("         ✓ 출력값 일치 (정상)")
            else:
                print("         ⚠ 출력값 차이가 큼 — opset 버전 또는 모델 구조 확인 필요")
        except ImportError:
            print("(onnxruntime 미설치 — 출력 검증 생략. pip install onnxruntime)")

    # ── 파일 크기 출력 ─────────────────────────────────────────────────────────
    import os
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\n출력 파일: {args.output}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
