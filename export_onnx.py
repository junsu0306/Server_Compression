"""
Reduced 모델 (reduced.pt) → ONNX 변환 CLI  [Stage 1 — channel pruning 전용]

reduce.py 출력(reduced.pt)을 받아 ONNX로 변환한다. Stage 2(EViT token pruning)
산출물(token_pruned_*.pt) 변환과 NPU 우회 로직은 token_pruning_archive/export_onnx.py
로 분리했다 (해당 실험은 NPU 컴파일 실패로 아카이브됨 — token_pruning_archive/
TOKEN_PRUNING.md 참고).

사용법:
    # --output 생략 시 체크포인트 메타데이터(모델명/압축률)로 자동 네이밍
    python export_onnx.py --input ./output/vit_tiny_30_final/reduced.pt --verify
    #   → ./output/vit_tiny_30_final/vit_tiny_c30_reduced.onnx

    # 경로 직접 지정
    python export_onnx.py --input reduced.pt --output custom.onnx

변환 후 검증:
    pip install onnx onnxruntime
    python export_onnx.py --input reduced.pt --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import timm
from pruning.vit_reducing import apply_reduced_config


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Reduced ViT → ONNX (Stage 1)")
    p.add_argument("--input",      required=True,
                   help="reduce.py 출력 파일 (reduced.pt) 경로")
    p.add_argument("--output",     default="",
                   help="저장할 .onnx 경로. 생략 시 체크포인트 메타데이터로 자동 네이밍")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--opset",      type=int, default=17,
                   help="ONNX opset 버전 (기본 17)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="고정 배치 크기. --dynamic 사용 시 검증용으로만 쓰임")
    p.add_argument("--dynamic",    action="store_true", default=True,
                   help="배치 차원을 dynamic으로 export")
    p.add_argument("--verify",     action="store_true",
                   help="onnxruntime 으로 출력값 일치 검증")
    p.add_argument("--num-threads", type=int, default=0,
                   help="ORT 스레드 수. 0=자동(CPU 코어 수)")
    return p.parse_args()


def load_model(path: str) -> tuple[torch.nn.Module, dict]:
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    model = timm.create_model(ckpt["model_name"], pretrained=False)
    apply_reduced_config(model, ckpt["mlp_dims"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _short_model_name(name: str) -> str:
    """'vit_tiny_patch16_224' → 'vit_tiny' (파일명 간결화용)."""
    return name.replace("_patch16_224", "")


def default_output_path(input_path: str, ckpt: dict) -> str:
    """--output 생략 시 자동 네이밍.

    체크포인트 안의 compression_rate만 사용하므로 폴더 위치와 무관하게 파일명만으로
    구분된다 (예: vit_tiny_c30_reduced.onnx).
    """
    model = _short_model_name(ckpt["model_name"])
    c_pct = round(ckpt.get("compression_rate", 0))
    return str(Path(input_path).parent / f"{model}_c{c_pct}_reduced.onnx")


def main():
    args  = get_args()
    model, ckpt = load_model(args.input)
    output_path = args.output or default_output_path(args.input, ckpt)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Export] model={ckpt['model_name']}")
    print(f"         params={n_params:,}  (compression={ckpt.get('compression_rate', '?'):.2f}%)")
    print(f"         opset={args.opset}  dynamic={args.dynamic}")

    dummy = torch.zeros(args.batch_size, 3, args.input_size, args.input_size)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    print(f"\nExporting → {output_path} ...")
    torch.onnx.export(
        model, dummy, output_path,
        opset_version=args.opset,
        input_names=["input"], output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )
    print("Export 완료.")

    # ── ONNX 구조 검증 ─────────────────────────────────────────────────────────
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX graph check: OK")
        print(f"  graph nodes: {len(onnx_model.graph.node)}")
    except ImportError:
        print("(onnx 미설치 — graph check 생략. pip install onnx)")

    # ── onnxruntime 출력값 일치 검증 ──────────────────────────────────────────
    if args.verify:
        try:
            import onnxruntime as ort
            import numpy as np

            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = args.num_threads   # 0 = 자동
            sess_opts.inter_op_num_threads = args.num_threads
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            sess = ort.InferenceSession(
                output_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
            )
            with torch.no_grad():
                pt_out = model(dummy).numpy()
            ort_out = sess.run(["output"], {"input": dummy.numpy()})[0]

            max_diff = float(np.abs(pt_out - ort_out).max())
            print(f"\n[Verify] PyTorch vs ONNX Runtime 최대 차이: {max_diff:.2e}")
            print("         ✓ 출력값 일치 (정상)" if max_diff < 1e-4
                  else "         ⚠ 출력값 차이가 큼 — opset/모델 구조 확인 필요")

            n_threads = args.num_threads if args.num_threads > 0 else "auto (CPU cores)"
            print(f"\n[Threading] intra_op={n_threads}  inter_op={n_threads}  "
                  f"graph_optimization=ORT_ENABLE_ALL")
        except ImportError:
            print("(onnxruntime 미설치 — 출력 검증 생략. pip install onnxruntime)")

    import os
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\n출력 파일: {output_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
