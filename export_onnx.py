"""
Reduced / Token Pruned 모델 → ONNX 변환 CLI

reduce.py 출력(reduced.pt)과 train_token_pruning.py 출력(token_pruned_best.pt)을
모두 --input 하나로 받는다. token_pruned_best.pt는 "token_pruning" 키의 존재로
자동 판별해 EViT token pruning 그래프(TopK + Gather)까지 포함해서 export한다.

--npu-safe로 학습된 체크포인트("npu_safe": True 저장됨)는 자동으로 감지해서
pruning/token_pruning_npu.py의 NPU 호환 forward + batch=1 강제 + 레거시
(dynamo=False) exporter를 적용한다 — 플래그를 따로 챙길 필요 없음.

사용법:
    # --output을 생략하면 체크포인트 메타데이터(모델명/압축률/keep_rate)로
    # 자동 네이밍된 파일이 --input과 같은 폴더에 저장된다. 예:
    #   vit_tiny_30_final/reduced.pt              → vit_tiny_c30_reduced.onnx
    #   vit_tiny_30_final/token_prune70/*.pt       → vit_tiny_c30_token70.onnx
    # 폴더 밖으로 꺼내 한곳에 모아도 이름만으로 구분 가능하도록 하기 위함
    # (reduced.onnx / token_pruned.onnx 같은 이름은 여러 모델에서 겹침).

    # Stage 1 산출물 (channel pruning만) — 자동 네이밍
    python export_onnx.py --input ./output/vit_tiny_30_final/reduced.pt

    # Stage 2 산출물 (channel pruning + token pruning) — 자동 네이밍
    python export_onnx.py \\
        --input ./output/vit_tiny_30_final/token_prune70/token_pruned_last.pt

    # --output을 직접 지정하면 그걸 그대로 씀 (자동 네이밍 무시)
    python export_onnx.py --input reduced.pt --output custom_name.onnx

변환 후 검증:
    pip install onnx onnxruntime
    python export_onnx.py --input token_pruned_last.pt --verify

주의 (token pruning 포함 시):
    keep_rate가 고정 비율이라 남는 토큰 개수(k)는 입력에 관계없이 항상 동일한
    정수 상수이므로 그래프의 텐서 shape은 완전히 정적이다. 다만 "어떤" 토큰이
    선택되는지는 TopK+Gather로 런타임에 결정되므로, 이 두 op을 대상 NPU
    컴파일러가 지원하는지 사전에 반드시 확인해야 한다 (IMPLEMENTATION.md
    §14 참고).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import timm
from pruning.vit_reducing import apply_reduced_config
from pruning.token_pruning import apply_token_pruning


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Reduced / Token Pruned ViT → ONNX")
    p.add_argument("--input",      required=True,
                   help="reduced.pt 또는 token_pruned_*.pt 경로 (자동 판별)")
    p.add_argument("--output",     default="",
                   help="저장할 .onnx 파일 경로. 생략하면 체크포인트 메타데이터로 "
                        "자동 네이밍해서 --input과 같은 폴더에 저장 (default_output_path 참고)")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--opset",      type=int, default=17,
                   help="ONNX opset 버전 (기본 17). TopK/Gather는 opset 17에서 지원됨")
    p.add_argument("--batch-size", type=int, default=1,
                   help="고정 배치 크기. --dynamic 사용 시 이 값은 검증용으로만 쓰임")
    p.add_argument("--dynamic",    action="store_true", default=True,
                   help="배치 차원을 dynamic으로 export (추론 시 임의 배치 가능)")
    p.add_argument("--verify",     action="store_true",
                   help="onnxruntime 으로 출력값 일치 검증")
    p.add_argument("--num-threads", type=int, default=0,
                   help="ORT 스레드 수. 0=자동(CPU 코어 수). verify 및 출력 정보에 사용")
    p.add_argument("--npu-mode",   default="index_select",
                   choices=["index_select", "onehot_matmul"],
                   help="--npu-safe 체크포인트에서 gather를 어떻게 표현할지. "
                        "index_select=실측 여전히 미지원 확인됨 | "
                        "onehot_matmul=Equal+Cast+MatMul로 대체 (미검증, §14.6-1)")
    return p.parse_args()


def load_model(path: str, npu_mode: str = "index_select") -> tuple[torch.nn.Module, dict, bool, bool]:
    """reduced.pt / token_pruned_best.pt 공용 로더.

    체크포인트에 "npu_safe": True가 저장돼 있으면(train_token_pruning.py
    --npu-safe로 학습된 경우) pruning/token_pruning_npu.py의 NPU 호환 forward를
    자동으로 붙이고 export mode(batch=1, npu_mode로 gather 대체 전략 선택)까지
    켜준다 — export할 때 플래그를 따로 챙길 필요가 없도록.

    반환: (model, ckpt, has_token_pruning, npu_safe)
    """
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    model = timm.create_model(ckpt["model_name"], pretrained=False)
    apply_reduced_config(model, ckpt["mlp_dims"])

    has_token_pruning = "token_pruning" in ckpt
    npu_safe = bool(ckpt.get("npu_safe", False))
    if has_token_pruning:
        tp_cfg = ckpt["token_pruning"]
        if npu_safe:
            from pruning.token_pruning_npu import apply_token_pruning_npu, set_npu_export_mode
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
    """'vit_tiny_patch16_224' → 'vit_tiny' (파일명 간결화용)."""
    return name.replace("_patch16_224", "")


def default_output_path(
    input_path: str, ckpt: dict, has_token_pruning: bool,
    npu_safe: bool = False, npu_mode: str = "index_select",
) -> str:
    """--output 생략 시 자동 네이밍.

    체크포인트에 이미 들어있는 값(model_name, compression_rate 또는
    n_params_before/after, token_pruning.base_keep_rate)만 사용한다 —
    폴더 이름 파싱에 의존하지 않으므로 어디로 옮겨도 파일명만으로 구분 가능하다.

        reduced.pt        → vit_tiny_c30_reduced.onnx
        token_pruned_*.pt → vit_tiny_c30_token70.onnx
                             (npu_safe면 _npusafe_<mode> 접미사 추가 — 두 mode를
                             같은 폴더에서 비교할 수 있도록 파일명이 겹치지 않게)
    """
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
            print("[NPU-SAFE] --dynamic 무시 → batch 고정 (구조 자체가 batch=1 전용이라 "
                  "가변 배치로 선언하면 그래프와 선언이 모순됨)")
            dynamic = False

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Export] model={ckpt['model_name']}")
    if has_token_pruning:
        tp_cfg = ckpt["token_pruning"]
        print(f"         params={n_params:,}  (channel+token pruning 적용{'  [NPU-SAFE]' if npu_safe else ''})")
        print(f"         token_pruning: prune_layers={tp_cfg['prune_layers']}  "
              f"base_keep_rate={tp_cfg['base_keep_rate']}  fuse_token={tp_cfg['fuse_token']}")
    else:
        print(f"         params={n_params:,}  (compression={ckpt.get('compression_rate', '?'):.2f}%)")
    print(f"         opset={args.opset}  dynamic={dynamic}")

    dummy = torch.zeros(args.batch_size, 3, args.input_size, args.input_size)

    # dynamic_axes: batch 차원을 가변으로 설정
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input":  {0: "batch"},
            "output": {0: "batch"},
        }

    # npu_safe면 dynamo 기반 exporter 대신 레거시(TorchScript 기반) exporter를 강제한다.
    # 실제로 겪은 문제(그래프 output이 여러 개로 노출됨, /blocks/blocks.N/TopK_output_0 등)가
    # dynamo exporter 쪽에서 발생한 것으로 추정되어서다 — 확실친 않아 아래에서 export 직후
    # output 개수를 바로 확인한다.
    export_kwargs = dict(
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,   # 명시적 eval mode
    )
    if npu_safe:
        export_kwargs["dynamo"] = False

    print(f"\nExporting → {output_path} ...")
    torch.onnx.export(model, dummy, output_path, **export_kwargs)
    print("Export 완료.")

    # ── ONNX 구조 검증 ─────────────────────────────────────────────────────────
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX graph check: OK")
        n_nodes = len(onnx_model.graph.node)
        print(f"  graph nodes: {n_nodes}")

        out_names = [o.name for o in onnx_model.graph.output]
        print(f"  graph outputs ({len(out_names)}): {out_names}")
        if len(out_names) != 1:
            print("  ⚠ output이 1개가 아니다 — qbcompiler quantizer가 어느 노드를 "
                  "기준으로 calibration할지 못 정해서 실패할 수 있다 (이전에 겪은 문제).")
        if npu_safe:
            op_types = {n.op_type for n in onnx_model.graph.node}
            always_bad = ["ScatterElements", "GatherElements"]
            mode_bad = ["Gather"] if args.npu_mode == "onehot_matmul" else []
            for bad_op in always_bad + mode_bad:
                if bad_op in op_types:
                    print(f"  ⚠ {bad_op}가 그래프에 여전히 남아있다 (npu_mode={args.npu_mode}) — "
                          f"NPU-safe 수정이 기대만큼 안 먹힌 부분이 있다는 뜻, qbcompiler "
                          f"이전에 여기서 먼저 확인됨.")
            if args.npu_mode == "onehot_matmul":
                for expect_op in ("Equal", "MatMul"):
                    if expect_op not in op_types:
                        print(f"  ⚠ onehot_matmul 모드인데 {expect_op}가 그래프에 없다 — "
                              f"const-fold로 사라졌을 수 있음(=idx가 진짜 동적 값이 아닐 수도 있다는 신호).")
    except ImportError:
        print("(onnx 미설치 — graph check 생략. pip install onnx)")

    # ── onnxruntime 출력값 일치 검증 + 스레드 설정 확인 ───────────────────────
    if args.verify:
        try:
            import onnxruntime as ort
            import numpy as np

            # 멀티코어 세션 설정
            # intra_op_num_threads: 하나의 op(행렬곱 등) 내 병렬 스레드 수
            # inter_op_num_threads: 독립적인 op들을 동시에 실행하는 스레드 수
            # 둘 다 0 = ORT가 자동으로 CPU 코어 수에 맞게 결정
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = args.num_threads   # 0 = 자동
            sess_opts.inter_op_num_threads = args.num_threads   # 0 = 자동
            sess_opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL       # 최대 그래프 최적화
            )

            sess = ort.InferenceSession(
                output_path,
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
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

            # 스레드 설정 확인 출력
            n_threads = args.num_threads if args.num_threads > 0 else "auto (CPU cores)"
            print(f"\n[Threading] intra_op={n_threads}  inter_op={n_threads}")
            print(f"            graph_optimization=ORT_ENABLE_ALL")

        except ImportError:
            print("(onnxruntime 미설치 — 출력 검증 생략. pip install onnxruntime)")

    # ── 파일 크기 출력 ─────────────────────────────────────────────────────────
    import os
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\n출력 파일: {output_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
