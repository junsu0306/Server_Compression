# Tiny30 3-way 메모리·Activation 비교

측정일: 2026-08-10, GPU 5, FP32, batch 1, 입력 224×224, warmup 10회,
측정 forward 50회. 세 모델을 동일한 `profile_memory.py`로 연속 측정했다.

## 비교 대상

1. Baseline: pruning 전 `vit_tiny_patch16_224`
2. Reduced: `output/vit_tiny_30_final/reduced.pt`
3. Patch Slimming: Tiny30 reduced + paper rel002, matmul 선택행렬

Patch Slimming token schedule:

```text
197 → 141 → 141 → 131 → 121 → 121 → 121
    → 121 → 121 → 121 → 121 → 121 → 1
```

## 결과

| 지표 | Baseline | Tiny30 reduced | Tiny30 + Patch Slimming |
|---|---:|---:|---:|
| Parameter | 21.81 MiB | 15.27 MiB | 15.27 MiB |
| CUDA peak allocated | 32.83 MiB | 26.17 MiB | 26.90 MiB |
| CUDA peak reserved | 48.00 MiB | 40.00 MiB | 40.00 MiB |
| Hook module-output 합계 | 54.54 MiB | 41.16 MiB | 22.10 MiB |
| Block tensor-volume 합계 상한 | 29.71 MiB | 26.36 MiB | 14.73 MiB |
| Attention score+prob 합계 | 10.66 MiB | 10.66 MiB | 4.22 MiB |
| 평균 latency | 52.24 ms | 47.63 ms | 84.15 ms |
| Throughput | 19.14 img/s | 21.00 img/s | 11.88 img/s |

Baseline 대비 Tiny30 reduced는 parameter 30.0%, CUDA peak allocated 20.3%, hook
activation 24.5%를 줄였다. 토큰 수가 197로 유지되므로 attention matrix는 감소하지
않는다.

Baseline 대비 Patch Slimming 모델은 parameter 30.0%, hook activation 59.5%, block
tensor-volume 50.4%, attention matrix 60.4%를 줄였다. Reduced 대비로도 hook activation
46.3%, block tensor-volume 44.1%, attention matrix 60.4%가 감소했다.

CUDA peak allocated는 Patch Slimming이 reduced보다 0.73 MiB 높다. PyTorch matmul
선택행렬과 임시 allocator workspace가 추가되며, batch 1에서는 모델 weight가 peak의 큰
부분을 차지하기 때문이다. 반면 실제 token-dependent activation 지표는 명확히 감소한다.

Patch Slimming의 GPU latency가 느린 결과는 작은 rectangular MatMul을 여러 번 실행하고
선택행렬 MatMul을 추가하는 현재 GPU 구현의 kernel launch overhead를 포함한다. 이는
activation 절감과 별개의 지표이며 NPU 속도를 직접 예측하지 않는다. GPU 5에 다른 작업이
상주한 환경이므로 절대 latency보다 동일 측정 내 상대 경향만 참고하고, 최종 속도는 MXQ를
실제 NPU에서 별도로 측정한다.

프로세스 RSS/USS는 Python, timm, hook 결과 보관량까지 포함하고 Patch Slimming hook 수와
결과 구조의 영향을 받으므로 모델 activation 비교 지표로 사용하지 않는다. 실제 장치
allocator peak는 CUDA 항목, activation 구성 변화는 hook 및 analytical block 항목을 쓴다.

## 원본 결과

- `output/memory_profiles/vit_tiny_baseline_b1_fp32_gpu5.json`
- `output/memory_profiles/tiny30_reduced_b1_fp32_gpu5.json`
- `output/memory_profiles/tiny30_paper_rel002_matmul_b1_fp32_gpu5.json`
- 각 JSON과 같은 이름의 `_blocks.csv`
