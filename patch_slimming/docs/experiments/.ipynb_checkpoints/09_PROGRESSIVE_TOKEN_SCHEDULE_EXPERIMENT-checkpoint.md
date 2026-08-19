# 09. Progressive token schedule experiment

## 목적

논문 Figure 5의 정확한 layer별 token 수를 복사하지 않으면서, 얕은 layer에서 많은 token을
유지하고 깊어질수록 점진적으로 줄어드는 경향을 structural model에도 부여한다.

## 방법

기존 조건은 그대로 유지한다.

- paper path-energy ranking
- top-down nested mask와 다음 block 출력 reconstruction
- 독립 score/train/eval subset 1,024장씩
- mean per-sample relative Frobenius squared, epsilon 0.02

가장 깊은 searchable block은 error search 결과를 그대로 사용하고, 그보다 얕은 각 block은
다음 block의 base mask보다 최소 5개 token을 더 유지한다. 추가 token은 임의 위치가 아니라
해당 layer의 path-energy ranking 상위 token이다. Error가 epsilon을 넘으면 기존처럼 10개씩
더 추가한다.

이 제약은 논문 Algorithm 1 자체가 아니라 배포용 architecture prior다. 마지막 block의
CLS-only 출력은 분류 head contract이므로 점진적 제약의 예외다.

## 실행 설정

- 모델: structural Tiny-30
- GPU: 6
- W&B project: `vit-patch-slimming-progressive-search`
- runs: `tiny30-relative002-progressive-g5`, `tiny30-relative002-progressive-g3`

## 결과

| 방식 | Token schedule | Body MAC 감소 | Pre-FT Top-1 | Baseline 하락 |
|---|---|---:|---:|---:|
| Relative, 제약 없음 | `[131,131,131,121,121,121,121,121,121,121,121,1]` | 42.59% | 68.482% | -5.048%p |
| Progressive +5 | `[171,166,161,156,151,146,141,136,131,126,121,1]` | 30.04% | 70.220% | -3.310%p |
| **Progressive +3** | **`[151,148,145,142,139,136,133,130,127,124,121,1]`** | **35.77%** | **69.626%** | **-3.904%p** |
| Manual medium | `[197,197,157,157,157,118,118,118,79,79,79,1]` | 34.75% | 66.078% | -7.448%p |

표의 schedule은 197-token embedding 입력 이후 각 block의 출력 token 수다. 마지막 1은
classifier CLS-only 출력이다.

### W&B

- +5 search: `s5qy8gl1`
- +5 full ImageNet validation: `tji72ibo`
- +3 search: `1hhnwi9z`
- +3 full ImageNet validation: `k4flyuz6`
- project: `vit-patch-slimming-progressive-search`

## 해석과 선택

+5는 정확도가 가장 높지만 MAC 감소가 manual medium보다 작아 다소 보수적이다. +3은
manual medium보다 MAC을 1.02%p 더 줄이면서 Pre-FT Top-1이 3.55%p 높고, 모든 searchable
layer에서 정확히 점진적으로 감소한다. 따라서 Tiny-30의 현재 권장 progressive prior는
`min_growth_per_shallower_layer=3`이다.

각 layer의 +3 token은 reconstruction error 때문에 추가된 것이 아니라 architecture prior의
최솟값이다. 다만 모든 mask가 독립 eval subset에서 epsilon 0.02를 통과했으며, token 위치는
각 layer의 path-energy ranking으로 선택됐다. 따라서 결과를 순수 논문 Algorithm 1 재현이라고
부르면 안 되고, `paper score + reconstruction guard + progressive deployment prior`의 hybrid로
표기해야 한다.

이 숫자 3을 다른 structural 모델에 그대로 고정할 필요는 없다. embedding token 수는 같아도
width와 structural FFN이 다르므로 Small 및 50% 모델에서는 +2/+3/+5 중 accuracy-MAC Pareto를
다시 확인해야 한다.
