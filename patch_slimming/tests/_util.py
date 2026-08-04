"""테스트 공용 — sys.path 설정 + PASS/FAIL 러너."""
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_PS = os.path.dirname(_TESTS)               # patch_slimming/
_ROOT = os.path.dirname(_PS)                 # repo 루트
for p in (_PS, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


def run_tests(namespace: dict):
    """namespace의 test_* 함수를 모두 실행하고 PASS/FAIL 요약."""
    fns = {k: v for k, v in namespace.items() if k.startswith("test_") and callable(v)}
    failed = 0
    for name, fn in fns.items():
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{'='*50}\n{len(fns)-failed}/{len(fns)} passed"
          + (f", {failed} FAILED" if failed else "") + f"\n{'='*50}")
    return failed == 0
