"""Persist logs and exit codes for the CPU/export regression gates."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    folder = Path(args.output); folder.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = ""; env["OMP_NUM_THREADS"] = "2"
    commands = [("unit", ["-m", "unittest", "discover", "-s", "tests", "-v"]),
                ("patch", ["patch_slimming/tests/run_all.py"]),
                ("export", ["tests/verify_exports.py"])]
    results = []
    for name, command in commands:
        with (folder / f"{name}.log").open("w") as log:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        results.append({"check": name, "command": command, "exit_code": result.returncode})
        print(f"Preflight {name}: exit={result.returncode}", flush=True)
        (folder / "checks.json").write_text(json.dumps({"updated_utc": datetime.now(timezone.utc).isoformat(),
                                                      "checks": results}, indent=2) + "\n")
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
