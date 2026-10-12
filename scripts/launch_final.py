"""Detached supervisor for one independent final experiment per configured GPU."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    temp = path.with_suffix(".tmp"); temp.write_text(json.dumps(value, indent=2) + "\n"); temp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default="configs/final.yaml")
    ap.add_argument("--detach", action="store_true")
    args = ap.parse_args()
    config = (ROOT / args.matrix).resolve()
    matrix = yaml.safe_load(config.read_text())
    root = ROOT / matrix["output_root"]
    root.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (root / "supervisor.log").open("a") as stream:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--matrix", str(config)],
                                     cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(f"Supervisor PID={child.pid}; log={root / 'supervisor.log'}"); return
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = root / "provenance/source"
        paths = [ROOT / "README.md", ROOT / ".gitignore", ROOT / "requirements.txt"]
        for folder in ("pruning", "patch_slimming", "scripts", "configs", "tests", "docs"):
            paths += [p for p in (ROOT / folder).rglob("*") if p.is_file() and
                      p.suffix in (".py", ".yaml", ".md") and "__pycache__" not in p.parts]
        current = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        manifest = root / "provenance/source_manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != current:
                raise ValueError("Source changed since launch. Use a new output root for a new implementation.")
        else:
            for path in paths:
                target = source / path.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            write(manifest, current)
        environment = {"python": sys.version, "executable": sys.executable,
                       "packages": {name: importlib.metadata.version(name) for name in
                                    ("torch", "torchvision", "timm", "numpy", "onnx", "onnxruntime", "wandb")},
                       "gpu_status": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version",
                                                              "--format=csv"], text=True)}
        write(root / "provenance/environment.json", environment)
        subprocess.run([sys.executable, str(ROOT / "tests/run_preflight.py"),
                        "--output", str(root / "provenance/preflight")], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/run_pipeline.py"), "--matrix", str(config),
                        "--phase", "materialize"], check=True)
        children = {}; streams = []
        for gpu in sorted({v["gpu"] for v in matrix["models"].values()}):
            stream = (root / f"gpu{gpu}_pipeline.log").open("a"); streams.append(stream)
            children[gpu] = subprocess.Popen([sys.executable, str(ROOT / "scripts/run_pipeline.py"),
                                             "--matrix", str(config), "--gpu", str(gpu), "--phase", "all"],
                                            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        status_path = root / "supervisor_status.json"
        while True:
            codes = {gpu: child.poll() for gpu, child in children.items()}
            status = {"pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat(),
                      "workers": {gpu: {"pid": children[gpu].pid, "exit_code": code} for gpu, code in codes.items()},
                      "state": "running" if any(c is None for c in codes.values()) else
                               ("complete" if all(c == 0 for c in codes.values()) else "failed")}
            write(status_path, status)
            with (root / "report.lock").open("a") as report_lock:
                fcntl.flock(report_lock, fcntl.LOCK_EX)
                result = subprocess.run([sys.executable, str(ROOT / "scripts/build_report.py"),
                                         "--matrix", str(config)], cwd=ROOT)
            if status["state"] != "running":
                if result.returncode:
                    status["state"] = "failed"; status["report_error"] = result.returncode; write(status_path, status)
                elif status["state"] == "complete":
                    evidence = json.loads((root / "reports/FINAL_RESULTS.json").read_text())
                    if evidence["state"] != "complete":
                        status["state"] = "failed"
                        status["report_error"] = "Required artifacts are missing or not verified"
                        write(status_path, status)
                break
            time.sleep(30)
        for stream in streams:
            stream.close()
        raise SystemExit(0 if status["state"] == "complete" else 1)


if __name__ == "__main__":
    main()
