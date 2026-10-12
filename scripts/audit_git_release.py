"""Read-only audit of the Git index before the user commits a release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "output/release_final_20261006/"
CODE_DIRS = ("pruning/", "scripts/", "tests/", "patch_slimming/psvit/",
             "patch_slimming/scripts/", "patch_slimming/tests/")
ROOT_FILES = {".gitignore", ".gitattributes", "README.md", "requirements.txt", "configs/final.yaml",
              "output/README.md", "patch_slimming/references/README.md"}
SECRET = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16}|hf_[a-zA-Z0-9]{30,}")
POINTER = re.compile(rb"version https://git-lfs.github.com/spec/v1\noid sha256:([a-f0-9]{64})\nsize ([0-9]+)\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def allowed_code(path):
    if any(part.startswith(".") for part in Path(path).parts):
        return path in ROOT_FILES
    return (path in ROOT_FILES or (path.startswith("docs/") and path.endswith(".md"))
            or (path.startswith(CODE_DIRS) and path.endswith(".py")))


def main():
    tracked = set(git("ls-files", "-z").decode().strip("\0").split("\0"))
    manifest = json.loads(git("show", f":{RELEASE}MANIFEST.json"))
    entries = {entry["path"]: entry for entry in manifest["files"]}
    expected = {RELEASE + name for name in entries} | {RELEASE + "MANIFEST.json", RELEASE + ".gitignore"}
    actual = {p for p in tracked if p.startswith(RELEASE)}
    if actual != expected:
        raise ValueError(f"Staged release differs from manifest: {actual ^ expected}")
    errors = []; lfs = []; largest = (0, "")
    for path in sorted(tracked):
        if path not in expected and not allowed_code(path):
            errors.append(f"Unexpected tracked source/artifact: {path}")
        blob_size = int(git("cat-file", "-s", f":{path}"))
        if blob_size > 10 * 1024**2:
            errors.append(f"Large ordinary Git blob (expected LFS): {path} ({blob_size} bytes)")
            continue
        content = git("show", f":{path}")
        if SECRET.search(content):
            errors.append(f"Potential credential/private key: {path}")
        if (blob_size, path) > largest:
            largest = (blob_size, path)
        if path.endswith((".onnx", ".pt")):
            pointer = POINTER.fullmatch(content)
            if not pointer:
                errors.append(f"Model is not an LFS pointer: {path}")
                continue
            entry = entries[path.removeprefix(RELEASE)]
            if pointer[1].decode() != entry["sha256"] or int(pointer[2]) != entry["bytes"]:
                errors.append(f"LFS pointer differs from manifest: {path}")
            lfs.append((pointer[1].decode(), int(pointer[2])))
            attributes = git("check-attr", "--cached", "filter", "--", path).decode().strip()
            if not attributes.endswith(": lfs"):
                errors.append(f"Missing staged LFS attribute: {path}")
        elif path.startswith(RELEASE) and path.removeprefix(RELEASE) in entries:
            entry = entries[path.removeprefix(RELEASE)]
            if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                errors.append(f"Staged evidence differs from manifest: {path}")
    if len(lfs) != 36:
        errors.append(f"Expected 36 LFS models (24 PT + 12 ONNX), got {len(lfs)}")
    # Check ignored runtime files without creating any dummy data.
    ignored = ["past/probe.md", "data/imagenet/probe.jpg", ".deps/probe.py", ".tools/probe",
               "output/final_20261006/tiny30/01_structural/checkpoint_last.pt",
               RELEASE + "tiny30/01_structural/checkpoint_last.pt",
               RELEASE + "tiny30/03_patch_slimming_paper/finetune.log"]
    for path in ignored:
        result = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT)
        if result.returncode != 0:
            errors.append(f"Runtime/extra artifact not ignored: {path}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps({"state": "ready_to_commit", "tracked_files": len(tracked),
                      "release_files_including_manifest_and_ignore": len(expected), "lfs_model_files": len(lfs),
                      "lfs_unique_objects": len(dict(lfs)), "lfs_unique_bytes": sum(dict(lfs).values()),
                      "largest_normal_git_blob": {"bytes": largest[0], "path": largest[1]},
                      "commit_created": False, "remote_modified": False}, indent=2))


if __name__ == "__main__":
    main()
