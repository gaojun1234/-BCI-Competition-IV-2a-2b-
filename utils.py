"""Shared paths, hashes and deterministic project-local runtime settings."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for name, value in {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "MPLCONFIGDIR": str(ROOT / "runtime/matplotlib"),
    "_MNE_FAKE_HOME_DIR": str(ROOT / "runtime"),
}.items():
    os.environ.setdefault(name, value)
SEEDS = [11, 29, 47, 71, 101]
SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def recipes():
    return json.loads((ROOT / "config/subjects.json").read_text(encoding="utf8"))


def source_hash():
    h = hashlib.sha256()
    for p in sorted(ROOT.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    h.update((ROOT / "config/subjects.json").read_bytes())
    return h.hexdigest()
