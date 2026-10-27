#!/usr/bin/env python3
"""The Plato-PET-LR gates of mlipcpp-train GGUFs (e.g. OUT.epoch*.gguf): one JSON line per
file with mean |g| and max |g| of every *_gate tensor.

    python gates.py OUT.epoch*.gguf > gates.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from gguf_to_checkpoint import read_gguf  # noqa: E402

for path in sys.argv[1:]:
    _, t = read_gguf(path)
    row = {"file": path}
    for name in sorted(t):
        if name.endswith("_gate") or name.endswith("gate_weight"):
            g = np.abs(t[name].astype(np.float64))
            row[name] = {"mean_abs": float(g.mean()), "max_abs": float(g.max())}
    print(json.dumps(row))
