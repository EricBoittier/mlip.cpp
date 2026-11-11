#!/usr/bin/env python3
"""His fp32 floor over independent JAX fp32 runs (CPU default / padding / single-thread variants, GPU MIG H100 with
two padding + autotune settings, GPU L40S), each against JAX fp64, next to mlip.cpp fp32 (CPU) against fp64.
Metrics as parity_table.py (E: |dE| / sum|E_i|; F: max|dF| / max|F|; h: node features), max over frames.

    python floor_spread.py ROOT CKPT TAG [families...]   (ROOT/ref/CKPT/{fam}_{variant}.npz, ROOT/out/{fam}_trunk_TAG.json)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from parity_table import cpp, m, ref  # noqa: E402

R, ck, tag = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
fams = sys.argv[4:] or ["mol", "cluster", "bulk"]
names = ["cpu32", "cpu32_v1", "cpu32_v2", "cpu32_v3", "gpu32", "gpu32_g2", "gpu32_g3"]
for s in fams:
    d = R / "ref" / ck
    r64 = ref(d / f"{s}_cpu64.npz")
    rows = {}
    for v in names:
        p = d / f"{s}_{v}.npz"
        if p.exists():
            rows[v] = m(ref(p), r64)
    c = m(cpp(R / f"out/{s}_trunk_{tag}.json", R / f"out/{s}_feats_{tag}.bin", R / f"frames/set_{s}.extxyz"), r64)
    print(f"== {ck} {s}: E / F / h / V vs JAX fp64")
    for v, x in rows.items():
        print(f"  JAX {v:10s} " + " ".join("     –   " if y is None else f"{y:9.2e}" for y in x))
    for i, lab in enumerate("EFhV"):
        vals = [x[i] for x in rows.values() if x[i] is not None]
        if vals and c[i] is not None:
            print(f"  {lab}: his floor over {len(vals)} runs [{min(vals):.2e}, {max(vals):.2e}]; mlip.cpp {c[i]:.2e} "
                  f"-> {'within' if c[i] <= max(max(vals), 1e-5) else 'ABOVE'} (ratio to his max {c[i] / max(vals):.2f})")
