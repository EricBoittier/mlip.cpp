#!/usr/bin/env python3
"""Paired per-frame test of an mlip.cpp result against his independent fp32 runs, all vs JAX fp64: medians, p90,
max, P(mlip.cpp worse than run X on a frame), and whether the worst frame is shared.

    python floor_paired.py ROOT CKPT FAMILY METRIC CPP_JSON CPP_FEATS RUN [RUN ...]   (METRIC: F or V)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from parity_table import cpp, ref  # noqa: E402


def per_frame(x, y, key):
    out = []
    for a, b in zip(x, y):
        if key == "F":
            out.append(np.abs(a["F"] - b["F"]).max() / np.abs(b["F"]).max())
        elif np.isfinite(a["V"]).all() and np.isfinite(b["V"]).all():
            sa, sb = (a["V"] + a["V"].T) / 2, (b["V"] + b["V"].T) / 2
            out.append(np.abs(sa - sb).max() / np.abs(sb).max())
    return np.array(out)


R, ck, s, key, js, fb = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
r64 = ref(R / f"ref/{ck}/{s}_cpu64.npz")
c = per_frame(cpp(R / js, R / fb, R / f"frames/set_{s}.extxyz"), r64, key)
print(f"== {ck} {s} {key} ({js}): n={len(c)}; mlip.cpp median {np.median(c):.3e} p90 {np.percentile(c, 90):.3e} max {c.max():.3e}")
for v in sys.argv[7:]:
    p = R / f"ref/{ck}/{s}_{v}.npz"
    if not p.exists():
        continue
    j = per_frame(ref(p), r64, key)
    print(f"  JAX {v:9s} median {np.median(j):.3e} p90 {np.percentile(j, 90):.3e} max {j.max():.3e} "
          f"P(cpp worse) {np.mean(c > j):.2f} same worst frame {int(np.argmax(c)) == int(np.argmax(j))}")
