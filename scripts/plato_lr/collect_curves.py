#!/usr/bin/env python3
"""Training curves of mlipcpp-train runs as one JSON: per epoch the log.jsonl rows, per 200 steps the
train.log loss terms and the MLIPCPP_LOG_PARAMS gates (plato_les_gate, plato_mesh_gate).

    python collect_curves.py -o curves.json NAME=RUN_DIR ...
"""
import argparse
import json
import re
from pathlib import Path

STEP = re.compile(r"^epoch (\d+) step (\d+)/(\d+): loss (\S+) \(E (\S+), F (\S+), F_nc [^,]+(?:, q \S+: mean r ([0-9.eE+-]+))?")
GATE = re.compile(r"params at step (\d+): plato_les_gate (\S+) \(max \S+\) plato_mesh_gate (\S+) \(max (\S+)\)")


def run(d):
    d = Path(d)
    out = {"epochs": [], "steps": []}
    if (d / "log.jsonl").exists():
        out["epochs"] = [json.loads(x) for x in open(d / "log.jsonl") if x.strip()]
    if not (d / "train.log").exists():
        return out
    last = None
    for line in open(d / "train.log"):
        m = STEP.match(line)
        if m:
            ep, st, spe = int(m[1]), int(m[2]), int(m[3])
            last = {"epoch": ep + st / spe, "step": ep * spe + st, "loss": float(m[4]), "loss_e": float(m[5]),
                    "loss_f": float(m[6]), "q_r": float(m[7]) if m[7] else None}
            out["steps"].append(last)
            continue
        g = GATE.search(line)
        if g and last is not None:
            last.update(les_gate=float(g[2]), mesh_gate=float(g[3]), mesh_gate_max=float(g[4]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    res = {}
    for x in args.runs:
        name, d = x.split("=", 1)
        res[name] = run(d)
    json.dump(res, open(args.output, "w"))


if __name__ == "__main__":
    main()
