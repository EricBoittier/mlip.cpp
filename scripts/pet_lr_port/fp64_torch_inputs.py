#!/usr/bin/env python3
"""Bisection of an fp64 executor difference: the eager torch module (iris_torch IrisPETLRGraph / IrisPETLRGraphPBC /
IrisTrunk, float64) on EXACTLY the graph inputs mlip.cpp's fp64 driver built (MLIPCPP_FP64_DUMP=<file> on mlipcpp-eval
--fp64). Equal outputs (1e-15) put a difference vs his JAX on the host side (inputs); unequal ones on the graph side.

    python fp64_torch_inputs.py PARAMS.npz CKPT_DIR MODEL.gguf DUMP.jsonl [--pbc]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from check_grad64 import build  # noqa: E402
from gguf_diff import read  # noqa: E402
from iris_torch import IrisPETLRGraph, IrisPETLRGraphPBC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params"); ap.add_argument("ckpt"); ap.add_argument("gguf"); ap.add_argument("dump")
    ap.add_argument("--pbc", action="store_true")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    cfg, trunk, lr, scale = build(a.params, a.ckpt)
    _, kv, _ = read(a.gguf)
    gj = json.loads(kv["graph_pbc.json" if a.pbc else "graph.json"])
    mod = (IrisPETLRGraphPBC if a.pbc else IrisPETLRGraph)(trunk, lr, scale).double().eval() if lr is not None else trunk
    for line in open(a.dump):
        d = json.loads(line)
        args = []
        for spec in gj["inputs"]:
            v = d["inputs"][spec["name"]]
            arr = np.asarray(v["data"], np.float64).reshape(v["shape"])
            if spec["dtype"] == "i32":
                args.append(torch.as_tensor(arr.astype(np.int64)))
            elif spec["dtype"] == "bool":
                args.append(torch.as_tensor(arr != 0))
            else:
                args.append(torch.as_tensor(arr))
        with torch.no_grad():
            out = mod(*args)
        out = out.numpy().reshape(-1)
        cpp = np.asarray(d["output"]["data"], np.float64)
        C = np.asarray(d["output"]["shape"])[-1] if len(d["output"]["shape"]) > 1 else 1
        o, c = out.reshape(-1, C), cpp.reshape(-1, C)
        cols = " ".join(f"c{k} {np.abs(o[:, k] - c[:, k]).max() / max(np.abs(o[:, k]).max(), 1e-300):.1e}" for k in range(C))
        print(f"n={o.shape[0]} max|torch - fp64 executor| / max|torch| per column: {cols}")


if __name__ == "__main__":
    main()
