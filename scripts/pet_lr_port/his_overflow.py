#!/usr/bin/env python3
"""Which frames does M. Langer's ASE Calculator (iris a2658e3) refuse with its max_neighbors overflow error? One line
per frame: index, 1 if it raised (the RuntimeError "an atom has more than max_neighbors"), else 0. Each frame is a
fresh system for his calculator (a new Calculator per frame), as mlipcpp-eval evaluates them.

    python his_overflow.py CKPT_DIR FRAMES.extxyz OUT.txt
"""
import sys

from ase.io import read


def main():
    ckpt, frames, out = sys.argv[1:4]
    import jax
    jax.config.update("jax_default_matmul_precision", "float32")
    from iris.pet.calculator import Calculator
    rows = []
    for f, at in enumerate(read(frames, ":")):
        at.info.setdefault("spin_multiplicity", 1)
        calc = Calculator.from_checkpoint(ckpt, discretize=True, info_keys={"spin_multiplicity": "spin_multiplicity"})
        at.calc = calc
        try:
            at.get_potential_energy()
            rows.append(f"{f} 0")
        except RuntimeError as e:
            if "max_neighbors" not in str(e):
                raise
            rows.append(f"{f} 1")
        print(rows[-1], flush=True)
    open(out, "w").write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
