#!/usr/bin/env python3
"""One line per out/pbc/parity_*.txt: max ratio to his max and the P range against his runs, per output."""
import glob, re
rows = []
for p in sorted(glob.glob("out/pbc/parity_*.txt")):
    txt = open(p).read()
    head = txt.splitlines()[0] if txt else p
    nruns = len(re.findall(r"^  JAX ", txt, re.M))
    cells = []
    for k in ("E", "F", "W", "q", "mu", "metallicity"):
        m = re.search(rf"^  {k}\s+his max .*?\(([\d.]+)x\); .*?((?:P [\d.]+ \(z [^)]*\);? ?)+)$", txt, re.M)
        if m:
            ps = [float(x) for x in re.findall(r"P ([\d.]+)", m.group(2))]
            cells.append(f"{k} {m.group(1)}x P {min(ps):.2f}-{max(ps):.2f}")
    rows.append(f"{p.split('parity_')[1][:-4]:34s} ({nruns} runs of his) " + "; ".join(cells))
print("\n".join(rows))
