"""OMol25 isolated-atom references (HF facebook/OMol25 references/iso_atom_elem_refs.yaml, block
omol_elem_refs, Z -> {atom charge: eV}) -> baseline.json {composition: {Z: E(neutral atom)}} for
make_memmap.py (energy = E - sum of neutral-atom energies: a fixed, label-free reference).

    python -I omol_refs.py iso_atom_elem_refs.yaml OUT.json
"""
import json
import sys

lines = open(sys.argv[1]).read().splitlines()
i = lines.index("omol_elem_refs:") + 1
refs, z = {}, None
while i < len(lines) and (lines[i].startswith(" ") or not lines[i].strip()):
    s = lines[i].strip()
    if s.endswith(":") and lines[i].startswith("    ") and not lines[i].startswith("        "):
        z = int(s[:-1]); refs[z] = {}
    elif ":" in s and z is not None:
        q, v = s.split(":")
        refs[z][int(q)] = float(v)
    i += 1
comp = {str(z): v[0] for z, v in refs.items() if 0 in v}
json.dump({"composition": comp, "source": "OMol25 iso_atom_elem_refs.yaml omol_elem_refs, neutral atoms",
           "all_charges": {str(z): {str(q): e for q, e in v.items()} for z, v in refs.items()}}, open(sys.argv[2], "w"), indent=1)
print(len(comp), "elements; H", comp.get("1"), "C", comp.get("6"))
