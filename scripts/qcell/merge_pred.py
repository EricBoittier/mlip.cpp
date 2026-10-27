"""Concatenate mlipcpp-eval JSONs (frame order kept): OUT.json IN0.json IN1.json ..."""
import json
import sys

out, parts = sys.argv[1], sys.argv[2:]
d = json.load(open(parts[0]))
for p in parts[1:]:
    d["frames"] += json.load(open(p))["frames"]
json.dump(d, open(out, "w"))
print(out, len(d["frames"]))
