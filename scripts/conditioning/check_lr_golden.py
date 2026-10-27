#!/usr/bin/env python3
"""--dump-lr-schedule of mlipcpp-train and mlipcpp-train-plato against metatrain's golden
per-group learning rates (tests/resources/lr_schedule_golden.json, from
src/metatrain/pet/modules/lr_schedule.py on EricBoittier/metatrain feat/pet-plato-lr).

    python check_lr_golden.py build/bin [--golden tests/resources/lr_schedule_golden.json]

Every case's spec goes to the binaries as --lr-schedule FILE --dump-lr-schedule T
--dump-steps-per-epoch S --epochs E; lr must agree to 1e-15 relative and frozen exactly.
(Name matching is tested in tests/test_lr_schedule.cpp.)
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bin")
    ap.add_argument("--golden", default=str(Path(__file__).resolve().parents[2] / "tests/resources/lr_schedule_golden.json"))
    args = ap.parse_args()
    golden = json.load(open(args.golden))
    ok = True
    for tool in ("mlipcpp-train", "mlipcpp-train-plato"):
        for case in golden["cases"]:
            S, E = case["steps_per_epoch"], case["num_epochs"]
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(case["spec"], f)
            out = subprocess.run([str(Path(args.bin) / tool), "--lr-schedule", f.name, "--dump-lr-schedule", str(S * E),
                                  "--dump-steps-per-epoch", str(S), "--epochs", str(E)],
                                 capture_output=True, text=True, check=True).stdout
            got = json.loads(out.strip().splitlines()[-1])
            worst, frozen_ok = 0.0, True
            for g, ref in case["lr"].items():
                for a, b in zip(got["lr"][g], ref):
                    worst = max(worst, abs(a - b) / max(abs(b), 1e-300))
                frozen_ok &= got["frozen"][g] == case["frozen"][g]
                frozen_ok &= len(got["lr"][g]) == len(ref)
            good = worst <= 1e-15 and frozen_ok
            ok &= good
            print(f"{tool:20s} {case['spec']['shape']:9s} {S} steps/epoch x {E}: groups {list(case['lr'])}, "
                  f"max relative lr difference {worst:.1e}, frozen {'match' if frozen_ok else 'DIFFER'} -> "
                  f"{'ok' if good else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
