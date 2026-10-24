#!/usr/bin/env bash
# Test metrics of the repeated runs on the full LOREM test file (13,743 structures): for each run,
# the best-validation checkpoint (model.gguf.best.gguf, LOREM's R2(E)+R2(F) criterion) and the
# final weights (model.gguf) -> metatrain checkpoint -> mtt export -> the study's evaluate.py;
# then analyze_chi.py (charge identifiability) on the best checkpoints, and a summary over seeds.
#   NAMES="chi-s0 chi-s1 ..." sbatch repeats_eval.sbatch     (default: every run under rep/)
#SBATCH --job-name=plato-rep-eval
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato/logs/%x-%j.out
set -uo pipefail
D=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato
STUDY=/work/cosmo/boittier/lyra/plato/study
source /work/cosmo/boittier/lyra/plato/venv/bin/activate
export PYTHONPATH=$D/metatrain-chi/src:$D/pylib
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}
E=$D/eval-full; mkdir -p $E/runs/rep $E/runs/options_cheap; cd $E
ln -sfn $D/data-full data
ln -sfn $STUDY/runs/options_cheap/plato-D2-att runs/options_cheap/plato-D2-att
NAMES=${NAMES:-$(ls $D/rep)}
for n in $NAMES; do
  for which in best final; do
    src=$D/rep/$n/model.gguf; [ $which = best ] && src=$D/rep/$n/model.gguf.best.gguf
    m=runs/rep/$n-$which; mkdir -p $m
    [ -f $src ] || { echo "missing $src"; continue; }
    python $D/mlip.cpp/scripts/plato/gguf_to_plato_ckpt.py $src -o $m/model.ckpt 2>&1 | grep -v Warn
    mtt export $m/model.ckpt -o $m/model.pt > $m/export.log 2>&1 || tail -3 $m/export.log
    python -W ignore $STUDY/evaluate.py $m > $m/eval.log 2>&1 || tail -3 $m/eval.log
    echo "$n $which: $(python -c "import json; d=json.load(open('$m/metrics.json')); print({k: round(d[k], 4) for k in ('e_mae_mev_atom', 'f_mae_mev_a', 'f_rmse_mev_a', 'cc_f_mae_mev_a', 'far_e_mae_mev')})")"
  done
done
chi_names=$(for n in $NAMES; do [[ $n == id-* ]] || echo $n-best; done)
python -W ignore $STUDY/analyze_chi.py rep $chi_names > runs/rep/identifiability.log 2>&1 || tail -5 runs/rep/identifiability.log
python - $D/rep $E/runs/rep <<'EOF'
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
rep, ev = Path(sys.argv[1]), Path(sys.argv[2])
K = ("e_mae_mev_atom", "f_mae_mev_a", "f_rmse_mev_a", "cc_f_mae_mev_a", "far_e_mae_mev")
rows = defaultdict(list)
for m in sorted(ev.glob("*-*")):
    if not (m / "metrics.json").exists():
        continue
    name, which = m.name.rsplit("-", 1)
    model, seed = name.rsplit("-s", 1)
    d = json.load(open(m / "metrics.json"))
    ep = None
    if which == "best":
        log = [json.loads(l) for l in open(rep / name / "log.jsonl") if '"best": true' in l]
        ep = log[-1]["epoch"] if log else None
    rows[(model, which)].append([d[k] for k in K] + [ep])
out = {}
print("model     which  n  " + "  ".join(K) + "  (mean +- std over seeds; best epochs)")
for (model, which), r in sorted(rows.items()):
    a = np.array([x[:-1] for x in r], float)
    out[f"{model}/{which}"] = {"n": len(r), "mean": dict(zip(K, a.mean(0).tolist())), "std": dict(zip(K, a.std(0, ddof=1).tolist() if len(r) > 1 else [0] * len(K))),
                               "seeds": [dict(zip(K + ("best_epoch",), x)) for x in r]}
    print(f"{model:9s} {which:5s} {len(r)}  " + "  ".join(f"{m:.3f}+-{s:.3f}" for m, s in zip(a.mean(0), a.std(0, ddof=1) if len(r) > 1 else [0] * len(K))),
          [x[-1] for x in r])
json.dump(out, open(ev / "summary.json", "w"), indent=1)
EOF
grep -v "^\s*$" runs/rep/identifiability.log | cut -c1-600 | tail -12
