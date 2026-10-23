#!/usr/bin/env bash
# Milestone evaluation of a long run: GGUF -> metatrain checkpoint (round trip), mtt export,
# the study's evaluate.py (test E / F / CC MAE, its timing), and analyze_chi.py (charge
# identifiability) on all milestones of the run.
#   NAME=chi EPOCHS="100 400" sbatch milestone.sbatch     (GGUF=path: one file instead)
#SBATCH --job-name=plato-milestone
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato/logs/%x-%j.out
set -uo pipefail
D=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato
STUDY=/work/cosmo/boittier/lyra/plato/study
source /work/cosmo/boittier/lyra/plato/venv/bin/activate
export PYTHONPATH=$D/metatrain-chi/src:$D/pylib
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}
E=$D/eval; mkdir -p $E/runs/mlipcpp $E/runs/options_cheap; cd $E
ln -sfn $STUDY/data data
ln -sfn $STUDY/runs/options_cheap/plato-D2-att runs/options_cheap/plato-D2-att
names=""
for ep in ${EPOCHS:-}; do
  tag=$(printf "%05d" $ep)
  src=${GGUF:-$D/long/$NAME/model.gguf.epoch$tag.gguf}
  m=runs/mlipcpp/$NAME-e$ep; mkdir -p $m
  [ -f $src ] || { echo "missing $src"; continue; }
  python $D/mlip.cpp/scripts/plato/gguf_to_plato_ckpt.py $src -o $m/model.ckpt 2>&1 | grep -v Warn
  mtt export $m/model.ckpt -o $m/model.pt > $m/export.log 2>&1 || tail -3 $m/export.log
  python -W ignore $STUDY/evaluate.py $m > $m/eval.log 2>&1 || tail -3 $m/eval.log
  echo "$NAME epoch $ep: $(python -c "import json; d=json.load(open('$m/metrics.json')); print({k: round(d[k], 4) for k in ('e_mae_mev_atom', 'f_mae_mev_a', 'cc_f_mae_mev_a', 'far_e_mae_mev')})")"
  names="$names $NAME-e$ep"
done
python -W ignore $STUDY/analyze_chi.py mlipcpp $names > runs/mlipcpp/identifiability-$NAME.log 2>&1 || tail -5 runs/mlipcpp/identifiability-$NAME.log
cp runs/mlipcpp/identifiability.json runs/mlipcpp/identifiability-$NAME.json 2>/dev/null
grep -v "^\s*$" runs/mlipcpp/identifiability-$NAME.log | cut -c1-900 | head -20
