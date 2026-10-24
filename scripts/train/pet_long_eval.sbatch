#!/usr/bin/env bash
# Milestones of the long plain-PET run (pet_long.sbatch): GGUF -> metatrain checkpoint
# (gguf_to_checkpoint.py, architecture from the run's init), mtt export, the study's
# evaluate.py on the full 13,743-structure test split (data-full; eval/runs/<tag>/metrics.json) and on
# the study's 3,000-structure test file that the Plato milestones used (eval/study/runs/<tag>/), the exported model
# copied to o3diag/models/long/<NAME>-e<epoch>.pt (best: <NAME>-best.pt); PROBE=1 also checks
# batched rotated periodic copies (probe_pbc_rotations.py) on the first milestone.
#   NAME=pet EPOCHS="100 1000 2000 4000 best" sbatch pet_long_eval.sbatch   (RUN=dir, COPY=0: no copy to o3diag)
#SBATCH --job-name=pet-long-eval
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-pet/logs/%x-%j.out
set -uo pipefail
P=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-pet
STUDY=/work/cosmo/boittier/lyra/plato/study
O3=/work/cosmo/boittier/lyra/plato/o3diag
source /work/cosmo/boittier/lyra/plato/venv/bin/activate
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}
NAME=${NAME:-pet}
RUN=${RUN:-$P/long/$NAME}
E=$P/eval; mkdir -p $E/runs $E/study/runs; cd $E
ln -sfn /work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato/data-full data   # test.xyz: 13,743 structures
first=""
for ep in ${EPOCHS:-100 1000 2000 4000 best}; do
  if [ $ep = best ]; then src=$RUN/model.gguf.best.gguf; tag=$NAME-best
  else src=$RUN/model.gguf.epoch$(printf "%05d" $ep).gguf; tag=$NAME-e$ep; fi
  [ -f $src ] || { echo "missing $src"; continue; }
  m=runs/$tag; mkdir -p $m
  python -W ignore $P/mlip.cpp/scripts/train/gguf_to_checkpoint.py --init $P/models/init.pt --gguf $src -o $m/model 2>&1 | grep -v Warn | tail -2
  mtt export $m/model.ckpt -o $m/model.pt > $m/export.log 2>&1 || tail -3 $m/export.log
  python -W ignore $STUDY/evaluate.py $m > $m/eval.log 2>&1 || tail -3 $m/eval.log
  echo "$tag full test: $(python -c "import json; d=json.load(open('$m/metrics.json')); print({k: round(d[k], 4) for k in ('e_mae_mev_atom', 'e_rmse_mev_atom', 'f_mae_mev_a', 'f_rmse_mev_a', 'cc_f_mae_mev_a')})")"
  ms=study/runs/$tag; mkdir -p $ms; cp $m/model.ckpt $m/model.pt $ms/
  (cd study && ln -sfn $STUDY/data data && python -W ignore $STUDY/evaluate.py runs/$tag > runs/$tag/eval.log 2>&1 || tail -3 runs/$tag/eval.log)
  echo "$tag study 3k test: $(python -c "import json; d=json.load(open('$ms/metrics.json')); print({k: round(d[k], 4) for k in ('e_mae_mev_atom', 'e_rmse_mev_atom', 'f_mae_mev_a', 'f_rmse_mev_a', 'cc_f_mae_mev_a')})")"
  [ "${COPY:-1}" = 1 ] && mkdir -p $O3/models/long && cp $m/model.pt $O3/models/long/$tag.pt
  first=${first:-$E/$m/model.pt}
done
if [ "${PROBE:-1}" = 1 ] && [ -n "$first" ]; then
  export LD_LIBRARY_PATH=/ssoft/spack/fendant/lyra-rtx6000/v2/spack/opt/spack/linux-rhel9-x86_64_v3/gcc-11.4.1/gcc-14.3.0-twlvodyosy4faolutbg4mtea23mkt4tp/lib64:$LD_LIBRARY_PATH
  deactivate
  $O3/venv/bin/python -W ignore $P/mlip.cpp/scripts/train/probe_pbc_rotations.py $first 45 2 189
fi
