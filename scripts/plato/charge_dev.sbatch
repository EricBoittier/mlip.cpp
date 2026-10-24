#!/usr/bin/env bash
# Molecule-charge loss: data with molecules, gradient checks, a short training run.
#   STEPS="build data check gradcheck smoke" sbatch charge_dev.sbatch
#SBATCH --job-name=plato-charge-dev
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato/logs/%x-%j.out
set -uo pipefail
D=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato
ARCHIVE=/work/cosmo/boittier/lyra/plato/lorem-tmlr-archive/datasets
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg
module purge; module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9 nccl
source /work/cosmo/boittier/lyra/plato/venv/bin/activate
S=$D/mlip.cpp/scripts/plato
STEPS=${STEPS:-build data check gradcheck smoke}
has() { [[ " $STEPS " == *" $1 "* ]]; }
B=$D/build-cuda
T=$B/bin/mlipcpp-train-plato
echo "node $(hostname) job $SLURM_JOB_ID steps: $STEPS"
export GGML_CUDA_NO_TF32=1

if has build; then
  cmake --build $B -j 16 --target mlipcpp-train-plato > $D/logs/charge-build-$SLURM_JOB_ID.log 2>&1 \
    || { grep -B2 -A8 -E " error|Error " $D/logs/charge-build-$SLURM_JOB_ID.log | head -60; exit 1; }
  echo built
fi

M=$D/data-mol
if has data; then
  mkdir -p $M
  python $S/plato_dataset.py $D/data-full/train.xyz $M/train.bin --energy e_res --forces f --molecules $ARCHIVE/bio_dimers_train.xyz
  python $S/plato_dataset.py $D/data-full/val.xyz $M/val.bin --energy e_res --forces f --molecules $ARCHIVE/bio_dimers_valid.xyz
  python $S/plato_dataset.py $D/data-full/test.xyz $M/test.bin --energy e_res --forces f --molecules $ARCHIVE/bio_dimers_test.xyz
fi

if has check; then # indexB against the bond graph: atoms [0, indexB) and [indexB, n) are the two molecules
  python - $ARCHIVE <<'EOF'
import sys
import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs
from scipy.sparse.csgraph import connected_components
for f in ("bio_dimers_train.xyz", "bio_dimers_valid.xyz", "bio_dimers_test.xyz"):
    fr = read(f"{sys.argv[1]}/{f}", ":")
    rng = np.random.default_rng(0)
    bad = n2 = 0
    for k in rng.choice(len(fr), 300, replace=False):
        a = fr[k].copy()
        nl = NeighborList([1.2 * c for c in natural_cutoffs(a)], skin=0.0, self_interaction=False, bothways=True)
        nl.update(a)
        nc, lab = connected_components(nl.get_connectivity_matrix(sparse=True), directed=False)
        s = int(a.info["indexB"])
        n2 += nc == 2
        if nc == 2 and not ((lab[:s] == lab[0]).all() and (lab[s:] != lab[0]).all()):
            bad += 1
    print(f, "300 checked:", n2, "with two bonded fragments,", bad, "where indexB disagrees")
EOF
fi

if has gradcheck; then
  for init in $D/models/rep/chi-0.gguf $D/long/chi/model.gguf; do
    echo "== $init"
    for w in "0 0 1" "1 1 0" "1 1 1"; do
      set -- $w
      for h in 1e-2 3e-3 1e-3; do
        $T --init $init --train $M/train.bin --backend cuda --gradcheck $h --batch-size 8 \
          --energy-weight $1 --force-weight $2 --charge-weight $3 2>&1 | grep -E "^gradcheck|error"
      done
    done
  done
fi

if has smoke; then
  O=$D/charge-smoke; rm -rf $O; mkdir -p $O
  $T --init $D/models/rep/chi-0.gguf --train $M/train.bin --val $M/val.bin --backend cuda --epochs 3 --batch-size 32 \
    --lr 1e-4 --schedule linear --decay-after 10 --min-lr 1e-6 --grad-clip 1e30 --seed 0 --charge-weight ${CW:-0.01} \
    -o $O/model.gguf --log $O/log.jsonl 2>&1 | grep -E "^epoch|^done|error"
fi
