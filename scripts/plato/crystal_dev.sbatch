#!/usr/bin/env bash
# Crystals (ai4sci-atomistic-materials: MPTrj + sAlex) for Plato-PET-LR in mlip.cpp, development:
#   build      mlipcpp-train-plato and mlipcpp-plato (CUDA)
#   regress    bio_dimers unchanged by the mixed-cell changes: inference parity of D2-att, and the
#              gradient check of the 4000-epoch chi model (same batch and loss as before)
#   data       training / validation JSONL -> .bin, baseline (composition, scale) from training
#   init       fresh inits (identity, D2-att) with the bio_dimers hypers
#   gradcheck  E / F / stress gradients on mixed-cell crystal batches
#   smoke      a short training run on the validation frames with stress
#   timing     300 steps on the full training set: time per step, graph builds
#   STEPS="build regress data init gradcheck smoke timing" sbatch [-p b200] crystal_dev.sbatch
# Crystals need the B200 (or a partition with >= 160 KB shared memory per block): the ragged
# attention backward holds a sequence of up to 1 + max neighbours (185 in the validation frames).
#SBATCH --job-name=plato-crystal-dev
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato/logs/%x-%j.out
set -uo pipefail
D=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato
H=/work/cosmo/boittier/lyra/ai4sci
X=$D/crystal
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg
module purge; module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9 nccl
source /work/cosmo/boittier/lyra/plato/venv/bin/activate
export PYTHONPATH=$D/metatrain-chi/src:$D/pylib
S=$D/mlip.cpp/scripts/plato
B=$D/build-cuda
T=$B/bin/mlipcpp-train-plato
STEPS=${STEPS:-build regress data init gradcheck smoke}
has() { [[ " $STEPS " == *" $1 "* ]]; }
mkdir -p $X
export GGML_CUDA_NO_TF32=1
echo "node $(hostname) job $SLURM_JOB_ID steps: $STEPS"

if has build; then
  cmake --build $B -j 16 --target mlipcpp-train-plato mlipcpp-plato > $D/logs/crystal-build-$SLURM_JOB_ID.log 2>&1 \
    || { grep -B2 -A8 -E " error|Error " $D/logs/crystal-build-$SLURM_JOB_ID.log | head -60; exit 1; }
  echo built
fi

if has regress; then
  echo "== inference parity, D2-att (bio_dimers)"
  $B/bin/mlipcpp-plato check $D/models/D2-att.gguf $D/ref --backend cuda 2>&1 | grep -E "^cc0 |^cc0_x8|full" | head -6
  echo "== gradcheck, 4000-epoch chi model (before: E 1.58e-06 F 0.0004315 Q 0.122, rel 6e-6..2e-4 at Q 1)"
  $T --init $D/long/chi/model.gguf --train $D/data-mol/train.bin --backend cuda --gradcheck 1e-3 --batch-size 8 \
    --energy-weight 1 --force-weight 1 --charge-weight 1 2>&1 | grep -E "^gradcheck|error"
fi

if has data; then
  python $S/plato_dataset.py $H/training/train.jsonl.gz $X/train.bin --fit-baseline $X/baseline.json
  python $S/plato_dataset.py $H/training/val.jsonl.gz $X/val.bin
  python - $X/baseline.json $H/training/val.jsonl.gz <<'EOF'
import gzip, json, sys
bl = json.load(open(sys.argv[1]))
have = set(int(z) for z in bl["composition"])
miss = {}
for l in gzip.open(sys.argv[2], "rt"):
    for z in set(json.loads(l)["atoms"]["numbers"]) - have:
        miss[z] = miss.get(z, 0) + 1
print("baseline:", len(have), "elements; energy scale", bl["energy_scale"], "residual MAE/atom", bl["residual_per_atom_mae_ev"])
print("validation elements missing from training:", miss)
EOF
fi

if has init; then
  for f in identity D2; do
    python -W ignore $S/make_crystal_init.py $D/data/D2-att-rt.ckpt $X/baseline.json $X/init-$f-0.ckpt --seed 0 --frames $f
    python -W ignore $S/export_plato_gguf.py $X/init-$f-0.ckpt --baseline $X/baseline.json --mesh-round fft -o $X/init-$f-0.gguf 2>&1 | tail -1
  done
fi

if has gradcheck; then
  for w in "1 1 0" "0 0 1" "1 1 1"; do
    set -- $w
    for h in 3e-3 1e-3; do
      $T --init $X/init-D2-0.gguf --train $X/val.bin --backend cuda --gradcheck $h --batch-size 8 \
        --energy-weight $1 --force-weight $2 --stress-weight $3 2>&1 | grep -E "^gradcheck|error|groups"
    done
  done
fi

if has smoke; then
  O=$X/smoke; rm -rf $O; mkdir -p $O
  $T --init $X/init-D2-0.gguf --train $X/val.bin --val $X/val.bin --backend cuda --epochs 2 --batch-size 32 \
    --lr 1e-4 --schedule linear --decay-after 10 --min-lr 1e-6 --grad-clip 1e30 --seed 0 --stress-weight 1 --graph-cache ${CACHE:-24} \
    -o $O/model.gguf --log $O/log.jsonl 2>&1 | grep -E "^epoch|groups|^done|error|structures|CUDA|current device"
fi

if has timing; then
  O=$X/timing; rm -rf $O; mkdir -p $O
  $T --init $X/init-D2-0.gguf --train $X/train.bin --val $X/val.bin --no-val --backend cuda --epochs 1 --batch-size 32 \
    --lr 1e-4 --schedule linear --decay-after 10 --min-lr 1e-6 --grad-clip 1e30 --seed 0 --stress-weight 1 --graph-cache ${CACHE:-24} \
    --max-steps 300 -o $O/model.gguf --log $O/log.jsonl 2>&1 | grep -E "^epoch|groups|step 300|^done|error|structures|CUDA|current device"
fi
