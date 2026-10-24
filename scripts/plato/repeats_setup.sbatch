#!/usr/bin/env bash
# Setup of the repeated 4000-epoch runs (3 models x 3 seeds), LOREM's evaluation protocol:
#   data   the full validation file (3,204) and the full test file (13,743) of the LOREM archive,
#          with the study's per-element baseline (train.bin unchanged)
#   init   fresh inits per seed: D2-att with the frozen-QEq electronegativity term, D2-att
#          without, and the same model without frames (identity)
#   build  mlipcpp-train-plato and mlipcpp-plato (CUDA)
#   check  parity of the identity-frame model against PyTorch (random init, fp32)
#   smoke  3 epochs of the identity model with the full validation set and the best checkpoint
#   STEPS="data init build check smoke" SEEDS="0 1 2" sbatch repeats_setup.sbatch
#SBATCH --job-name=plato-rep-setup
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato/logs/%x-%j.out
set -uo pipefail
D=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato
STUDY=/work/cosmo/boittier/lyra/plato/study
ARCHIVE=/work/cosmo/boittier/lyra/plato/lorem-tmlr-archive/datasets
SRC=$STUDY/runs/options_cheap/plato-D2-att/model.ckpt
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg
module purge; module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9 nccl
source /work/cosmo/boittier/lyra/plato/venv/bin/activate
export PYTHONPATH=$D/metatrain-chi/src:$D/pylib
S=$D/mlip.cpp/scripts/plato
STEPS=${STEPS:-data init build check smoke}
SEEDS=${SEEDS:-0 1 2}
has() { [[ " $STEPS " == *" $1 "* ]]; }
echo "node $(hostname) job $SLURM_JOB_ID steps: $STEPS"

if has data; then
  F=$D/data-full; mkdir -p $F
  python - "$ARCHIVE" "$STUDY/data/baseline.json" "$F" <<'EOF'
import json, sys
from pathlib import Path
from ase.io import read, write
src, baseline, out = Path(sys.argv[1]), json.load(open(sys.argv[2])), Path(sys.argv[3])
for name, f in (("val", "bio_dimers_valid.xyz"), ("test", "bio_dimers_test.xyz")):
    clean = []
    for a in read(src / f, ":"):
        e = a.get_potential_energy() - sum(baseline[str(int(z))] for z in a.numbers)
        fr = a.get_forces().copy()
        a.calc = None
        a.info = {k: a.info[k] for k in ("label", "distance", "chargeA", "chargeB", "dimer_id") if k in a.info}
        a.info["e_res"] = e
        a.arrays["f"] = fr
        clean.append(a)
    write(out / f"{name}.xyz", clean)
    print(name, len(clean), "structures,", sum(len(a) for a in clean), "atoms")
EOF
  ln -sfn $STUDY/data/train.xyz $F/train.xyz
  cp $STUDY/data/baseline.json $F/
  cp $D/data/train.bin $F/train.bin
  python $S/plato_dataset.py $F/val.xyz $F/val.bin --energy e_res --forces f
  python $S/plato_dataset.py $F/test.xyz $F/test.bin --energy e_res --forces f
  ls -la $F
fi

M=$D/models/rep
if has init; then
  mkdir -p $M
  for s in $SEEDS; do
    python -W ignore $S/make_init.py $SRC $M/chi-$s.ckpt --seed $s --electronegativity linear --chi0-frozen --dchi-bound 1.0
    python -W ignore $S/make_init.py $SRC $M/nochi-$s.ckpt --seed $s
    python -W ignore $S/make_init.py $SRC $M/id-$s.ckpt --seed $s --frames identity
    for m in chi nochi id; do
      python -W ignore $S/export_plato_gguf.py $M/$m-$s.ckpt -o $M/$m-$s.gguf 2>&1 | tail -1
    done
  done
fi
if has init || has hypers; then
  python - $M <<'EOF'
import sys, torch
import metatomic.torch  # noqa: F401
M = sys.argv[1]
for n in ("chi-0", "nochi-0", "id-0"):
    h = torch.load(f"{M}/{n}.ckpt", weights_only=False, map_location="cpu")["model_data"]["model_hypers"]
    print(n, {k: h.get(k) for k in ("frames", "frame_mode", "rope", "electronegativity", "chi0_trainable", "dchi_bound")})
EOF
fi

B=$D/build-cuda
if has build; then
  cmake --build $B -j 16 --target mlipcpp-train-plato mlipcpp-plato > $D/logs/rep-build-$SLURM_JOB_ID.log 2>&1 \
    || { grep -B2 -A8 -E " error|Error " $D/logs/rep-build-$SLURM_JOB_ID.log | head -60; exit 1; }
  echo "built"
fi

if has check; then
  R=$D/ref-id; mkdir -p $R
  python -W ignore $S/plato_reference.py $D/models/rep/id-0.ckpt $STUDY/data/test.xyz $R 2>&1 | tail -2
  GGML_CUDA_NO_TF32=1 $B/bin/mlipcpp-plato check $D/models/rep/id-0.gguf $R --backend cuda 2>&1 | grep -v "^Backend\|^ggml_cuda\|Device "
fi

if has smoke; then
  O=$D/rep-smoke; rm -rf $O; mkdir -p $O
  GGML_CUDA_NO_TF32=1 $B/bin/mlipcpp-train-plato --init $D/models/rep/id-0.gguf --train $D/data-full/train.bin \
    --val $D/data-full/val.bin --backend cuda --epochs 3 --batch-size 32 --lr 1e-4 --schedule linear --decay-after 10 \
    --min-lr 1e-6 --grad-clip 1e30 --seed 0 --save-every 100 --state-every 10 -o $O/model.gguf --log $O/log.jsonl 2>&1 \
    | grep -E "^epoch|^done|^backend|structures|error"
  ls $O; cat $O/log.jsonl
fi
