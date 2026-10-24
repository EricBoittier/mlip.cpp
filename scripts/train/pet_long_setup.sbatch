#!/usr/bin/env bash
# Plain PET (the study's pet-s) on bio_dimers in mlipcpp-train, with the protocol of the long
# Plato-PET-LR runs (scripts/plato/longrun.sbatch). Setup steps:
#   copy   snapshot of dev-plato's ggml source and CPM dependencies (read-only there)
#   data   train/ val/ test/ memmap splits of dev-plato/data-full (e_res, f)
#   init   fresh seed-0 init with pet-s's architecture (make_init.py) and its GGUF (--nc-forces:
#          mlipcpp-train needs the direct head; its loss weight is 0 in the runs)
#   build  mlipcpp-train, mlipcpp-train-check (CUDA, NCCL)
#   check  conservative loss and gradients of one batch against metatrain's PET (float64)
#   smoke  3 epochs of the long-run settings, and the same 3 epochs paused after 1 and resumed
#   STEPS="copy data init build check smoke" sbatch pet_long_setup.sbatch
#SBATCH --job-name=pet-long-setup
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-pet/logs/%x-%j.out
set -uo pipefail
P=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-pet
DP=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato
STUDY=/work/cosmo/boittier/lyra/plato/study
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg
module purge; module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9 nccl
source /work/cosmo/boittier/lyra/plato/venv/bin/activate
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}
S=$P/mlip.cpp/scripts
B=$P/build-cuda
STEPS=${STEPS:-copy data init build check smoke}
has() { [[ " $STEPS " == *" $1 "* ]]; }
echo "node $(hostname) gpu $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) job $SLURM_JOB_ID steps: $STEPS"
mkdir -p $P/models $P/runs

if has copy; then
  [ -d $P/ggml-src ] || cp -r $DP/ggml-src $P/ggml-src
  [ -d $P/deps ] || cp -r $DP/deps $P/deps
  du -sh $P/ggml-src $P/deps
fi

if has data; then
  for s in train val test; do
    rm -rf $P/data/$s
    python $S/train/xyz_to_memmap.py $DP/data-full/$s.xyz $P/data/$s --energy-key e_res --forces-key f
  done
fi

if has init; then
  # pet-s's options with the memmap splits; targets energy + the direct forces (the head
  # mlipcpp-train needs); the loss section (training only) dropped
  python - $STUDY/options/pet-s.yaml $P/models/pet-s-init.yaml <<'EOF'
import sys, yaml
o = yaml.safe_load(open(sys.argv[1]))
T = {"energy": {"quantity": "energy", "key": "energy", "unit": "eV", "forces": False, "stress": False, "virial": False},
     "non_conservative_force": {"quantity": "force", "key": "non_conservative_force", "unit": "eV/A",
                                "sample_kind": "atom", "type": {"cartesian": {"rank": 1}}}}
for k, d in (("training_set", "train/"), ("validation_set", "val/"), ("test_set", "test/")):
    o[k] = {"systems": {"read_from": d, "length_unit": "angstrom"}, "targets": T}
o["architecture"]["training"].pop("loss", None)
assert o["seed"] == 0
yaml.safe_dump(o, open(sys.argv[2], "w"), sort_keys=False)
print(open(sys.argv[2]).read())
EOF
  (cd $P/data && python -W ignore $S/train/make_init.py $P/models/pet-s-init.yaml -o $P/models/init.pt 2>&1 | tail -3)
  (cd $S/export_pytorch && python -W ignore export_pet_gguf.py --model $P/models/init.pt --nc-forces -o $P/models/init.gguf 2>&1 | tail -8)
fi

if has build; then
  O=$P/runs/build-$SLURM_JOB_ID; mkdir -p $O
  cmake -S $P/mlip.cpp -B $B -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=100;120" -DGGML_NATIVE=OFF \
    -DMLIPCPP_GGML_SOURCE_DIR=$P/ggml-src -DCPM_vesin_SOURCE=$P/deps/vesin-src -DCPM_fmt_SOURCE=$P/deps/fmt-src \
    -DCPM_nlohmann_json_SOURCE=$P/deps/nlohmann_json-src -DCPM_Catch2_SOURCE=$P/deps/catch2-src -DCPM_gpulite_SOURCE=$P/deps/gpulite-src \
    > $O/cmake.log 2>&1 || { tail -30 $O/cmake.log; exit 1; }
  grep -i nccl $O/cmake.log | head -3
  cmake --build $B -j 16 --target mlipcpp-train mlipcpp-train-check > $O/build.log 2>&1 || { grep -B2 -A8 -E " error|Error " $O/build.log | head -80; exit 1; }
  echo "built"
fi
T=$B/bin/mlipcpp-train
export GGML_CUDA_NO_TF32=1

if has check; then
  O=$P/runs/check-$SLURM_JOB_ID; mkdir -p $O
  $B/bin/mlipcpp-train-check --data $P/data --init $P/models/init.gguf --out $O --n 16 --gpus 1 \
    --conservative 1 --energy-weight 1 --force-weight 1 --nc-force-weight 0 2>&1 | grep -v "^ggml_cuda\|Device \|^Backend" | tail -15
  python -W ignore $S/train/check_gradients.py --init $P/models/init.pt --data $P/data --dump $O 2>&1 | tail -15
fi

# the long-run settings (pet_long.sbatch)
ARGS="--data $P/data --gpus 1 --batch-size 32 --conservative --energy-weight 1 --force-weight 1 --nc-force-weight 0
  --optimizer adam --lr 1e-4 --schedule linear --decay-after 10 --min-lr 1e-6 --grad-clip 1e30 --seed 0
  --threads 12 --cache-dir $P/nl-cache --log-every 100"
if has smoke; then
  O=$P/runs/smoke-$SLURM_JOB_ID; mkdir -p $O/a $O/b
  $T $ARGS --init $P/models/init.gguf --epochs ${SMOKE_EPOCHS:-3} --save-every 1 --state-every 1 -o $O/a/model.gguf --log $O/a/log.jsonl 2>&1 \
    | grep -v "^ggml_cuda\|Device \|^Backend" > $O/a/train.out
  grep -E "^epoch .* done|^test|^setup|^data|^model|^targets|step 400/" $O/a/train.out
  cat $O/a/log.jsonl
  # paused after epoch 1, then resumed: epochs 2, 3 as in a/
  $T $ARGS --init $P/models/init.gguf --epochs ${SMOKE_EPOCHS:-3} --max-hours 1e-6 -o $O/b/model.gguf --log $O/b/log.jsonl 2>&1 \
    | grep -v "^ggml_cuda\|Device \|^Backend" > $O/b/train1.out
  grep -E "^paused|^epoch .* done" $O/b/train1.out
  $T $ARGS --init $O/b/model.gguf --resume $O/b/model.gguf.state --epochs ${SMOKE_EPOCHS:-3} -o $O/b/model.gguf --log $O/b/log.jsonl 2>&1 \
    | grep -v "^ggml_cuda\|Device \|^Backend" > $O/b/train2.out
  grep -E "^resumed|^epoch .* done|^test" $O/b/train2.out
  cat $O/b/log.jsonl
  ls $O/a $O/b
fi
