#!/usr/bin/env bash
# Plato-PET-LR training in mlip.cpp (mlipcpp-train-plato) vs metatrain: restart of the
# 40-epoch plato-D2-att weights on bio_dimers. STEPS: build check speed train mtt rt
#SBATCH --job-name=plato-train
#SBATCH --account=cosmo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato/logs/%x-%j.out
set -euo pipefail
D=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato
C=/work/cosmo/boittier/cache
export CCACHE_DIR=$C/ccache CPM_SOURCE_CACHE=$C/cpm XDG_CACHE_HOME=$C/xdg
module purge; module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9
echo "node $(hostname) gpu $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
B=$D/build-cuda
STEPS=${STEPS:-build check speed}
OUT=$D/runs/${SLURM_JOB_ID}; mkdir -p $OUT
has() { [[ " $STEPS " == *" $1 "* ]]; }
T=$B/bin/mlipcpp-train-plato
EPOCHS=${EPOCHS:-3}; LR=${LR:-1e-4}; BS=${BS:-16}; EPS=${EPS:-5e-3}
export GGML_CUDA_NO_TF32=${NO_TF32:-1}
if has build; then
  cmake -S $D/mlip.cpp -B $B > $OUT/cmake.log 2>&1 || { tail -30 $OUT/cmake.log; exit 1; }
  cmake --build $B -j 16 --target mlipcpp-plato mlipcpp-train-plato > $OUT/build.log 2>&1 || { grep -B2 -A8 -E " error|Error " $OUT/build.log | head -80; exit 1; }
  echo built
fi
if has check; then
  for r in grefE gref; do echo "== gradient check $r (CUDA, fp32 GEMMs=$GGML_CUDA_NO_TF32)"; $T --init $D/models/D2-att.gguf --train $D/data/train.bin --check $D/data/$r --backend cuda --fd-eps $EPS 2>&1 | grep -v "^ggml_cuda\|Device 0"; done
fi
if has speed; then
  echo "== speed: 200 steps"
  $T --init $D/models/D2-att.gguf --train $D/data/train.bin --val $D/data/val.bin --epochs 1 --max-steps ${SPEED_STEPS:-200} --backend cuda --fd-eps $EPS -o $OUT/speed.gguf 2>&1 | grep -v "^ggml_cuda\|Device 0\|warmup"
fi
if has train; then
  echo "== mlipcpp-train-plato $EPOCHS epochs, lr $LR, batch $BS"
  /usr/bin/time -v $T --init $D/models/D2-att.gguf --train $D/data/train.bin --val $D/data/val.bin --epochs $EPOCHS --batch-size $BS --lr $LR --backend cuda --fd-eps $EPS --seed 0 -o $OUT/trained.gguf 2>&1 | grep -v "^ggml_cuda\|Device 0\|warmup" | grep -v "^\s*\(Page\|Volun\|Invol\|Swaps\|File\|Socket\|Signals\|Average\|Minor\|Major\|Exit\|Percent\|System time\|User time\)"
fi
if has mtt; then
  source /work/cosmo/boittier/lyra/plato/venv/bin/activate
  export PYTHONPATH=/work/cosmo/boittier/lyra/plato/metatrain-cheap/src
  export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}
  M=$OUT/mtt; mkdir -p $M; cd $M
  python - <<PY
import yaml
o = yaml.safe_load(open("$D/data/options_restart.yaml"))
t = o["architecture"]["training"]; t["num_epochs"] = $EPOCHS; t["learning_rate"] = $LR; t["batch_size"] = $BS
data = "/work/cosmo/boittier/lyra/plato/study/data"
for k, f in (("training_set", "train"), ("validation_set", "val"), ("test_set", "test")):
    for v in (o[k] if isinstance(o[k], list) else [o[k]]):
        v["systems"]["read_from"] = f"{data}/{f}.xyz"
        for tg in v["targets"].values():
            tg["read_from"] = tg["forces"]["read_from"] = tg["stress"]["read_from"] = f"{data}/{f}.xyz"
yaml.safe_dump(o, open("options.yaml", "w"))
PY
  t0=$(date +%s)
  mtt train options.yaml --restart $D/data/D2-att-rt.ckpt -o model.pt > train.log 2>&1 || { tail -20 train.log; }
  echo "mtt wall_s=$(( $(date +%s) - t0 ))"
  grep -E "Epoch|Training took|Starting training" train.log | cut -c1-420
  deactivate
fi
