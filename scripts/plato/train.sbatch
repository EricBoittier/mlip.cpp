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
module purge; module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9 nccl
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
  for r in grefE gref; do echo "== gradient check $r (CUDA, fp32 GEMMs=$GGML_CUDA_NO_TF32)"; $T --init $D/models/D2-att.gguf --train $D/data/train.bin --check $D/data/$r --backend cuda --fd-eps $EPS 2>&1 | grep -E "loss|gradient|largest"; done
  $D/build-cuda/bin/mlipcpp-plato check $D/models/D2-att.gguf $D/ref --backend cuda 2>&1 | grep -E "^cc0  |x8     full|open   full"
fi
if has speed; then
  for v in ${SPEED_VARIANTS:-none}; do
    echo "== speed: ${SPEED_STEPS:-200} steps, batch $BS, $v"
    env $([[ $v != none ]] && echo $v=1) $T --init $D/models/D2-att.gguf --train $D/data/train.bin --val $D/data/val.bin --epochs ${SPEED_EPOCHS:-1} --max-steps ${SPEED_STEPS:-200} --batch-size $BS --backend cuda --fd-eps $EPS --no-val --gpus ${GPUS:-1} ${SPEED_ARGS:-} -o $OUT/speed.gguf 2>&1 | grep -v "^ggml_cuda\|Device \|warmup"
  done
fi
if has chicheck; then
  echo "== electronegativity model: gradient check, parity, speed"
  $T --init $D/models/chi-test.gguf --train $D/data/train.bin --check $D/data/grefC --backend cuda --fd-eps $EPS 2>&1 | grep -E "loss|gradient|largest"
  $D/build-cuda/bin/mlipcpp-plato check $D/models/chi-test.gguf $D/ref-chi --backend cuda 2>&1 | grep -E "full|D2|d2_"
  $T --init $D/models/init-chi.gguf --train $D/data/train.bin --val $D/data/val.bin --epochs 1 --max-steps 200 --batch-size 32 --backend cuda --fd-eps $EPS -o $OUT/chi-speed.gguf 2>&1 | grep -E "^epoch"
  $T --init $D/models/init-nochi.gguf --train $D/data/train.bin --val $D/data/val.bin --epochs 1 --max-steps 200 --batch-size 32 --backend cuda --fd-eps $EPS -o $OUT/nochi-speed.gguf 2>&1 | grep -E "^epoch"
fi
if has prof; then
  rm -f $OUT/prof.*
  nsys profile --cuda-graph-trace=node -t cuda,nvtx -o $OUT/prof $T --init $D/models/D2-att.gguf --train $D/data/train.bin --val $D/data/val.bin --epochs 1 --max-steps ${SPEED_STEPS:-60} --batch-size $BS --backend cuda --fd-eps $EPS --no-val -o $OUT/prof.gguf > $OUT/prof.log 2>&1
  grep "^epoch" $OUT/prof.log
  nsys stats --report cuda_gpu_kern_sum --format csv -o $OUT/kern $OUT/prof.nsys-rep > /dev/null 2>&1
  python3 - $OUT/kern_cuda_gpu_kern_sum.csv <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
tot = sum(float(r["Total Time (ns)"]) for r in rows)
print(f"GPU kernel time total {tot/1e6:.1f} ms, {sum(int(r['Instances']) for r in rows)} launches")
for r in sorted(rows, key=lambda r: -float(r["Total Time (ns)"]))[:25]:
    print(f"{float(r['Total Time (ns)'])/1e6:9.1f} ms {int(r['Instances']):8d}  {r['Name'][:110]}")
PY
  nsys stats --report cuda_api_sum --format csv -o $OUT/api $OUT/prof.nsys-rep > /dev/null 2>&1
  head -8 $OUT/api_cuda_api_sum.csv | cut -c1-160
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
