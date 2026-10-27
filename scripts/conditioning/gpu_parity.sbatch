#!/bin/bash
#SBATCH --job-name=cond-gpu
# D holds mlip.cpp/ (the source), deps/ (ggml-src, vesin-src, fmt-src, nlohmann_json-src), cpm-cache/,
# catch2-src/, gpulite-src/, stubbin/ (a git stub), in/ (the GGUFs, frames.xyz of parity_pet.py, the
# make_cond_dataset.py data, plato_grad_reference.py outputs gc-plato.*, sched*.json); results in out/:
# compare with parity_pet.py --mlipcpp-json out/eval-*.json and check_gradients.py --dump out/gc-*
#SBATCH --account=lp161
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --output=/capstor/scratch/cscs/boittier/agents/conditioning/job-%j.out
# charge / spin conditioning and per-group lr on a GH200 (fp32, TF32 off)
set -x
D=${D:-/capstor/scratch/cscs/boittier/agents/conditioning}
srun --environment=qm9esp-cuda bash -c '
set -e
D=${D:-/capstor/scratch/cscs/boittier/agents/conditioning}
export PATH=$D/stubbin:/users/boittier/venvs/mptrj-py/bin:$PATH
export GGML_CUDA_NO_TF32=1 CUDA_VISIBLE_DEVICES=0
B=$D/build
rm -rf $B/CMakeCache.txt $B/_deps
if [ ! -x $B/bin/mlipcpp-eval ] || [ "${REBUILD:-1}" = 1 ]; then
  cmake -S $D/mlip.cpp -B $B -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DMLIPCPP_BUILD_TESTS=ON -DCPM_SOURCE_CACHE=$D/cpm-cache -DCPM_Catch2_SOURCE=$D/catch2-src \
    -DFETCHCONTENT_SOURCE_DIR_GPULITE=$D/gpulite-src -DMLIPCPP_GGML_SOURCE_DIR=$D/deps/ggml-src \
    -DCPM_vesin_SOURCE=$D/deps/vesin-src -DCPM_fmt_SOURCE=$D/deps/fmt-src -DCPM_nlohmann_json_SOURCE=$D/deps/nlohmann_json-src -DCPM_gpulite_SOURCE=$D/gpulite-src > $D/out/cfg.log 2>&1 || { tail -30 $D/out/cfg.log; exit 1; }
  cmake --build $B -j 64 > $D/out/build.log 2>&1 || { grep -B2 -A8 " error" $D/out/build.log | head -60; exit 1; }
fi
echo BUILD_OK
I=$D/in; O=$D/out
$B/tests/test_conditioning | tail -1
$B/tests/test_lr_schedule | tail -1
for m in pet-cond pet-cond-res pet-omol-s plato-cond pet1 plato1; do
  $B/bin/mlipcpp-eval $I/$m.gguf $I/frames.xyz $O/eval-$m-cuda.json --backend cuda 2>&1 | tail -1
  MLIPCPP_NO_GRAPH_CACHE=1 $B/bin/mlipcpp-eval $I/$m.gguf $I/frames.xyz $O/eval-$m-cuda-nocache.json --backend cuda 2>&1 | tail -1
done
MLIPCPP_HOST_INPUTS=1 $B/bin/mlipcpp-eval $I/pet-cond.gguf $I/frames.xyz $O/eval-pet-cond-cuda-hostinputs.json --backend cuda 2>&1 | tail -1
$B/bin/mlipcpp-train-check --data $I/data --init $I/pet-cond.gguf --out $O/gc-direct --n 16 --gpus 1 2>&1 | tail -1
$B/bin/mlipcpp-train-check --data $I/data --init $I/pet-cond.gguf --out $O/gc-cons --n 16 --gpus 1 --conservative 1 --energy-weight 1 --force-weight 1 --nc-force-weight 0.5 2>&1 | tail -1
for e in 5e-3 1e-3; do $B/bin/mlipcpp-train-plato --init $I/plato-cond.gguf --train $I/data/train.bin --check $I/gc-plato --backend cuda --fd-eps $e 2>&1 | grep -E "loss|gradient|system_cond"; done
$B/bin/mlipcpp-train --data $I/data --init $I/pet-cond.gguf -o $O/sched-gpu.gguf --epochs 4 --batch-size 8 --gpus 1 --lr-schedule $I/sched.json --log-every 1000 --seed 7 2>&1 | grep -E "group |epoch . lr|done:"
$B/bin/mlipcpp-train --data $I/data --init $I/pet-cond.gguf -o $O/sched-gpu2.gguf --epochs 2 --batch-size 4 --gpus 2 --lr-schedule $I/sched.json --log-every 1000 --seed 7 2>&1 | grep -E "epoch . lr|done:|error"
export CUDA_VISIBLE_DEVICES=0,1
$B/bin/mlipcpp-train --data $I/data --init $I/pet-cond.gguf -o $O/sched-gpu2.gguf --epochs 2 --batch-size 4 --gpus 2 --lr-schedule $I/sched.json --log-every 1000 --seed 7 2>&1 | grep -E "epoch . lr|done:|error|worker"
export CUDA_VISIBLE_DEVICES=0
$B/bin/mlipcpp-train-plato --init $I/plato-cond.gguf --train $I/data-pbc/train.bin --val $I/data-pbc/val.bin -o $O/plato-sched-gpu.gguf --epochs 3 --batch-size 8 --backend cuda --lr-schedule $I/sched-plato.json 2>&1 | grep -E "group |epoch . lr|^epoch"
echo ALL_DONE
'
