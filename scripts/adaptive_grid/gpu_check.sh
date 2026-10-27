#!/bin/bash
# GH200 checks of the "grid" adaptive cutoff (daint, qm9esp-cuda), e.g.
#   srun --jobid=<holder> --overlap --environment=qm9esp-cuda -n1 bash gpu_check.sh
# D holds mlip.cpp/, deps/, cpm-cache/, catch2-src/, gpulite-src/, stubbin/ (as scripts/conditioning/gpu_parity.sbatch)
# and in/: pet-omol-s.gguf, frames.xyz (parity_pet.py), bench.xyz (make_bench.py), data/ (make_cond_dataset.py),
# omol-data/ (make_omol_like.py), freeze-nc.json. Compare: parity_pet.py --mlipcpp-json, check_gradients.py --pad-exact 1.
set -e
D=${D:-/capstor/scratch/cscs/boittier/agents/adgrid}
export PATH=$D/stubbin:/users/boittier/venvs/mptrj-py/bin:$PATH
# the least used GPU of the node
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
export GGML_CUDA_NO_TF32=1 CUDA_VISIBLE_DEVICES=$G
echo "GPU $G"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
B=$D/build; I=$D/in; O=$D/out
if [ "${REBUILD:-1}" = 1 ]; then
  rm -rf $B/CMakeCache.txt $B/_deps
  cmake -S $D/mlip.cpp -B $B -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DMLIPCPP_BUILD_TESTS=ON -DCPM_SOURCE_CACHE=$D/cpm-cache -DCPM_Catch2_SOURCE=$D/catch2-src \
    -DFETCHCONTENT_SOURCE_DIR_GPULITE=$D/gpulite-src -DMLIPCPP_GGML_SOURCE_DIR=$D/deps/ggml-src \
    -DCPM_vesin_SOURCE=$D/deps/vesin-src -DCPM_fmt_SOURCE=$D/deps/fmt-src -DCPM_nlohmann_json_SOURCE=$D/deps/nlohmann_json-src -DCPM_gpulite_SOURCE=$D/gpulite-src > $O/cfg.log 2>&1 || { tail -30 $O/cfg.log; exit 1; }
  cmake --build $B -j 32 > $O/build.log 2>&1 || { grep -B2 -A8 " error" $O/build.log | head -60; exit 1; }
fi
echo BUILD_OK
M=$I/pet-omol-s.gguf
MLIPCPP_TEST_GRID_GGUF=$M $B/tests/test_adaptive_grid 2>&1 | tail -2 || true
for f in frames bench; do
  $B/bin/mlipcpp-eval $M $I/$f.xyz $O/$f-cuda.json --backend cuda 2>&1 | tail -1
  MLIPCPP_HOST_INPUTS=1 $B/bin/mlipcpp-eval $M $I/$f.xyz $O/$f-cuda-host.json --backend cuda 2>&1 | tail -1
  MLIPCPP_NO_GRAPH_CACHE=1 $B/bin/mlipcpp-eval $M $I/$f.xyz $O/$f-cuda-general.json --backend cuda 2>&1 | tail -1
done
$B/bin/mlipcpp-eval $M $I/bench.xyz $O/bench-cuda-t.json --backend cuda --repeat 10 2>&1 | tail -1
MLIPCPP_NO_GRAPH_CACHE=1 $B/bin/mlipcpp-eval $M $I/bench.xyz $O/bench-cuda-general-t.json --backend cuda --repeat 10 2>&1 | tail -1
export MLIPCPP_PAD_EXACT=1
$B/bin/mlipcpp-train-check --data $I/data --init $M --out $O/gc-direct --n 8 --gpus 1 2>&1 | tail -1
$B/bin/mlipcpp-train-check --data $I/data --init $M --out $O/gc-cons --n 8 --gpus 1 --conservative 1 --energy-weight 1 --force-weight 1 --nc-force-weight 0.5 2>&1 | tail -1
unset MLIPCPP_PAD_EXACT
for m in direct cons; do X=""; [ $m = cons ] && X="--conservative --nc-force-weight 0.5"
  $B/bin/mlipcpp-train --data $I/omol-data --init $M -o $O/tr-$m.gguf --epochs 3 --batch-size 8 --gpus 1 --lr-schedule $I/freeze-nc.json --log-every 1 --seed 1 --model-baselines $X > $O/tr-$m.log 2>&1 || tail -5 $O/tr-$m.log
  grep -E "done|test:" $O/tr-$m.log
done
echo ALL_DONE
