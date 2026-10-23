#!/usr/bin/env bash
# Plato-PET-LR in mlip.cpp (feat/plato-d2): build, parity vs PyTorch, timing.
# STEPS: any of build check bench torch petexport pet (default: build check bench)
#SBATCH --job-name=plato-dev
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
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "node $(hostname) gpu $GPU driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
B=$D/build-${BUILD_TAG:-cuda}
STEPS=${STEPS:-build check bench}
OUT=$D/runs/${SLURM_JOB_ID}; mkdir -p $OUT
has() { [[ " $STEPS " == *" $1 "* ]]; }
if has build; then
  cmake -S $D/mlip.cpp -B $B -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=100;120" -DGGML_NATIVE=OFF \
    -DMLIPCPP_GGML_SOURCE_DIR=$D/ggml-src -DCPM_vesin_SOURCE=$D/deps/vesin-src -DCPM_fmt_SOURCE=$D/deps/fmt-src \
    -DCPM_nlohmann_json_SOURCE=$D/deps/nlohmann_json-src -DCPM_Catch2_SOURCE=$D/deps/catch2-src -DCPM_gpulite_SOURCE=$D/deps/gpulite-src \
    > $OUT/cmake.log 2>&1 || { tail -30 $OUT/cmake.log; exit 1; }
  cmake --build $B -j 16 --target mlipcpp-plato mlipcpp-predict > $OUT/build.log 2>&1 || { grep -B2 -A8 -E " error|Error " $OUT/build.log | head -80; exit 1; }
  echo "built"
fi
P=$B/bin/mlipcpp-plato
if has check; then
  for prec in fp32 tf32; do
    for be in cuda cpu; do
      [[ $be == cpu && $prec == tf32 ]] && continue
      echo "== check $be $prec"
      env $([[ $prec == fp32 ]] && echo GGML_CUDA_NO_TF32=1) MLIPCPP_NUM_THREADS=16 $P check $D/models/D2-att.gguf $D/ref --backend $be --json $OUT/check-$be-$prec.json 2>&1 | grep -v "^Backend"
    done
  done
  echo "== check cuda fp32, the training layout (FrameLinear folds in the graph)"
  GGML_CUDA_NO_TF32=1 MLIPCPP_PLATO_FOLD_IN_GRAPH=1 $P check $D/models/D2-att.gguf $D/ref --backend cuda --json $OUT/check-cuda-fold.json 2>&1 | grep -E "full|D2|d2_" 
fi
X=$D/ref/structures.xyz
if has bench; then
  for prec in fp32 tf32; do
    for m in D2-att D2-att-sr; do
      for K in ${KS:-1 2 8}; do
        [[ $m == D2-att && $K -gt ${MAXK_LR:-2} ]] && continue
        echo "== bench $m K=$K $prec: $(env $([[ $prec == fp32 ]] && echo GGML_CUDA_NO_TF32=1) $P bench $D/models/$m.gguf $X --tile $K --repeat ${REPEAT:-30} --backend cuda 2>&1 | grep '^{')"
      done
    done
  done
fi
if has parts; then
  for v in MLIPCPP_PLATO_NO_LES MLIPCPP_PLATO_NO_MESH "MLIPCPP_PLATO_NO_LES=1 MLIPCPP_PLATO_NO_MESH"; do
    for K in 1 2; do
      echo "== bench D2-att K=$K fp32 $v=1: $(env GGML_CUDA_NO_TF32=1 $v=1 $P bench $D/models/D2-att.gguf $X --tile $K --repeat ${REPEAT:-30} --backend cuda 2>&1 | grep '^{')"
    done
  done
fi
if has torch; then
  source /work/cosmo/boittier/lyra/plato/venv/bin/activate
  export PYTHONPATH=/work/cosmo/boittier/lyra/plato/metatrain-cheap/src
  export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}
  R=/work/cosmo/boittier/lyra/plato/study/runs
  python -W ignore $D/time_torch.py $R/options_cheap/plato-D2-att 1 2 2>&1 | grep '^{'
  python -W ignore $D/time_torch.py $R/options_cheap/plato-D2-att-sr 1 2 8 2>&1 | grep '^{'
  python -W ignore $D/time_torch.py $R/options/pet-s 1 2 8 2>&1 | grep '^{'
fi
if has petexport; then
  source /work/cosmo/boittier/lyra/plato/venv/bin/activate
  (cd $D/mlip.cpp/scripts/export_pytorch && python -W ignore export_pet_gguf.py --model /work/cosmo/boittier/lyra/plato/study/runs/options/pet-s/model.ckpt -o $D/models/pet-s.gguf 2>&1 | tail -15)
  deactivate || true
fi
if has pet; then
  # mlip.cpp's own PET of a similar size (pet-s of the study: d_pet 64, 4 heads), as GraphModel
  for prec in fp32 tf32; do
    for K in ${KS:-1 2 8}; do
      echo "== pet-s K=$K $prec: $(env $([[ $prec == fp32 ]] && echo GGML_CUDA_NO_TF32=1) MLIPCPP_RAGGED=${RAGGED:-1} $B/bin/mlipcpp-plato bench-any $D/models/pet-s.gguf $X --tile $K --repeat ${REPEAT:-30} 2>&1 | grep '^{')"
    done
  done
fi
