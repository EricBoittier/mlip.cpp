#!/bin/bash -l
#SBATCH --job-name=petlr-cuda-fm
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: is the CUDA-only seeded lr_species excess nvcc's -use_fast_math (ggml-cuda's default: __expf / __logf /
# approximate division and sqrt, ftz)? Build a second CUDA tree from a copy of the patched ggml without -use_fast_math
# (build_nofm) and evaluate the species models (tiny + seeded) and the readout-pol trunk (ion-F) with both builds.
# Evaluation only; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
export CUDAToolkit_ROOT=$(dirname $(dirname $(which nvcc)))
W=/scratch/boittier/pet_lr_port/cuda; D=$W/deps; P=/scratch/boittier/pet_lr_port; F=$P/frames
if [ ! -d $D/ggml_nofm ]; then cp -a $D/ggml $D/ggml_nofm; sed -i 's/set(CUDA_FLAGS -use_fast_math -extended-lambda)/set(CUDA_FLAGS -extended-lambda)/' $D/ggml_nofm/src/ggml-cuda/CMakeLists.txt; fi
grep -n "set(CUDA_FLAGS" $D/ggml_nofm/src/ggml-cuda/CMakeLists.txt
cmake -S $W/src -B $W/build_nofm -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCUDAToolkit_ROOT=$CUDAToolkit_ROOT \
  -DMLIPCPP_BUILD_TESTS=ON -DCPM_ggml_SOURCE=$D/ggml_nofm -DCPM_fmt_SOURCE=$D/fmt -DCPM_nlohmann_json_SOURCE=$D/nlohmann_json \
  -DCPM_vesin_SOURCE=$D/vesin -DCPM_gpulite_SOURCE=$D/gpulite -DCPM_Catch2_SOURCE=$D/Catch2 > $W/cmake_nofm.log 2>&1 || { tail -30 $W/cmake_nofm.log; exit 1; }
cmake --build $W/build_nofm -j 16 --target mlipcpp-eval > $W/build_nofm.log 2>&1; echo "build rc=$?"; grep -m5 "error" $W/build_nofm.log
O=$W/out/fm; mkdir -p $O
for b in build_lr build_nofm; do
  for s in ion cluster; do
    for g in $W/src/tests/resources/petlr_tiny_species.gguf $W/gguf/seeded_lr_species.gguf; do n=$(basename $g .gguf)
      $W/$b/bin/mlipcpp-eval $g $F/set_$s.extxyz $O/${n}_${s}_$b.json --backend cuda > $O/${n}_${s}_$b.log 2>&1; echo "$b $n $s rc=$?"
    done
  done
  $W/$b/bin/mlipcpp-eval $W/gguf/readout-pol_lr_nobase.gguf $F/set_ion.extxyz $O/readout-pol_lr_ion_$b.json --backend cuda > $O/rpol_$b.log 2>&1
  $W/$b/bin/mlipcpp-eval $W/gguf/readout-pol_direct_nobase.gguf $F/set_ion.extxyz $O/readout-pol_trunk_ion_$b.json --backend cuda > $O/rpolt_$b.log 2>&1
  echo "$b readout-pol rc=$?"
done
