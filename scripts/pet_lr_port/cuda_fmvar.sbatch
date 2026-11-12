#!/bin/bash -l
#SBATCH --job-name=petlr-cuda-fmvar
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: which fast-math kernel makes the CUDA species preset 5x his floor? VAR=unary (unary.cu without
# -use_fast_math, rest as default), binbcast (binbcast.cu without), log (default flags, op_log in double),
# exp (default flags, op_exp + op_sigmoid in double). Evaluation only; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
export CUDAToolkit_ROOT=$(dirname $(dirname $(which nvcc)))
W=/scratch/boittier/pet_lr_port/cuda; D=$W/deps; P=/scratch/boittier/pet_lr_port; F=$P/frames
G=$D/ggml_$VAR; rm -rf $G; cp -a $D/ggml $G; C=$G/src/ggml-cuda
case $VAR in
  unary|binbcast)
    sed -i 's/set(CUDA_FLAGS -use_fast_math -extended-lambda)/set(CUDA_FLAGS -extended-lambda)/' $C/CMakeLists.txt
    cat >> $C/CMakeLists.txt <<CM
foreach(_s IN LISTS GGML_SOURCES_CUDA)
  if(NOT _s MATCHES "/$VAR\\\\.cu\$")
    set_source_files_properties(\${_s} PROPERTIES COMPILE_OPTIONS "-use_fast_math")
  endif()
endforeach()
CM
    ;;
  log) sed -i 's/    return logf(x);/    return (float) log((double) x);/' $C/unary.cu ;;
  exp) sed -i -e 's/    return expf(x);/    return (float) exp((double) x);/' -e 's|    return 1.0f / (1.0f + expf(-x));|    return (float) (1.0 / (1.0 + exp(-(double) x)));|' $C/unary.cu ;;
esac
tail -6 $C/CMakeLists.txt; grep -n "GGML_SOURCES_CUDA" $C/CMakeLists.txt | head -3; diff <(cat $D/ggml/src/ggml-cuda/unary.cu) $C/unary.cu
B=$W/build_$VAR
cmake -S $W/src -B $B -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCUDAToolkit_ROOT=$CUDAToolkit_ROOT \
  -DMLIPCPP_BUILD_TESTS=ON -DCPM_ggml_SOURCE=$G -DCPM_fmt_SOURCE=$D/fmt -DCPM_nlohmann_json_SOURCE=$D/nlohmann_json \
  -DCPM_vesin_SOURCE=$D/vesin -DCPM_gpulite_SOURCE=$D/gpulite -DCPM_Catch2_SOURCE=$D/Catch2 > $W/cmake_$VAR.log 2>&1 || { tail -30 $W/cmake_$VAR.log; exit 1; }
cmake --build $B -j 16 --target mlipcpp-eval -- VERBOSE=1 > $W/build_$VAR.log 2>&1; echo "build rc=$?"; grep -m5 "error" $W/build_$VAR.log
grep -o "nvcc.*\(unary\|binbcast\|softmax\)\.cu\.o -c" $W/build_$VAR.log | grep -o "use_fast_math\|[a-z]*\.cu\.o" | paste -sd' '
O=$W/out/fm; mkdir -p $O
for s in ion cluster; do
  $B/bin/mlipcpp-eval $W/gguf/seeded_lr_species.gguf $F/set_$s.extxyz $O/seeded_lr_species_${s}_$VAR.json --backend cuda > $O/seeded_lr_species_${s}_$VAR.log 2>&1; echo "$VAR $s rc=$?"
done
