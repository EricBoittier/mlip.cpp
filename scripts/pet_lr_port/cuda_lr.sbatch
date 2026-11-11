#!/bin/bash -l
#SBATCH --job-name=petlr-cuda-lr
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: build ALL targets of feat/pet-lr-port with CUDA on kuma (incl. the CUDA-only src/md/md_engine.cpp),
# run the PET-LR tests on the GPU (tiny committed GGUFs, every reference type) and evaluate the full-PETLR GGUFs
# (trunk + LR, six omol-4m checkpoints) on ion / cluster / mol with --backend cuda, FP32 (no TF32). Evaluation only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
which nvcc || { echo "no nvcc after module load"; exit 1; }
export CUDAToolkit_ROOT=$(dirname $(dirname $(which nvcc)))
W=/scratch/boittier/pet_lr_port/cuda; D=$W/deps; P=/scratch/boittier/pet_lr_port
nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv
cmake -S $W/src -B $W/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCUDAToolkit_ROOT=$CUDAToolkit_ROOT \
  -DMLIPCPP_BUILD_TESTS=ON -DCPM_ggml_SOURCE=$D/ggml -DCPM_fmt_SOURCE=$D/fmt -DCPM_nlohmann_json_SOURCE=$D/nlohmann_json \
  -DCPM_vesin_SOURCE=$D/vesin -DCPM_gpulite_SOURCE=$D/gpulite -DCPM_Catch2_SOURCE=$D/Catch2 > $W/cmake.log 2>&1 || { tail -30 $W/cmake.log; exit 1; }
cmake --build $W/build -j 16 > $W/build_all.log 2>&1; echo "build all rc=$?"; grep -m10 -E "error|md_engine" $W/build_all.log
grep -c "md_engine.cpp.o" $W/build_all.log; find $W/build -name "md_engine.cpp.o" -newer $W/src/src/md/md_engine.cpp -printf "%p %TY-%Tm-%Td %TH:%TM\n"
$W/build/tests/test_pet_lr_interp 2>&1 | tail -2
$W/build/tests/test_pet_lr_port 2>&1 | tail -4      # tiny GGUFs (tests/resources), --backend default = CUDA
F=$P/frames; mkdir -p $W/out/lr
for g in ${CKS:-readout-pol control readout readout-1m readout-2m readout-4m}; do for s in ${FAMS:-ion cluster mol}; do
  fr=$F/set_$s.extxyz; [ $s = mol ] && fr=$F/set_mol_chunked.extxyz
  $W/build/bin/mlipcpp-eval $W/gguf/${g}_lr_nobase.gguf $fr $W/out/lr/${g}_${s}_cuda.json --backend cuda > $W/out/lr/${g}_${s}.log 2>&1
  echo "$g $s rc=$? $(tail -n1 $W/out/lr/${g}_${s}.log)"; done; done
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
