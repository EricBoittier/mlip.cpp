#!/bin/bash
#SBATCH --job-name=petlr-cuda-trunk
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: build feat/pet-lr-port with CUDA on kuma and evaluate the converted iris trunks (readout-pol,
# readout-4m) on the five frame families with --backend cuda, FP32 (no TF32). Evaluation only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cuda/12.5.1 cmake/3.27.9
W=/scratch/boittier/pet_lr_port/cuda; D=$W/deps
nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv
cmake -S $W/src -B $W/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DMLIPCPP_BUILD_TESTS=ON -DCPM_ggml_SOURCE=$D/ggml -DCPM_fmt_SOURCE=$D/fmt -DCPM_nlohmann_json_SOURCE=$D/nlohmann_json \
  -DCPM_vesin_SOURCE=$D/vesin -DCPM_gpulite_SOURCE=$D/gpulite -DCPM_Catch2_SOURCE=$D/Catch2 > $W/cmake.log 2>&1 || { tail -30 $W/cmake.log; exit 1; }
cmake --build $W/build -j 16 --target mlipcpp-eval mlipcpp-feats test_pet_lr_port > $W/build.log 2>&1 || { grep -m5 error $W/build.log; exit 1; }
F=/scratch/boittier/pet_lr_port/frames; mkdir -p $W/out
MLIPCPP_PETLR_GGUF=$W/gguf/readout-pol_direct_nobase.gguf $W/build/tests/test_pet_lr_port 2>&1 | tail -2
for ck in readout-pol:d readout-4m:d4; do g=${ck%%:*}; t=${ck#*:}; for s in mol ion cluster slab bulk; do
  $W/build/bin/mlipcpp-eval $W/gguf/${g}_direct_nobase.gguf $F/set_$s.extxyz $W/out/${s}_trunk_${t}_cuda.json --backend cuda > $W/out/${s}_${t}.log 2>&1
  $W/build/bin/mlipcpp-feats $W/gguf/${g}_direct_nobase.gguf $F/set_$s.extxyz $W/out/${s}_feats_${t}_cuda.bin 209 > /dev/null 2>&1
  echo "$g $s rc=$? $(tail -n1 $W/out/${s}_${t}.log)"; done; done
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
