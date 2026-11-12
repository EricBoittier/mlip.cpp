#!/bin/bash -l
#SBATCH --job-name=petlr-gemm
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: rebuild the CUDA tree with ggml-cuda-precise-mulmat.patch (F32 MUL_MAT accumulating in double when
# flagged; applied by hand to the pre-patched deps/ggml before this job), the GPU tests, then
#  (a) the CUDA charge-error candidate: the charge families (readout-pol, control, readout; bulk + slab, 60 frames)
#      with MLIPCPP_PRECISE_MATMUL_K=1036 (the transfer MLP's first LINEAR only) and MLIPCPP_PRECISE_MATMUL=1 (every
#      F32 MUL_MAT), A/B against the head config (out/pbc/cuda3 = cuda4_skipsum);
#  (b) the long-range transfer on 3-D cells (lrt_species_*; water and NaCl in water) -> out/branch_pbc_cuda;
#  (c) the conserving-charges presets (cc_*; mol_z18, water) -> out/cc_cuda.
# Evaluation only; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32 MLIPCPP_PRECISE_SIGMOID MLIPCPP_PRECISE_MATH_SKIP MLIPCPP_PRECISE_SUM MLIPCPP_PRECISE_MATMUL MLIPCPP_PRECISE_MATMUL_K
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
W=/scratch/boittier/pet_lr_port/cuda; P=/scratch/boittier/pet_lr_port; F=$P/frames; B=$W/build_lr/bin
cmake --build $W/build_lr -j 16 > $W/build_gemm.log 2>&1; echo "build rc=$?"; grep -m10 -E " error|error:" $W/build_gemm.log
$W/build_lr/tests/test_pet_lr_interp 2>&1 | grep -E "passed|failed"
$W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E -A12 "FAILED" | head -40
$W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E "passed|failed"
# (a)
for mode in k1036 all; do
  O=$W/out/pbc/gemm_$mode; mkdir -p $O
  for g in readout-pol_lr_nobase control_lr_nobase readout_lr_nobase; do for s in bulk slab; do
    if [ $mode = k1036 ]; then export MLIPCPP_PRECISE_MATMUL_K=1036; unset MLIPCPP_PRECISE_MATMUL; else export MLIPCPP_PRECISE_MATMUL=1; unset MLIPCPP_PRECISE_MATMUL_K; fi
    $B/mlipcpp-eval $W/gguf_pbc/$g.gguf $F/set_${s}_60.extxyz $O/${g}_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1
    echo "gemm $mode $g $s rc=$?"
  done; done
done
unset MLIPCPP_PRECISE_MATMUL MLIPCPP_PRECISE_MATMUL_K
# (b)
O=$W/out/branch_pbc_cuda; mkdir -p $O
for p in lrt_species_fill lrt_species_weighted; do for s in bulk_f32 bulknacl_f32; do
  $B/mlipcpp-eval $W/gguf_branch/$p.gguf $F/set_${s}_12.extxyz $O/${p}_${s}_cuda.json --backend cuda > $O/${p}_${s}.log 2>&1; echo "$p $s rc=$?"
done; done
# (c)
O=$W/out/cc_cuda; mkdir -p $O
for p in cc_fermi cc_residual cc_residual_plain; do for s in mol_z18 bulk_f32; do
  fr=$F/set_$s.extxyz; [ $s = bulk_f32 ] && fr=$F/set_bulk_f32_12.extxyz
  $B/mlipcpp-eval $W/cc/$p.gguf $fr $O/${p}_${s}_cuda.json --backend cuda > $O/${p}_${s}.log 2>&1; echo "$p $s rc=$?"
done; done
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
