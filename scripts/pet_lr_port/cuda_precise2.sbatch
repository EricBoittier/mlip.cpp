#!/bin/bash -l
#SBATCH --job-name=petlr-precise2
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: rebuild the CUDA tree with ggml-cuda-precise-sigmoid-div.patch (applied by hand to the pre-patched
# deps/ggml before this job), the GPU tests (incl. the flagged SIGMOID / DIV op test), then the intervention: the
# periodic + open LR GGUFs on CUDA again (out/pbc/cuda2/, out/lr/cuda2/) and the per-node dump of seeded
# lr_fermi_cold slab. Evaluation only; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
W=/scratch/boittier/pet_lr_port/cuda; P=/scratch/boittier/pet_lr_port; F=$P/frames; N=$W/nodegrad
cmake --build $W/build_lr -j 16 > $W/build_precise2.log 2>&1; echo "build rc=$?"; grep -m10 -E " error|error:" $W/build_precise2.log
$W/build_lr/tests/test_pet_lr_interp "*precise*" -s 2>&1 | grep -E "relative|passed|FAILED" | tail -6
$W/build_lr/tests/test_pet_lr_interp 2>&1 | grep -E "passed|failed"
$W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E -A12 "FAILED" | head -40
$W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E "passed|failed"
rm -f $N/cold_slab_p2_cuda.bin
MLIPCPP_NODE_GRAD_DUMP=$N/cold_slab_p2_cuda.bin $W/build_lr/bin/mlipcpp-eval $N/seeded_lr_fermi_cold_v5.gguf $N/slab_60_3.extxyz $N/cold_slab_p2_cuda.json --backend cuda > /dev/null 2>&1; echo "dump rc=$?"
O=$W/out/pbc/cuda3; mkdir -p $O
for g in ${GS:-seeded_lr_fermi_cold seeded_lr_both readout-pol_lr_nobase control_lr_nobase readout_lr_nobase readout-1m_lr_nobase readout-2m_lr_nobase readout-4m_lr_nobase seeded_lr_species seeded_lr_both}; do
  for s in bulk slab; do
    $W/build_lr/bin/mlipcpp-eval $W/gguf_pbc/$g.gguf $F/set_${s}_60.extxyz $O/${g}_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1
    echo "$g $s rc=$? $(tail -n1 $O/${g}_${s}.log)"
  done
done
O=$W/out/lr/cuda3; mkdir -p $O
for g in readout-pol control readout readout-1m readout-2m readout-4m; do for s in ion cluster; do
  $W/build_lr/bin/mlipcpp-eval $W/gguf/${g}_lr_nobase.gguf $F/set_$s.extxyz $O/${g}_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1
  echo "$g $s rc=$?"; done; done
for g in seeded_lr_species seeded_lr_both seeded_lr_fermi_cold; do for s in ion cluster; do
  $W/build_lr/bin/mlipcpp-eval $W/gguf/$g.gguf $F/set_${s}.extxyz $O/${g}_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1
  echo "$g $s rc=$?"; done; done
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
