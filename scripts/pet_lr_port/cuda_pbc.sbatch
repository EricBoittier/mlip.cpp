#!/bin/bash -l
#SBATCH --job-name=petlr-cuda-pbc
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: periodic iris PET-LR on CUDA (H100, FP32, no TF32): rebuild, the PET-LR tests on the GPU (incl. the
# periodic FD tests), the periodic GGUFs (gguf_pbc/: open + graph_pbc.json) on bulk / slab (first 60 frames), and a
# run-to-run determinism check of the CUDA forces (the same trunk GGUF twice). Evaluation only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
W=/scratch/boittier/pet_lr_port/cuda; P=/scratch/boittier/pet_lr_port; F=$P/frames
cmake --build $W/build_lr -j 16 > $W/build_pbc.log 2>&1; echo "build rc=$?"; grep -m10 -E " error|error:" $W/build_pbc.log
$W/build_lr/tests/test_pet_lr_interp 2>&1 | tail -1
$W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E "FAILED|max \|" | head -20
$W/build_lr/tests/test_pet_lr_port 2>&1 | tail -2
O=$W/out/pbc; mkdir -p $O
for r in 1 2; do $W/build_lr/bin/mlipcpp-eval $W/gguf/readout-pol_direct_nobase.gguf $F/set_ion.extxyz $O/determinism_trunk_ion_$r.json --backend cuda > /dev/null 2>&1; done
for g in ${GS:-readout-pol_lr_nobase control_lr_nobase readout_lr_nobase readout-1m_lr_nobase readout-2m_lr_nobase readout-4m_lr_nobase seeded_lr_species seeded_lr_both seeded_lr_fermi_cold}; do
  for s in bulk slab; do
    $W/build_lr/bin/mlipcpp-eval $W/gguf_pbc/$g.gguf $F/set_${s}_60.extxyz $O/${g}_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1
    echo "$g $s rc=$? $(tail -n1 $O/${g}_${s}.log)"
  done
done
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
