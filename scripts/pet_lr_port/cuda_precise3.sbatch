#!/bin/bash -l
#SBATCH --job-name=petlr-precise3
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port, ion-F / charges interventions on CUDA (H100, FP32, no TF32; evaluation only):
#  (a) the 204-frame readout-pol TRUNK force split with the precise-math flags forced on (MLIPCPP_PRECISE_MATH=1:
#      SIN / COS / LOG of the cutoff function, 4-5x the local fp32 error with the fast kernels per the node oracle);
#  (b) per-node dump of readout-pol slab with the SIGMOID / DIV fix (charges); (c) the trunk ion dump twice
#      (default flags) to locate the run-to-run nondeterministic backward node.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
W=/scratch/boittier/pet_lr_port/cuda; P=/scratch/boittier/pet_lr_port; F=$P/frames; N=$W/nodegrad
cmake --build $W/build_lr -j 16 > $W/build_precise3.log 2>&1; echo "build rc=$?"; grep -m10 -E " error|error:" $W/build_precise3.log
$W/build_lr/tests/test_pet_lr_port 2>&1 | tail -1
rm -f $W/out/ionF_pol_trunk_cuda_precise.bin
MLIPCPP_PRECISE_MATH=1 MLIPCPP_NO_GRAPH_CACHE=1 MLIPCPP_FORCE_SPLIT=$W/out/ionF_pol_trunk_cuda_precise.bin $W/build_lr/bin/mlipcpp-eval \
  $W/gguf/readout-pol_direct_nobase.gguf $F/set_ion.extxyz $W/out/ionF_pol_trunk_cuda_precise.json --backend cuda > /dev/null 2>&1; echo "split precise rc=$?"
rm -f $W/out/ionF_pol_trunk_cuda_rerun.bin
MLIPCPP_NO_GRAPH_CACHE=1 MLIPCPP_FORCE_SPLIT=$W/out/ionF_pol_trunk_cuda_rerun.bin $W/build_lr/bin/mlipcpp-eval \
  $W/gguf/readout-pol_direct_nobase.gguf $F/set_ion.extxyz $W/out/ionF_pol_trunk_cuda_rerun.json --backend cuda > /dev/null 2>&1; echo "split default rerun rc=$?"
run() { rm -f $N/$1_cuda.bin; MLIPCPP_NODE_GRAD_DUMP=$N/$1_cuda.bin $W/build_lr/bin/mlipcpp-eval $N/$2 $N/$3 $N/$1_cuda.json --backend cuda > $N/$1.log 2>&1; echo "$1 rc=$?"; }
run pol_slab_p3 readout-pol_lr_v5.gguf slab_60_3.extxyz
run pol_trunk_ion_r1 readout-pol_trunk_v5.gguf ion_8.extxyz
run pol_trunk_ion_r2 readout-pol_trunk_v5.gguf ion_8.extxyz
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
