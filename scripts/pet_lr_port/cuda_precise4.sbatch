#!/bin/bash -l
#SBATCH --job-name=petlr-precise4
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: rebuild the CUDA tree with ggml-cuda-precise-sum.patch (SUM / SUM_ROWS accumulating in double when
# flagged, applied by hand to the pre-patched deps/ggml before this job), the GPU tests, then
#  (a) per-node dumps of readout-pol LR slab (3 frames), sum flagged vs MLIPCPP_PRECISE_MATH_SKIP=sum (does the flag
#      reach the potential sums?);
#  (b) seeded lr_both ion: MLIPCPP_PRECISE_SIGMOID 0/1 x sum flagged / skipped, 40 frames (mu P) + 3-frame dumps;
#  (c) the charge families (readout-pol, control, readout; bulk + slab) with SKIP=sum (A/B against (d));
#  (d) every periodic + open LR GGUF at the new head config -> out/pbc/cuda4, out/lr/cuda4.
# Evaluation only; FP32, no TF32. (Ran as job 4891425 while SUM was flagged by default; SUM is now opt-in,
# MLIPCPP_PRECISE_SUM=1, so "skip=none" here = MLIPCPP_PRECISE_SUM=1 at head and cuda4 = head + that flag.)
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32 MLIPCPP_PRECISE_SIGMOID MLIPCPP_PRECISE_MATH_SKIP
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
W=/scratch/boittier/pet_lr_port/cuda; P=/scratch/boittier/pet_lr_port; F=$P/frames; N=$W/nodegrad; B=$W/build_lr/bin
cmake --build $W/build_lr -j 16 > $W/build_precise4.log 2>&1; echo "build rc=$?"; grep -m10 -E " error|error:" $W/build_precise4.log
$W/build_lr/tests/test_pet_lr_interp "*precise*" -s 2>&1 | grep -E "relative|SUM_ROWS|passed|FAILED" | tail -8
$W/build_lr/tests/test_pet_lr_interp 2>&1 | grep -E "passed|failed"
$W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E -A12 "FAILED" | head -40
$W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E "passed|failed"
# (a)
for k in sum none; do
  rm -f $N/pol_slab_s${k}_cuda.bin
  MLIPCPP_PRECISE_MATH_SKIP=$k MLIPCPP_NODE_GRAD_DUMP=$N/pol_slab_s${k}_cuda.bin $B/mlipcpp-eval $N/readout-pol_lr_v5.gguf $N/slab_60_3.extxyz $N/pol_slab_s${k}_cuda.json --backend cuda > /dev/null 2>&1; echo "dump pol_slab skip=$k rc=$?"
done
# (b)
O=$W/out/mu_sum; mkdir -p $O
for sg in 0 1; do for k in sum none; do
  MLIPCPP_PRECISE_SIGMOID=$sg MLIPCPP_PRECISE_MATH_SKIP=$k $B/mlipcpp-eval $W/gguf/seeded_lr_both.gguf $F/set_ion_40.extxyz $O/sig${sg}_skip${k}.json --backend cuda > /dev/null 2>&1; echo "mu sig=$sg skip=$k rc=$?"
  rm -f $N/both_ion_sig${sg}_skip${k}_cuda.bin
  MLIPCPP_PRECISE_SIGMOID=$sg MLIPCPP_PRECISE_MATH_SKIP=$k MLIPCPP_NODE_GRAD_DUMP=$N/both_ion_sig${sg}_skip${k}_cuda.bin $B/mlipcpp-eval $N/seeded_lr_both_v5.gguf $N/ion_3.extxyz $N/both_ion_sig${sg}_skip${k}_cuda.json --backend cuda > /dev/null 2>&1; echo "dump both_ion sig=$sg skip=$k rc=$?"
done; done
# (c)
O=$W/out/pbc/cuda4_skipsum; mkdir -p $O
for g in readout-pol_lr_nobase control_lr_nobase readout_lr_nobase; do for s in bulk slab; do
  MLIPCPP_PRECISE_MATH_SKIP=sum $B/mlipcpp-eval $W/gguf_pbc/$g.gguf $F/set_${s}_60.extxyz $O/${g}_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1; echo "skipsum $g $s rc=$?"
done; done
# (d)
O=$W/out/pbc/cuda4; mkdir -p $O
for g in seeded_lr_fermi_cold seeded_lr_both readout-pol_lr_nobase control_lr_nobase readout_lr_nobase readout-1m_lr_nobase readout-2m_lr_nobase readout-4m_lr_nobase seeded_lr_species; do
  for s in bulk slab; do
    $B/mlipcpp-eval $W/gguf_pbc/$g.gguf $F/set_${s}_60.extxyz $O/${g}_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1
    echo "$g $s rc=$? $(tail -n1 $O/${g}_${s}.log)"
  done
done
O=$W/out/lr/cuda4; mkdir -p $O
for g in readout-pol control readout readout-1m readout-2m readout-4m; do for s in ion cluster; do
  $B/mlipcpp-eval $W/gguf/${g}_lr_nobase.gguf $F/set_$s.extxyz $O/${g}_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1
  echo "$g $s rc=$?"; done; done
for g in seeded_lr_species seeded_lr_both seeded_lr_fermi_cold; do for s in ion cluster; do
  $B/mlipcpp-eval $W/gguf/$g.gguf $F/set_${s}.extxyz $O/${g}_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1
  echo "$g $s rc=$?"; done; done
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
