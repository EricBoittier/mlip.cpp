#!/bin/bash -l
#SBATCH --job-name=petlr-mmdef-mig
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: precise MUL_MAT for K = 1036 ON BY DEFAULT (user decision 2026-11-11; opt-out MLIPCPP_PRECISE_MATMUL_K=0).
# Rebuild at the head; GPU tests (precise MUL_MAT op test printed); speed on vs off (readout-pol, bulk_60 and
# mol_z18... the 512-wide checkpoint: bulk set_bulk_60, molecules set_mol_chunked; --repeat 5, median seconds per
# structure); CUDA parity runs at the new default: periodic (6 ckpts x bulk / slab 60) -> out/pbc/cuda_mmdef_mig, open
# (6 ckpts x ion / cluster / mol) -> out/lr/cuda_mmdef_mig; the new presets (field, ladder-LR) -> out/new_cuda_mig.
# Also the multi-component conserving presets (frames/multi) and the field presets (re-exported with graph_multi).
# Evaluation only; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32 MLIPCPP_PRECISE_SIGMOID MLIPCPP_PRECISE_MATH_SKIP MLIPCPP_PRECISE_SUM \
  MLIPCPP_PRECISE_MATMUL MLIPCPP_PRECISE_MATMUL_K MLIPCPP_PRECISE_SQRT MLIPCPP_ENERGY_DOUBLE
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
W=/scratch/boittier/pet_lr_port/cuda; F=/scratch/boittier/pet_lr_port/frames; B=$W/build_lr/bin
cmake --build $W/build_lr -j 5 > $W/build_mmdef.log 2>&1; echo "build rc=$?"; grep -m10 -E " error|error:" $W/build_mmdef.log
nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv
$W/build_lr/tests/test_pet_lr_interp "*MUL_MAT*" -s 2>&1 | grep -A4 "flagged =" | head -6
$W/build_lr/tests/test_pet_lr_interp 2>&1 | grep -E "passed|failed"
$W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E -A8 "FAILED" | head -24
$W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E "passed|failed"
# speed: on (default) vs off (K=0); the first frame of each structure is a warm-up (--repeat)
S=$W/out/speed_mm_mig; mkdir -p $S
for mode in on off; do
  for fam in bulk_60 mol_chunked; do
    fr=$F/set_$fam.extxyz; g=$W/gguf/readout-pol_lr_nobase.gguf; [ $fam = bulk_60 ] && g=$W/gguf_pbc/readout-pol_lr_nobase.gguf
    if [ $mode = off ]; then export MLIPCPP_PRECISE_MATMUL_K=0; else unset MLIPCPP_PRECISE_MATMUL_K; fi
    $B/mlipcpp-eval $g $fr $S/${fam}_$mode.json --backend cuda --repeat 5 > /dev/null 2>&1; echo "speed $mode $fam rc=$?"
  done
done
unset MLIPCPP_PRECISE_MATMUL_K
O=$W/out/pbc/cuda_mmdef_mig; mkdir -p $O
for g in readout-pol control readout readout-1m readout-2m readout-4m; do for s in bulk slab; do
  $B/mlipcpp-eval $W/gguf_pbc/${g}_lr_nobase.gguf $F/set_${s}_60.extxyz $O/${g}_lr_nobase_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1
  echo "pbc $g $s rc=$?"
done; done
O=$W/out/lr/cuda_mmdef_mig; mkdir -p $O
for g in readout-pol control readout readout-1m readout-2m readout-4m; do for s in ion cluster mol; do
  fr=$F/set_$s.extxyz; [ $s = mol ] && fr=$F/set_mol_chunked.extxyz
  $B/mlipcpp-eval $W/gguf/${g}_lr_nobase.gguf $fr $O/${g}_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1
  echo "lr $g $s rc=$?"
done; done
O=$W/out/new_cuda_mig; mkdir -p $O
for p in cc_fermi_field cc_residual_field; do
  $B/mlipcpp-eval $W/gguf_extra/${p}_mc.gguf $F/set_mol_z18_field.extxyz $O/${p}_mol_z18_field_cuda.json --backend cuda > /dev/null 2>&1; echo "$p mol rc=$?"
  MLIPCPP_FIELD_CONSERVATIVE_ONLY=1 $B/mlipcpp-eval $W/gguf_extra/${p}_mc.gguf $F/set_bulk_f32_field.extxyz $O/${p}_bulk_f32_field_cuda.json --backend cuda > /dev/null 2>&1; echo "$p bulk rc=$?"
done
for p in cc_fermi_mc cc_residual_mc; do for s in multi scan_nacl; do  # several components (the multi graph)
  q=${p%_mc}; $B/mlipcpp-eval $W/gguf_extra/$p.gguf $F/multi/${q}_$s.extxyz $O/${q}_${s}_cuda.json --backend cuda > /dev/null 2>&1; echo "$p $s rc=$?"
done; done
for p in thr thr_cold_lrt; do :; done
for p in les_ladder_lr les_ladder_lr_c0; do for s in mol_z18 bulk_f32_12; do
  $B/mlipcpp-eval $W/gguf_extra/$p.gguf $F/set_$s.extxyz $O/${p}_${s}_cuda.json --backend cuda > /dev/null 2>&1; echo "$p $s rc=$?"
done; done
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
