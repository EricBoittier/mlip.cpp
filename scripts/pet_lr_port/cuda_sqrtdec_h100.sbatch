#!/bin/bash -l
#SBATCH --job-name=petlr-sqrtdec
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port, full H100. Replaces cuda_followup_h100.sbatch (4894719 / 4894720: an unclosed `if` -> bash syntax error
# at EOF, nothing ran). Trees: build_lr = main tree (old default: precise MUL_MAT naive), build_kv = kernel-variant tree
# (MLIPCPP_PRECISE_MATMUL_KERNEL unset = naive there), build_td = the committed head 6e2ead3 (unset = tiled).
# PRECISE_SQRT decision on build_td at the head default (tiled precise MUL_MAT), MLIPCPP_PRECISE_SQRT 0 / 1, same
# binary: periodic charge tables -> out/pbc/td_h100_sq<s>, open -> out/lr/td_h100_sq<s>, field / multi / ladder-LR /
# threshold / induced -> out/new_cuda_td_h100_sq<s>, les / seeded LR / branch / cc / qeq -> out/sqrtdec_h100_sq<s>;
# speed (--repeat 5) and GPU tests -> out/speed_td_h100.
# Evaluation only; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32 MLIPCPP_PRECISE_SIGMOID MLIPCPP_PRECISE_MATH_SKIP MLIPCPP_PRECISE_SUM \
  MLIPCPP_PRECISE_MATMUL MLIPCPP_PRECISE_MATMUL_K MLIPCPP_PRECISE_SQRT MLIPCPP_ENERGY_DOUBLE MLIPCPP_PRECISE_MATMUL_KERNEL
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
C=/scratch/boittier/pet_lr_port/cuda; W=$C; F=/scratch/boittier/pet_lr_port/frames
nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv
# (f)
BT=$C/build_td; B=$BT/bin
cmake --build $BT -j 16 > $C/build_td_h100.log 2>&1; echo "build_td rc=$?"; grep -m10 -E " error|error:" $C/build_td_h100.log
for sq in 0 1; do
  export MLIPCPP_PRECISE_SQRT=$sq
  O=$C/out/pbc/td_h100_sq$sq; mkdir -p $O
  for g in readout-pol control readout readout-1m readout-2m readout-4m; do for s in bulk slab; do
    $B/mlipcpp-eval $W/gguf_pbc/${g}_lr_nobase.gguf $F/set_${s}_60.extxyz $O/${g}_lr_nobase_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1
    echo "sq$sq pbc $g $s rc=$?"
  done; done
  O=$C/out/lr/td_h100_sq$sq; mkdir -p $O
  for g in readout-pol control readout readout-1m readout-2m readout-4m; do for s in ion cluster mol; do
    fr=$F/set_$s.extxyz; [ $s = mol ] && fr=$F/set_mol_chunked.extxyz
    $B/mlipcpp-eval $W/gguf/${g}_lr_nobase.gguf $fr $O/${g}_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1
    echo "sq$sq lr $g $s rc=$?"
  done; done
  O=$C/out/new_cuda_td_h100_sq$sq; mkdir -p $O
  for p in cc_fermi_field cc_residual_field; do
    $B/mlipcpp-eval $W/gguf_extra/${p}_mc.gguf $F/set_mol_z18_field.extxyz $O/${p}_mol_z18_field_cuda.json --backend cuda > /dev/null 2>&1; echo "sq$sq $p mol rc=$?"
    MLIPCPP_FIELD_CONSERVATIVE_ONLY=1 $B/mlipcpp-eval $W/gguf_extra/${p}_mc.gguf $F/set_bulk_f32_field.extxyz $O/${p}_bulk_f32_field_cuda.json --backend cuda > /dev/null 2>&1; echo "sq$sq $p bulk rc=$?"
    $B/mlipcpp-eval $W/gguf_extra/${p}_mc.gguf $F/set_bulk_f32_field.extxyz $O/${p}_bulk_f32_field_full_cuda.json --backend cuda > /dev/null 2>&1; echo "sq$sq $p bulk full rc=$?"
  done
  for p in cc_fermi_mc cc_residual_mc; do for s in multi scan_nacl; do
    q=${p%_mc}; $B/mlipcpp-eval $W/gguf_extra/$p.gguf $F/multi/${q}_$s.extxyz $O/${q}_${s}_cuda.json --backend cuda > /dev/null 2>&1; echo "sq$sq $p $s rc=$?"
  done; done
  for p in les_ladder_lr les_ladder_lr_c0; do for s in mol_z18 bulk_f32_12; do
    $B/mlipcpp-eval $W/gguf_extra/$p.gguf $F/set_$s.extxyz $O/${p}_${s}_cuda.json --backend cuda > /dev/null 2>&1; echo "sq$sq $p $s rc=$?"
  done; done
  for p in thr thr_cold_lrt; do for s in mol_z18 cluster bulk_f32; do
    $B/mlipcpp-eval $W/gguf_thr/$p.gguf $F/set_$s.extxyz $O/${p}_${s}_cuda.json --backend cuda > /dev/null 2>&1; echo "sq$sq $p $s rc=$?"
  done; done
  for p in cc_ind_field cc_ind_full; do
    for s in mol_z18_field mol_z18_field0; do
      $B/mlipcpp-eval $W/gguf_extra/$p.gguf $F/set_$s.extxyz $O/${p}_${s}_cuda.json --backend cuda > /dev/null 2>&1; echo "sq$sq $p $s rc=$?"
    done
    MLIPCPP_FIELD_CONSERVATIVE_ONLY=1 $B/mlipcpp-eval $W/gguf_extra/$p.gguf $F/set_bulk_f32_field.extxyz $O/${p}_bulk_f32_field_cuda.json --backend cuda > /dev/null 2>&1; echo "sq$sq $p bulk rc=$?"
  done
  O2=$C/out/sqrtdec_h100_sq$sq; mkdir -p $O2
  for p in les_local2 les_local3 les_ladder les_ladder_c0; do for s in mol_z18 bulk_f32; do
    $B/mlipcpp-eval $W/gguf_les/$p.gguf $F/set_$s.extxyz $O2/${p}_${s}_cuda.json --backend cuda > /dev/null 2>&1; echo "sq$sq $p $s rc=$?"
  done; done
  for g in seeded_lr_species seeded_lr_both seeded_lr_fermi_cold; do for s in ion cluster; do
    $B/mlipcpp-eval $W/gguf/$g.gguf $F/set_$s.extxyz $O2/${g}_${s}_cuda.json --backend cuda > /dev/null 2>&1; echo "sq$sq $g $s rc=$?"
  done; done
  for p in lrt_weighted lrt_fill lrt_fill_cold lrt_species_fill lrt_species_weighted; do for s in cluster mol_z18; do
    fr=$F/set_${s}.extxyz; [ $s = cluster ] && fr=$F/set_cluster_40.extxyz
    $B/mlipcpp-eval $W/gguf_branch/$p.gguf $fr $O2/${p}_${s}_cuda.json --backend cuda > /dev/null 2>&1; echo "sq$sq $p $s rc=$?"
  done; done
  for p in lrt_species_fill lrt_species_weighted; do for s in bulk_f32 bulknacl_f32; do
    $B/mlipcpp-eval $W/gguf_branch/$p.gguf $F/set_${s}_12.extxyz $O2/${p}_${s}_cuda.json --backend cuda > /dev/null 2>&1; echo "sq$sq $p $s rc=$?"
  done; done
  for p in cc_fermi cc_residual cc_residual_plain; do for s in mol_z18 bulk_f32; do
    fr=$F/set_$s.extxyz; [ $s = bulk_f32 ] && fr=$F/set_bulk_f32_12.extxyz
    $B/mlipcpp-eval $W/cc/$p.gguf $fr $O2/${p}_${s}_cuda.json --backend cuda > /dev/null 2>&1; echo "sq$sq $p $s rc=$?"
  done; done
  for s in mol_z18 bulk_f32; do
    $B/mlipcpp-eval $W/gguf_qeq/tiny_qeq.gguf $F/set_$s.extxyz $O2/tiny_qeq_${s}_cuda.json --backend cuda > /dev/null 2>&1; echo "sq$sq qeq $s rc=$?"
  done
  S=$C/out/speed_td_h100; mkdir -p $S
  for fam in bulk_60 mol_chunked; do
    fr=$F/set_$fam.extxyz; g=$W/gguf/readout-pol_lr_nobase.gguf; [ $fam = bulk_60 ] && g=$W/gguf_pbc/readout-pol_lr_nobase.gguf
    $B/mlipcpp-eval $g $fr $S/${fam}_sq$sq.json --backend cuda --repeat 5 > /dev/null 2>&1; echo "speed sq$sq $fam rc=$?"
  done
  $B/mlipcpp-eval $W/gguf_extra/les_ladder_lr.gguf $F/set_mol_z18.extxyz $S/ladder_mol_z18_sq$sq.json --backend cuda --repeat 5 > /dev/null 2>&1; echo "speed sq$sq ladder rc=$?"
  echo "== tests (build_td) PRECISE_SQRT=$sq"; T=$C/out/speed_td_h100/tests_sq$sq; 
  $BT/tests/test_pet_lr_interp "*SQRT*" -s > $T.sqrt_op.txt 2>&1; grep -E "flagged =" -A1 $T.sqrt_op.txt | head -3
  $BT/tests/test_pet_lr_port > $T.port.txt 2>&1; grep -E -A8 "FAILED" $T.port.txt | head -24; grep -E "passed|failed" $T.port.txt
  $BT/tests/test_pet_lr_interp > $T.interp.txt 2>&1; grep -E "passed|failed" $T.interp.txt
done
unset MLIPCPP_PRECISE_SQRT
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
echo "ALL DONE"
