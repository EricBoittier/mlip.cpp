#!/bin/bash
# 40-step smokes of bin6 (gpupool task, one GPU): bin4 vs bin6 on the FTq13 start (E path must reproduce bin4's losses),
# then the A / A+- / AC arms (the extra Fukui passes run, struct/s for the GPU-h estimate). Usage: smoke6.sh ARM [BIN]
set -u
B=/capstor/scratch/cscs/boittier/agents/plato-v2; M=$B/models/ladder; a=$1; tb=${2:-bin6}; d=$B/smoke6/$a-$tb; mkdir -p $d
export GGML_CUDA_NO_TF32=1 MLIPCPP_PAD_EXACT=1; unset NVIDIA_TF32_OVERRIDE
read -r dd ee tt <<< "$(cat $M/$a.data)"; cp -n $M/q.gguf $d/ 2>/dev/null
$B/$tb/mlipcpp-train --data $dd --exclude $ee --init $M/$a.gguf -o $d/model.gguf --gpus 1 --epochs 2 --max-steps 40 --conservative \
  --energy-weight 1 --force-weight 1 --nc-force-weight 0 $(cat $B/opt/chosen.args) --warmup-fraction 0.02 --seed 13 --threads 16 \
  --cache-dir $B/nlcache --log-every 10 $tt > $d/train.log 2>&1
echo "$a $tb rc=$? $(grep -o '[0-9.]* struct/s' $d/train.log | tail -1)"
