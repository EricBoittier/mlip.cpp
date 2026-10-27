#!/bin/bash -l
#SBATCH --job-name=qcell-smoke
#SBATCH --account=lp161
#SBATCH --partition=debug
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --output=/capstor/scratch/cscs/boittier/qcell/logs/%x-%j.out
# 1) parity: ft-init.gguf (20 species) vs the bio_dimers GGUF on 200 bio_dimers val structures
# 2) zero-shot dumps of ft-init and scratch-init on the 20k train sample (baseline fit), GPUs 0, 1
# 3) timing: 1 epoch of the 20k sample (bs 32), ft-init, muon, GPUs 2, 3 at once (two arms in parallel)
set -uo pipefail
Q=/capstor/scratch/cscs/boittier/qcell
B=$Q/build-cuda/bin
O=$Q/smoke; mkdir -p $O
srun --environment=qm9esp-cuda bash -c "
  export GGML_CUDA_NO_TF32=1 PATH=/users/boittier/venvs/mptrj-py/bin:\$PATH
  cmake --build $Q/build-cuda -j 64 --target mlipcpp-train-plato mlipcpp-plato > $Q/logs/make.log 2>&1 && echo REBUILD OK || { tail -30 $Q/logs/make.log; exit 4; }
  cd $O
  nvidia-smi --query-gpu=index,name,memory.used --format=csv
  (t0=\$(date +%s); CUDA_VISIBLE_DEVICES=0 $B/mlipcpp-plato dump $Q/models/ft-init.gguf $Q/data/xyz/train_s20k.xyz --backend cuda > zs_ft_train20k.jsonl 2> zs_ft.err; echo dump ft \$((\$(date +%s)-t0)) s) &
  (t0=\$(date +%s); CUDA_VISIBLE_DEVICES=1 $B/mlipcpp-plato dump $Q/models/scratch-init-s0.gguf $Q/data/xyz/train_s20k.xyz --backend cuda > zs_scratch_train20k.jsonl 2> zs_scratch.err; echo dump scratch \$((\$(date +%s)-t0)) s) &
  for g in 2 3; do
    opt=muon; [ \$g = 3 ] && opt=adam
    CUDA_VISIBLE_DEVICES=\$g $B/mlipcpp-train-plato --init $Q/models/ft-init.gguf --train $Q/data/xyz/train_s20k.bin --val $Q/data/xyz/train_s20k.bin \
      --backend cuda --epochs 2 --batch-size 32 --lr 1e-4 --schedule linear --decay-after 1 --min-lr 1e-6 --grad-clip 1 --val-every 1 \
      --optimizer \$opt --muon-lr-scale 3 --fd-eps 5e-3 --seed 0 --save-every 100 --state-every 1 --max-epochs-this-run 1 --bucket-growth 1.3 --share-graph-buffer --graph-cache 256 \
      -o $O/t\$g.gguf --log $O/t\$g.jsonl 2>&1 | grep -v '^ggml_cuda\|Device \|warmup\|^Backend\|  step ' > $O/t\$g.out &
  done
  for i in 1 2 3 4 5 6; do sleep 120; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; done; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
  wait
  tail -n 4 $O/t2.out $O/t3.out
"
