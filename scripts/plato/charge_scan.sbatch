#!/usr/bin/env bash
# Molecule-charge loss weight scan: restart (fine-tune) a 4000-epoch long run with
# --charge-weight CW, fresh Adam at a constant learning rate LR, EPOCHS epochs, full validation set
# (with molecules) every 2 epochs. CW=0 is the control: what the restart alone does.
#   SRC=chi CW=0.01 sbatch charge_scan.sbatch     [LR=1e-5 EPOCHS=100 CLIP=1]
#SBATCH --job-name=plato-qscan
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato/logs/%x-%j.out
set -uo pipefail
D=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato
module purge; module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9 nccl
: "${SRC:?}" "${CW:?}"
EP=${EPOCHS:-100}
O=$D/charge-scan/$SRC-w$CW; mkdir -p $O
cp $D/build-cuda/bin/mlipcpp-train-plato $O/
export GGML_CUDA_NO_TF32=1
$O/mlipcpp-train-plato --init $D/long/$SRC/model.gguf --train $D/data-mol/train.bin --val $D/data-mol/val.bin --backend cuda \
  --epochs $EP --batch-size 32 --lr ${LR:-1e-5} --min-lr ${LR:-1e-5} --schedule linear --decay-after $EP --grad-clip ${CLIP:-1} \
  --val-every 2 --seed 0 --charge-weight $CW --save-every 50 -o $O/model.gguf --log $O/log.jsonl 2>&1 \
  | grep -E "^epoch|^done|error" > $O/train.out
tail -3 $O/train.out
