#!/bin/bash
# sciCORE: OMol25 SpookyPhysNet with mlipcpp-train on one A100 (default) or H200; the setup of
# omol_spooky_pcstudix.sbatch (the JAX pilot's model, loss, optimizer and batch policy).
# After omol_spooky_scicore_setup.sbatch:
#   cd ~/octanol_runs/pcs_omol/mlipcpp
#   sbatch --qos=a100-30min --time=00:30:00 --export=ALL,CHECK=1 src/scripts/mmml/sbatch/omol_spooky_scicore.sbatch  # GPU check + 1 short epoch
#   sbatch src/scripts/mmml/sbatch/omol_spooky_scicore.sbatch                                                   # pilot, total-E loss
#   sbatch --export=ALL,ELOSS=peratom,EW=1000,NAME=pilot_peratom_w1000 src/scripts/mmml/sbatch/omol_spooky_scicore.sbatch
#   sbatch --partition=h200 --qos=h200-1day --gres=gpu:h200:1 --export=ALL,DATA=data/full,EPOCHS=40,WARMUP=5000,NAME=full_total ...
# Resumable: the same command again continues from $RUN/model.gguf.state.
#SBATCH --job-name=mlipcpp_omol
#SBATCH --partition=a100
#SBATCH --qos=a100-1day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scicore/home/meuwly/boitti0000/octanol_runs/pcs_omol/mlipcpp/logs/%x.%j.out
set -euo pipefail
P=/scicore/home/meuwly/boitti0000/octanol_runs/pcs_omol
M=$P/mlipcpp
SRC=$M/src
B=${BUILD:-$M/build-sm80-90}/bin
# MT=1: the multitask model (needs the *_mt data and inits copied from pc-studix, see the setup sbatch)
# RMSNORM=1: SpookyPhysNet feature_rmsnorm (mmml 1b5f7f615); init spooky_f64_l1_i3_seed0_rms.gguf (setup sbatch / pc-studix)
MT=${MT:-0}; FROZEN_EBIAS=${FROZEN_EBIAS:-0}; RMSNORM=${RMSNORM:-0}
[ "$RMSNORM" = 1 ] && { [ "$MT" = 1 ] || [ "$FROZEN_EBIAS" = 1 ]; } && { echo "RMSNORM=1: no MT / FROZEN_EBIAS init yet (pass INIT=)"; [ -n "${INIT:-}" ] || exit 2; }
SUF=$([ "$MT" = 1 ] && echo _mt || true)
DATA=${DATA:-$M/data/pilot200k$SUF}; [[ $DATA = /* ]] || DATA=$M/$DATA
if [ "$MT" = 1 ]; then INIT=${INIT:-$M/init/spooky_mt_f64_l1_i3_seed0_$(basename "${DATA%_mt}").gguf}
elif [ "$FROZEN_EBIAS" = 1 ]; then INIT=${INIT:-$M/init/spooky_f64_l1_i3_seed0_frozen_ebias.gguf}
elif [ "$RMSNORM" = 1 ]; then INIT=${INIT:-$M/init/spooky_f64_l1_i3_seed0_rms.gguf}
else INIT=${INIT:-$M/init/spooky_f64_l1_i3_seed0.gguf}; fi
ELOSS=${ELOSS:-total}; EW=${EW:-1}; FW=${FW:-52.91}
EPOCHS=${EPOCHS:-60}; LR=${LR:-5e-4}; WARMUP=${WARMUP:-2000}; END=${END:-0.02}
PAIRS=${PAIRS:-210000}; CAP=${CAP:-256}; EXTRA=${EXTRA:-}
# multitask loss weights (masked MSE, e^2 / e^2 / eV^2 / eV^2; mmml proposal defaults), Loewdin labels
QW=${QW:-50}; SW=${SW:-20}; HW=${HW:-1}; GW=${GW:-1}
[ "$MT" = 1 ] && EXTRA="$EXTRA --charge-weight $QW --spin-weight $SW --homo-weight $HW --gap-weight $GW --charge-scheme lowdin --spin-scheme lowdin"
NAME=${NAME:-$(basename "$DATA")_$([ "$MT" = 1 ] && echo mt || echo ef)_${ELOSS}_w${EW}$([ "$FROZEN_EBIAS" = 1 ] && echo _frozenbias || true)$([ "$RMSNORM" = 1 ] && echo _rms || true)}
RUN=${RUN:-$M/runs/$NAME}
HOURS=${HOURS:-23.0}
PY=/scicore/home/meuwly/boitti0000/mmml_new/.venv/bin/python
set +u   # Lmod init and module() are not nounset-safe; batch shells are non-login
[ -f /etc/profile.d/soft_stacks.sh ] && source /etc/profile.d/soft_stacks.sh
module purge
module load GCCcore/14.3.0 CUDA/12.9.1 NCCL/2.27.7-GCCcore-14.3.0-CUDA-12.9.1
set -u
export GGML_CUDA_NO_TF32=1
mkdir -p "$RUN"
echo "host=$(hostname) job=${SLURM_JOB_ID:-} run=$RUN data=$DATA eloss=$ELOSS w_E=$EW w_F=$FW"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
git -C "$SRC" log --oneline -1
$B/mlipcpp-jvp-check --data "$DATA" --init "$INIT" --n 4 --gpus 1 --cache-dir "$RUN/nlcache" 2>&1 | grep -E "^1\.|^2\.|rel|PASS|FAIL" || true
if [ "${CHECK:-0}" = 1 ]; then   # 30-min check: gradients on the GPU and ~1000 steps for the throughput
  EPOCHS=1; HOURS=0.3; RUN=$M/runs/check_${SLURM_JOB_ID:-local}; mkdir -p "$RUN"; EXTRA="$EXTRA --max-steps 1000"
  $B/mlipcpp-train-check --data "$DATA" --init "$INIT" --out "$RUN/gc" --n 16 --gpus 1 --conservative 1 \
    --energy-weight 1 --force-weight 52.91 --nc-force-weight 0 --model-baselines 1 --energy-per-structure 1 2>&1 | tail -2
fi
START=(--init "$INIT")
if [ -f "$RUN/model.gguf.state" ]; then START=(--init "$RUN/model.gguf" --resume "$RUN/model.gguf.state"); echo resuming; fi
LOSS=(); [ "$ELOSS" = total ] && LOSS=(--energy-per-structure)
set +e
$B/mlipcpp-train --data "$DATA" "${START[@]}" -o "$RUN/model.gguf" --gpus 1 --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --cache-dir "$RUN/nlcache" \
  --conservative --nc-force-weight 0 --model-baselines "${LOSS[@]}" --energy-weight "$EW" --force-weight "$FW" \
  --optimizer adam --lr "$LR" --warmup-steps "$WARMUP" --cosine-end-fraction "$END" --grad-clip 10 \
  --batch-pairs "$PAIRS" --bucket-width 8 --batch-cap "$CAP" --no-rotations --seed 0 \
  --epochs "$EPOCHS" --save-every 1 --state-every 1 --max-hours "$HOURS" --log "$RUN/log.jsonl" --log-every 200 \
  $EXTRA 2>&1 | tee "$RUN/train.${SLURM_JOB_ID:-local}.out"
rc=${PIPESTATUS[0]}
set -e
echo "mlipcpp-train exit $rc"
if [ $rc = 0 ] && ! grep -q "^paused at epoch" "$RUN/train.${SLURM_JOB_ID:-local}.out"; then
  for g in "$RUN/model.gguf.best.gguf" "$RUN/model.gguf"; do
    [ -f "$g" ] && "$PY" "$SRC/scripts/convert_mmml.py" to-json "$g" -o "${g%.gguf}_mmml.json" || true
  done
fi
exit $rc
