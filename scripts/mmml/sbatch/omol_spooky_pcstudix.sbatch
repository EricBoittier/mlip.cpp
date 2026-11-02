#!/bin/bash
# OMol25 SpookyPhysNet (F64, l1, 3 iterations, 32 RBF, cutoff 6 A, Q/S conditioned) with
# mlipcpp-train on one pc-studix GPU, in the JAX pilot's setup (omol_universal/configs/pilot.yaml):
# conservative E + F loss, Adam(W, wd 0) lr 5e-4, 2000 warmup steps, cosine to 0.02 x lr, clip 10,
# pad buckets of 8 atoms with min(256, 210000 / pad^2) structures per step, no rotations.
#
#   sbatch scripts/mmml/sbatch/omol_spooky_pcstudix.sbatch                         # pilot, mmml loss (total E, w_E 1)
#   sbatch --export=ALL,ELOSS=peratom,EW=1000,NAME=pilot_peratom_w1000 scripts/mmml/sbatch/omol_spooky_pcstudix.sbatch
#   sbatch --export=ALL,DATA=$OMOL/mlipcpp_data/full,EPOCHS=40,WARMUP=5000,NAME=full_total scripts/mmml/sbatch/omol_spooky_pcstudix.sbatch
#
# Resumable: submitting the same command again continues from $RUN/model.gguf.state (written every
# epoch and when --max-hours stops the run), e.g. chained: sbatch --dependency=afterany:<jobid> ...
# Logs: $RUN/log.jsonl (per epoch: val E/F MAE + RMSE (meV/atom, meV/A), train loss, lr, wall time),
# stdout (every 200 steps: loss terms, struct/s). Checkpoints: $RUN/model.gguf.epochNNNNN.gguf,
# model.gguf.best.gguf (highest R2(E)+R2(F) on val), model.gguf; at the end also *_mmml.json (mmml).
#SBATCH -p gpu --gres=gpu:1 -c 4 --mem=24G -t 24:00:00
#SBATCH --constraint=RTX3090|RTX3090Ti|RTX4090
#SBATCH --exclude=gpu01,gpu02,gpu04,gpu08,gpu09,gpu10,gpu11,gpu12,node08
#SBATCH -J mlipcpp_omol
#SBATCH -o /mmhome/boittier/home/metatomic-runs/training/omol_universal/mlipcpp_runs/logs/%x.%j.out
set -euo pipefail
REPO=${REPO:-/mmhome/boittier/home/mlip.cpp-dev}
OMOL=/mmhome/boittier/home/metatomic-runs/training/omol_universal
CUDA_ENV=/mmhome/boittier/home/.conda/envs/mmml-gpu          # nvcc / cudart / cuBLAS 12.9 (no CUDA module here)
BUILD=${BUILD:-$REPO/build-cuda-sm86}                         # CUDA_ARCHITECTURES 86;89 (3090 / 4090)
# MT=1: the multitask model (charges, spins, HOMO / gap heads; data *_mt with the npz_aux labels)
# FROZEN_EBIAS=1: the per-element energy_bias stays at its init (0), not trained
MT=${MT:-0}; FROZEN_EBIAS=${FROZEN_EBIAS:-0}
SUF=$([ "$MT" = 1 ] && echo _mt || true)
DATA=${DATA:-$OMOL/mlipcpp_data/pilot200k$SUF}                # train/ val/ (= the JAX 2 % valid split) test/
I0=$OMOL/mlipcpp_runs/init
if [ "$MT" = 1 ]; then INIT=${INIT:-$I0/spooky_mt_f64_l1_i3_seed0_$(basename "${DATA%_mt}").gguf}  # heads + HOMO/gap offsets
elif [ "$FROZEN_EBIAS" = 1 ]; then INIT=${INIT:-$I0/spooky_f64_l1_i3_seed0_frozen_ebias.gguf}
else INIT=${INIT:-$I0/spooky_f64_l1_i3_seed0.gguf}; fi        # flax init, seed 0 (make_random_checkpoint --pristine)
ELOSS=${ELOSS:-total}     # total: mmml's default loss on E_s; peratom: on E_s / N_s (mlipcpp-train's default)
EW=${EW:-1}; FW=${FW:-52.91}
EPOCHS=${EPOCHS:-60}; LR=${LR:-5e-4}; WARMUP=${WARMUP:-2000}; END=${END:-0.02}
PAIRS=${PAIRS:-210000}; CAP=${CAP:-256}
EXTRA=${EXTRA:-}
# multitask loss weights (masked MSE, e^2 / e^2 / eV^2 / eV^2; mmml proposal defaults), Loewdin labels
QW=${QW:-50}; SW=${SW:-20}; HW=${HW:-1}; GW=${GW:-1}
[ "$MT" = 1 ] && EXTRA="$EXTRA --charge-weight $QW --spin-weight $SW --homo-weight $HW --gap-weight $GW --charge-scheme lowdin --spin-scheme lowdin"          # more mlipcpp-train flags (e.g. the multitask weights)
NAME=${NAME:-$(basename "$DATA")_$([ "$MT" = 1 ] && echo mt || echo ef)_${ELOSS}_w${EW}$([ "$FROZEN_EBIAS" = 1 ] && echo _frozenbias || true)}
RUN=${RUN:-$OMOL/mlipcpp_runs/$NAME}
PY=/mmhome/boittier/home/mmml-latest/.venv-cu12/bin/python
export PYTHONPATH=/mmhome/boittier/home/metatomic-runs/training/watmeoh/perf/no_mpi4py:/mmhome/boittier/home/mmml-perf
export LD_LIBRARY_PATH=$CUDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export GGML_CUDA_NO_TF32=1   # full fp32 matmuls (the JAX pilot runs with NVIDIA_TF32_OVERRIDE=0)
mkdir -p "$RUN"
echo "host=$(hostname) job=${SLURM_JOB_ID:-} run=$RUN data=$DATA eloss=$ELOSS w_E=$EW w_F=$FW"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
git -C "$REPO" log --oneline -1

if [ ! -x "$BUILD/bin/mlipcpp-train" ]; then   # normally built on the login node beforehand
  cmake -S "$REPO" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89" \
    -DGGML_NATIVE=OFF -DCMAKE_C_COMPILER=/usr/bin/cc -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
    -DCMAKE_CUDA_COMPILER=$CUDA_ENV/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/c++ -DCUDAToolkit_ROOT=$CUDA_ENV
  cmake --build "$BUILD" -j "${SLURM_CPUS_PER_TASK:-4}" --target mlipcpp-train mlipcpp-jvp-check
fi
B=$BUILD/bin

# 10-second GPU sanity check: forward-mode tangent vs reverse mode, weight gradient vs central differences
$B/mlipcpp-jvp-check --data "$DATA" --init "$INIT" --n 4 --gpus 1 --cache-dir "$RUN/nlcache" 2>&1 | grep -E "^1\.|^2\.|rel|PASS|FAIL" || true

START=(--init "$INIT")
if [ -f "$RUN/model.gguf.state" ]; then START=(--init "$RUN/model.gguf" --resume "$RUN/model.gguf.state"); echo "resuming"; fi
LOSS=()
[ "$ELOSS" = total ] && LOSS=(--energy-per-structure)

set +e
$B/mlipcpp-train --data "$DATA" "${START[@]}" -o "$RUN/model.gguf" --gpus 1 --threads "${SLURM_CPUS_PER_TASK:-4}" \
  --cache-dir "$RUN/nlcache" \
  --conservative --nc-force-weight 0 --model-baselines "${LOSS[@]}" --energy-weight "$EW" --force-weight "$FW" \
  --optimizer adam --lr "$LR" --warmup-steps "$WARMUP" --cosine-end-fraction "$END" --grad-clip 10 \
  --batch-pairs "$PAIRS" --bucket-width 8 --batch-cap "$CAP" --no-rotations --seed 0 \
  --epochs "$EPOCHS" --save-every 1 --state-every 1 --max-hours 23.0 --log "$RUN/log.jsonl" --log-every 200 \
  $EXTRA 2>&1 | tee "$RUN/train.${SLURM_JOB_ID:-local}.out"
rc=${PIPESTATUS[0]}
set -e
echo "mlipcpp-train exit $rc"
# finished (not paused by --max-hours): the best and final checkpoints as mmml portable JSON
if [ $rc = 0 ] && ! grep -q "^paused at epoch" "$RUN/train.${SLURM_JOB_ID:-local}.out"; then
  for g in "$RUN/model.gguf.best.gguf" "$RUN/model.gguf"; do
    [ -f "$g" ] && "$PY" "$REPO/scripts/convert_mmml.py" to-json "$g" -o "${g%.gguf}_mmml.json" || true
  done
fi
exit $rc
