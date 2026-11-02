#!/bin/bash
# sciCORE, one-off (re-runnable, each step skips what exists): clone feat/physnet-spooky, CUDA build for
# A100 + H200 (sm_80, sm_90), the mlipcpp-train data from the Orbax caches already in
# ~/octanol_runs/pcs_omol/caches (same structures / split / energies as on pc-studix: byte-identical
# conversion), the init GGUF. Everything under ~/octanol_runs/pcs_omol/mlipcpp/. Build: CPU node, 30 min.
#   cd ~/octanol_runs/pcs_omol/mlipcpp && sbatch omol_spooky_scicore_setup.sbatch
#   (rerun if it times out: the build and the conversions continue / skip)
# then the GPU check: sbatch --export=ALL,CHECK=1 omol_spooky_scicore.sbatch   (a100, 30 min)
#SBATCH --job-name=mlipcpp_setup
#SBATCH --partition=scicore
#SBATCH --qos=30min
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/scicore/home/meuwly/boitti0000/octanol_runs/pcs_omol/mlipcpp/logs/setup.%j.out
set -euo pipefail
P=/scicore/home/meuwly/boitti0000/octanol_runs/pcs_omol
M=$P/mlipcpp
SRC=$M/src
BUILD=$M/build-sm80-90
PY=/scicore/home/meuwly/boitti0000/mmml_new/.venv/bin/python   # orbax, jax (cpu use here), flax
BRANCH=${BRANCH:-feat/physnet-spooky}
mkdir -p "$M/logs" "$M/data" "$M/init"
set +u   # Lmod init and module() are not nounset-safe; batch shells are non-login
[ -f /etc/profile.d/soft_stacks.sh ] && source /etc/profile.d/soft_stacks.sh
module purge
module load GCCcore/14.3.0 CMake/3.31.8-GCCcore-14.3.0 CUDA/12.9.1 NCCL/2.27.7-GCCcore-14.3.0-CUDA-12.9.1
set -u
echo "host=$(hostname) job=${SLURM_JOB_ID:-}"

# 1. source
if [ ! -d "$SRC/.git" ]; then git clone --branch "$BRANCH" https://github.com/EricBoittier/mlip.cpp.git "$SRC"
else git -C "$SRC" fetch origin "$BRANCH" && git -C "$SRC" checkout -q "$BRANCH" && git -C "$SRC" merge -q --ff-only "origin/$BRANCH"; fi
git -C "$SRC" log --oneline -1

# 2. data (CPU, minutes): pilot200k train / val (= the JAX trainer's 2 % valid split, seed 0) / test (val50k
# re-referenced to the pilot's linear reference); full train (2.49M) / val50k (full reference)
export JAX_PLATFORMS=cpu
C=$P/caches
conv() { "$PY" -I "$SRC/scripts/mmml/orbax_cache_to_memmap.py" "$@"; }
[ -f "$M/data/pilot200k/train/ns.npy" ] || conv "$C/omol_train4M_pilot200k" -o "$M/data/pilot200k" --valid-fraction 0.02 --split-seed 0 | tail -2
[ -f "$M/data/val50k_pilotref/val/ns.npy" ] || conv "$C/omol_val_50k" -o "$M/data/val50k_pilotref" --split-name val \
  --rereference "$C/omol_train4M_filtered/metadata.json" "$C/omol_train4M_pilot200k/metadata.json" | tail -1
[ -f "$M/data/val50k_fullref/val/ns.npy" ] || conv "$C/omol_val_50k" -o "$M/data/val50k_fullref" --split-name val | tail -1
ln -sfn ../val50k_pilotref/val "$M/data/pilot200k/test"
[ -f "$M/data/full/train/ns.npy" ] || conv "$C/omol_train4M_filtered" -o "$M/data/full" --split-name train | tail -1
ln -sfn ../val50k_fullref/val "$M/data/full/val"; ln -sfn ../val50k_fullref/val "$M/data/full/test"

# 3. init: flax's initialisation with seed 0 (as train_so3lr_spooky_extxyz.py --seed 0), as GGUF
if [ ! -f "$M/init/spooky_f64_l1_i3_seed0.gguf" ]; then
  PYTHONPATH=$P/mmml "$PY" "$SRC/scripts/mmml/make_random_checkpoint.py" --model-type spooky --features 64 --max-degree 1 \
    --num-iterations 3 --num-basis-functions 32 --n-res 2 --max-atomic-number 87 --cutoff 6.0 --seed 0 --pristine \
    -o "$M/init/spooky_f64_l1_i3_seed0.json"
  "$PY" "$SRC/scripts/convert_mmml.py" to-gguf "$M/init/spooky_f64_l1_i3_seed0.json" -o "$M/init/spooky_f64_l1_i3_seed0.gguf"
  "$PY" "$SRC/scripts/convert_mmml.py" to-gguf "$M/init/spooky_f64_l1_i3_seed0.json" --frozen energy_bias \
    -o "$M/init/spooky_f64_l1_i3_seed0_frozen_ebias.gguf"
fi
# multitask (MT=1 in omol_spooky_scicore.sbatch): the label data are not in the sciCORE caches; copy from
# pc-studix (user terminal): mlipcpp_data/{pilot200k_mt,val50k_pilotref_mt} -> $M/data/, and
# mlipcpp_runs/init/spooky_mt_f64_l1_i3_seed0_pilot200k.gguf -> $M/init/
[ -d "$M/data/pilot200k_mt" ] && ln -sfn ../val50k_pilotref_mt/val "$M/data/pilot200k_mt/test" || true

# 4. CUDA build (incremental: a rerun continues)
# sciCORE compute nodes have no internet: CPM downloads time out. Configure once on the login node first
# (bash -lc 'module load <the modules above>; cmake -S src -B build-sm80-90 <same flags>'); this step then reuses _deps.
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;90" \
  -DGGML_NATIVE=OFF > "$M/logs/cmake.log" 2>&1 || { tail -30 "$M/logs/cmake.log"; exit 1; }
cmake --build "$BUILD" -j 32 --target mlipcpp-train mlipcpp-jvp-check mlipcpp-train-check 2>&1 | grep -E "error|Built target mlipcpp" || true
ls -la "$BUILD/bin/" | grep mlipcpp
du -sh "$M/data"/* 2>/dev/null
echo SETUP_DONE
