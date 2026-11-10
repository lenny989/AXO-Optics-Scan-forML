#!/bin/bash
# Second batch: 24 more runs (100 epochs each) to complete the picture before the 500-epoch runs.
# Learning rate 1e-3 unless an entry says otherwise. Reference run: lr_1e-3.
#
#   sbatch overnight2.sh
#
# At most 3 tasks run at the same time (%3), so the GPUs are shared fairly.
# Afterwards:  python plot_groups.py

#SBATCH --job-name=exp2
#SBATCH --chdir=/vol/fob-vol7/mi26/vitaleon/Dokumente/AXO-Optics-Scan-forML
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-23%3

set -euo pipefail

PROJECT=/vol/fob-vol7/mi26/vitaleon/Dokumente/AXO-Optics-Scan-forML
cd "$PROJECT"

set +u; source ~/venvs/xrd/bin/activate; set -u

[ -f main.py ] || { echo "ERROR: main.py not found in $PROJECT"; exit 1; }
[ -f cache/xrd_128_images.npy ] || { echo "ERROR: cache/xrd_128_images.npy not found in $PROJECT"; exit 1; }

LR=1e-3        # default learning rate (an entry below can override it with its own --lr)

# name | extra options. Options at the end win over the defaults above.
EXPS=(
  # --- batch norm OFF: learning-rate curve (1e-3 already exists as exp_nobn) ---
  "nobn_1e-5|--no-batchnorm --lr 1e-5"
  "nobn_1e-4|--no-batchnorm --lr 1e-4"
  "nobn_3e-4|--no-batchnorm --lr 3e-4"
  "nobn_3e-3|--no-batchnorm --lr 3e-3"
  # --- regularization ---
  "do01|--dropout 0.1"
  # --- architecture ---
  "mid|--hidden 512 256 128"
  "xwide|--hidden 4096 2048 512"
  # --- augmentation strength (threshold that finds the signal; higher = larger shifts) ---
  "sf01|--signal-frac 0.1"
  "sf06|--signal-frac 0.6"
  # --- activation / loss ---
  "tanh|--activation tanh"
  "mse|--loss mse"
  # --- training setup ---
  "bs128|--batch-size 128"
  "bs512|--batch-size 512"
  "nosched|--scheduler none"
  # --- combinations of the settings that looked best ---
  "do0_wide|--dropout 0 --hidden 2048 1024 256"
  "do01_wide|--dropout 0.1 --hidden 2048 1024 256"
  # --- seeds 1 and 2 (seed 0 already exists) for 4 configurations ---
  "base_s1|--seed 1"
  "base_s2|--seed 2"
  "do0_s1|--dropout 0 --seed 1"
  "do0_s2|--dropout 0 --seed 2"
  "wide_s1|--hidden 2048 1024 256 --seed 1"
  "wide_s2|--hidden 2048 1024 256 --seed 2"
  "do0_wide_s1|--dropout 0 --hidden 2048 1024 256 --seed 1"
  "do0_wide_s2|--dropout 0 --hidden 2048 1024 256 --seed 2"
)
EXP="${EXPS[$SLURM_ARRAY_TASK_ID]}"
NAME="${EXP%%|*}"
ARGS="${EXP#*|}"
echo "task $SLURM_ARRAY_TASK_ID: exp_$NAME  (lr=$LR, options: $ARGS)  on $(hostname)"

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv   # the card this task got

# Try up to 3 times: a GPU that is full when the job starts can free up a few minutes later.
# A retry continues from last.pt if the run got that far.
OK=0
for attempt in 1 2 3; do
  RESUME=""
  if [ -f "runs/exp_$NAME/last.pt" ]; then RESUME="--resume"; fi
  if python -u main.py \
      --cache cache/xrd_128 \
      --val-scans \
        Mainz_20220912_02_AXO-Optics-Scan-forML_245W_Al-tape_noFlTu_SourceDet-130cm_detPos1 \
        Mainz_20220914_01_AXO-Optics-Scan-forML_245W_Al-tape_noFlTu_SourceDet-160cm_detPos1 \
      --epochs 100 \
      --lr "$LR" \
      --workers $((SLURM_CPUS_PER_TASK - 1)) \
      --name "exp_$NAME" $ARGS $RESUME; then
    OK=1
    break
  fi
  echo "attempt $attempt failed, waiting 2 minutes"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
  sleep 120
done
[ "$OK" = 1 ] || { echo "ERROR: all 3 attempts failed"; exit 1; }
