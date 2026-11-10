#!/bin/bash
# 5 networks, 100 epochs each, one learning rate per job (Slurm job array).
#
#   mkdir -p ~/Dokumente/AXO-Optics-Scan-forML/logs
#   sbatch lr_sweep.sh          # works from any folder
#
# Slurm starts 5 jobs (numbers 0-4), each gets its own GPU.

#SBATCH --job-name=lr
#SBATCH --chdir=/vol/fob-vol7/mi26/vitaleon/Dokumente/AXO-Optics-Scan-forML
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-4

set -euo pipefail

PROJECT=/vol/fob-vol7/mi26/vitaleon/Dokumente/AXO-Optics-Scan-forML
cd "$PROJECT"

set +u; source ~/venvs/xrd/bin/activate; set -u      # activate scripts can trip over 'set -u'

[ -f main.py ] || { echo "ERROR: main.py not found in $PROJECT"; exit 1; }
[ -f cache/xrd_128_images.npy ] || { echo "ERROR: cache/xrd_128_images.npy not found in $PROJECT"; exit 1; }

LRS=(1e-4 3e-4 1e-3 3e-3 1e-2)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID: lr=$LR on $(hostname)"

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv   # the card this task got

# Try up to 3 times: a GPU that is full when the job starts can free up a few minutes later.
# A retry continues from last.pt if the run got that far.
OK=0
for attempt in 1 2 3; do
  RESUME=""
  if [ -f "runs/lr_$LR/last.pt" ]; then RESUME="--resume"; fi
  if python -u main.py \
      --cache cache/xrd_128 \
      --val-scans \
        Mainz_20220912_02_AXO-Optics-Scan-forML_245W_Al-tape_noFlTu_SourceDet-130cm_detPos1 \
        Mainz_20220914_01_AXO-Optics-Scan-forML_245W_Al-tape_noFlTu_SourceDet-160cm_detPos1 \
      --epochs 100 \
      --lr "$LR" \
      --workers $((SLURM_CPUS_PER_TASK - 1)) \
      --name "lr_$LR" $RESUME; then
    OK=1
    break
  fi
  echo "attempt $attempt failed, waiting 2 minutes"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
  sleep 120
done
[ "$OK" = 1 ] || { echo "ERROR: all 3 attempts failed"; exit 1; }
