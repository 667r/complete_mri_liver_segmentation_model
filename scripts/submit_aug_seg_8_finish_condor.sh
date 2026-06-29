#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SLURM_OUT_DIR="${SLURM_OUT_DIR:-/mnt/workspace/$USER/slurm-out}"
JOB_SCRIPT="$SCRIPT_DIR/run_aug_seg_8_finish_condor.sh"

mkdir -p "$SLURM_OUT_DIR"

dependency_ids=""
if [[ $# -gt 0 ]]; then
  dependency_ids="$*"
elif [[ -n "${TRAINING_JOB_IDS:-}" ]]; then
  dependency_ids="$TRAINING_JOB_IDS"
elif [[ -n "${TRAINING_JOB_ID:-}" ]]; then
  dependency_ids="$TRAINING_JOB_ID"
fi

dependency_args=()
if [[ -n "$dependency_ids" ]]; then
  normalized_ids="$(echo "$dependency_ids" | sed 's/[ ,][ ,]*/:/g; s/^:*//; s/:*$//')"
  dependency_args=(--dependency="afterok:$normalized_ids")
  echo "Submitting post-training workflow after training job(s): $normalized_ids"
else
  echo "Submitting post-training workflow without dependency. It will fail fast if checkpoints are missing."
fi

PROJECT_DIR="$PROJECT_DIR" sbatch "${dependency_args[@]}" "$JOB_SCRIPT"
