#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/predict_couinaud.sh <input_images_dir> <output_predictions_dir>" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export nnUNet_results="${nnUNet_results:-$REPO_ROOT/models/nnUNet_results_seg_8}"

INPUT_DIR="$1"
OUTPUT_DIR="$2"
DATASET_ID="${DATASET_ID:-2}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer}"

nnUNetv2_predict -i "$INPUT_DIR" -o "$OUTPUT_DIR" -d "$DATASET_ID" -c "$CONFIGURATION" -tr "$TRAINER"
