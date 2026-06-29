#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/predict_couinaud.sh <input_images_dir> <output_predictions_dir>" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${BASE:-/mnt/researchers/julio-sotelo/datasets/mvarasr}"
export nnUNet_results="${nnUNet_results:-$BASE/nnUNet_results}"

INPUT_DIR="$1"
OUTPUT_DIR="$2"
DATASET_ID="${DATASET_ID:-102}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer_250epochs}"

nnUNetv2_predict -i "$INPUT_DIR" -o "$OUTPUT_DIR" -d "$DATASET_ID" -c "$CONFIGURATION" -tr "$TRAINER"
