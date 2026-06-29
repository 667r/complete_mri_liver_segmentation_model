#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export nnUNet_raw="${nnUNet_raw:-$REPO_ROOT/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$REPO_ROOT/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$REPO_ROOT/models/nnUNet_results_seg_8}"

DATASET_ID="${DATASET_ID:-2}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer}"
FOLDS="${FOLDS:-0 1 2 3 4}"

nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity

for FOLD in $FOLDS; do
  nnUNetv2_train "$DATASET_ID" "$CONFIGURATION" "$FOLD" -tr "$TRAINER"
done
