#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/mnt/researchers/julio-sotelo/datasets/mvarasr}"
DATASET_NAME="${DATASET_NAME:-Dataset102_LiverSegmentsAug}"
NNUNET_RAW="${NNUNET_RAW:-$BASE/nnUNet_raw}"
NNUNET_PREPROCESSED="${NNUNET_PREPROCESSED:-$BASE/nnUNet_preprocessed}"

LABELS_DIR="$NNUNET_RAW/$DATASET_NAME/labelsTr"
GT_DIR="$NNUNET_PREPROCESSED/$DATASET_NAME/gt_segmentations"

if [[ ! -d "$LABELS_DIR" ]]; then
  echo "ERROR: labelsTr not found: $LABELS_DIR" >&2
  exit 1
fi

mkdir -p "$GT_DIR"

copied=0
for label_file in "$LABELS_DIR"/*.nii "$LABELS_DIR"/*.nii.gz; do
  [[ -e "$label_file" ]] || continue
  target="$GT_DIR/$(basename "$label_file")"
  if [[ ! -e "$target" ]]; then
    cp "$label_file" "$target"
    copied=$((copied + 1))
  fi
done

label_count="$(find "$LABELS_DIR" -maxdepth 1 \( -name '*.nii' -o -name '*.nii.gz' \) | wc -l)"
gt_count="$(find "$GT_DIR" -maxdepth 1 \( -name '*.nii' -o -name '*.nii.gz' \) | wc -l)"

echo "Dataset: $DATASET_NAME"
echo "labelsTr: $LABELS_DIR ($label_count files)"
echo "gt_segmentations: $GT_DIR ($gt_count files, copied $copied)"

if [[ "$gt_count" -eq 0 ]]; then
  echo "ERROR: no gt segmentations were created." >&2
  exit 1
fi
