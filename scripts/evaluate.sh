#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/evaluate.sh <predictions_dir> <reference_labels_dir> <output_dir> [dataset_json] [plans_json]" >&2
  exit 1
fi

PREDICTIONS_DIR="$1"
REFERENCE_DIR="$2"
OUTPUT_DIR="$3"
DATASET_JSON="${4:-${DATASET_JSON:-}}"
PLANS_JSON="${5:-${PLANS_JSON:-}}"

if [[ -z "$DATASET_JSON" || -z "$PLANS_JSON" ]]; then
  echo "Provide dataset_json and plans_json as arguments or DATASET_JSON/PLANS_JSON environment variables." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
nnUNetv2_evaluate_folder "$REFERENCE_DIR" "$PREDICTIONS_DIR" -djfile "$DATASET_JSON" -pfile "$PLANS_JSON" -o "$OUTPUT_DIR/summary.json"
