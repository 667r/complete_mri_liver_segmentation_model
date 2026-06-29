#!/bin/bash
#SBATCH --job-name=volumetry
#SBATCH --time=0-02:00:00
#SBATCH --partition=batch
#SBATCH --qos=batch
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/mnt/workspace/%u/slurm-out/%j.out
#SBATCH --error=/mnt/workspace/%u/slurm-out/%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-mariano}"

BASE="${BASE:-/mnt/researchers/julio-sotelo/datasets/mvarasr}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VOLUMETRY_SCRIPT="${VOLUMETRY_SCRIPT:-/home/mvarasr/get_volumetry_mL.py}"
NNUNET_RAW="${NNUNET_RAW:-$BASE/nnUNet_raw}"
OUTPUT_DIR="${VOLUMETRY_OUTPUT_DIR:-/mnt/workspace/$USER/volumetry_outputs/$(date +%Y%m%d_%H%M%S)}"
PREDICTIONS_DIR_NAME="${PREDICTIONS_DIR_NAME:-predictions}"

mkdir -p "$OUTPUT_DIR"

if [[ -n "${DATASETS:-}" ]]; then
  read -r -a DATASET_ARGS <<< "$DATASETS"
else
  # Dataset002_LiverSegments is intentionally omitted because it did not produce predictions.
  DATASET_ARGS=(
    Dataset001_Liver
    Dataset101_LiverAug
    Dataset102_LiverSegmentsAug
  )
fi

echo "Project dir: $PROJECT_DIR"
echo "Volumetry script: $VOLUMETRY_SCRIPT"
echo "nnUNet_raw: $NNUNET_RAW"
echo "Output dir: $OUTPUT_DIR"
echo "Predictions dir name: $PREDICTIONS_DIR_NAME"
echo "Datasets: ${DATASET_ARGS[*]}"

if [[ ! -f "$VOLUMETRY_SCRIPT" ]]; then
  echo "ERROR: volumetry script not found: $VOLUMETRY_SCRIPT"
  exit 1
fi

python3 "$VOLUMETRY_SCRIPT" \
  --nnunet-raw "$NNUNET_RAW" \
  --output-dir "$OUTPUT_DIR" \
  --predictions-dir-name "$PREDICTIONS_DIR_NAME" \
  --datasets "${DATASET_ARGS[@]}"
