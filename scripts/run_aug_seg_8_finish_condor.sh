#!/bin/bash
#SBATCH --job-name=aug-seg8-finish
#SBATCH --time=1-00:00:00
#SBATCH --partition=batch
#SBATCH --qos=batch
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=/mnt/workspace/%u/slurm-out/%j.out
#SBATCH --error=/mnt/workspace/%u/slurm-out/%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET_ID="${DATASET_ID:-102}"
DATASET_NAME="${DATASET_NAME:-Dataset102_LiverSegmentsAug}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer}"
PLANS="${PLANS:-nnUNetPlans}"
TRAINER_CONFIG_DIR="${TRAINER_CONFIG_DIR:-${TRAINER}__${PLANS}__${CONFIGURATION}}"

NNUNET_RAW="${NNUNET_RAW:-$PROJECT_DIR/nnUNet_raw}"
NNUNET_PREPROCESSED="${NNUNET_PREPROCESSED:-$PROJECT_DIR/nnUNet_preprocessed}"
NNUNET_RESULTS="${NNUNET_RESULTS:-$PROJECT_DIR/models/nnUNet_results_aug_seg_8}"
if [[ -z "${MODEL_DIR:-}" ]]; then
  MODEL_DIR="$NNUNET_RESULTS/$DATASET_NAME/$TRAINER_CONFIG_DIR"
  if [[ ! -d "$MODEL_DIR" ]]; then
    shopt -s nullglob
    model_candidates=("$NNUNET_RESULTS/$DATASET_NAME/"*"__${PLANS}__${CONFIGURATION}")
    shopt -u nullglob
    if [[ "${#model_candidates[@]}" -eq 1 ]]; then
      MODEL_DIR="${model_candidates[0]}"
      TRAINER_CONFIG_DIR="$(basename "$MODEL_DIR")"
      TRAINER="${TRAINER_CONFIG_DIR%%__*}"
    fi
  fi
fi

IMAGES_DIR="${IMAGES_DIR:-$NNUNET_RAW/$DATASET_NAME/imagesTr}"
LABELS_DIR="${LABELS_DIR:-$NNUNET_RAW/$DATASET_NAME/labelsTr}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PREDICTIONS_DIR_NAME="${PREDICTIONS_DIR_NAME:-predictions_$RUN_ID}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-$NNUNET_RAW/$DATASET_NAME/$PREDICTIONS_DIR_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/workspace/$USER/aug_seg_8_finish_outputs/$RUN_ID}"

EVAL_FOLDS="${EVAL_FOLDS:-${FOLDS:-0 3}}"
PREDICT_FOLDS="${PREDICT_FOLDS:-${FOLDS:-0 3}}"
PREDICT_CHECKPOINT="${PREDICT_CHECKPOINT:-checkpoint_final.pth}"

RUN_EVALUATION="${RUN_EVALUATION:-1}"
RUN_EXPORT="${RUN_EXPORT:-1}"
RUN_PREDICTION="${RUN_PREDICTION:-1}"
RUN_VOLUMETRY="${RUN_VOLUMETRY:-1}"
FORCE_EVALUATION="${FORCE_EVALUATION:-0}"
FAIL_ON_MISSING_CHECKPOINTS="${FAIL_ON_MISSING_CHECKPOINTS:-1}"
CONDA_ENV="${CONDA_ENV:-mariano}"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export nnUNet_raw="$NNUNET_RAW"
export nnUNet_preprocessed="$NNUNET_PREPROCESSED"
export nnUNet_results="$NNUNET_RESULTS"

mkdir -p "$OUTPUT_DIR"

log "Project dir: $PROJECT_DIR"
log "Dataset: $DATASET_NAME ($DATASET_ID)"
log "Model dir: $MODEL_DIR"
log "nnUNet_raw: $nnUNet_raw"
log "nnUNet_preprocessed: $nnUNet_preprocessed"
log "nnUNet_results: $nnUNet_results"
log "Output dir: $OUTPUT_DIR"

required_paths=(
  "$PROJECT_DIR/src/export_nnunet_protocol_results.py"
  "$PROJECT_DIR/src/get_volumetry_mL.py"
  "$NNUNET_RAW/$DATASET_NAME/dataset.json"
  "$MODEL_DIR/dataset.json"
  "$MODEL_DIR/plans.json"
  "$IMAGES_DIR"
  "$LABELS_DIR"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required path not found: $path" >&2
    exit 1
  fi
done

read -r -a EVAL_FOLD_ARGS <<< "$EVAL_FOLDS"
read -r -a PREDICT_FOLD_ARGS <<< "$PREDICT_FOLDS"

missing_checkpoints=0
for fold in "${EVAL_FOLD_ARGS[@]}"; do
  fold_dir="$MODEL_DIR/fold_$fold"
  checkpoint="$fold_dir/$PREDICT_CHECKPOINT"
  if [[ ! -f "$checkpoint" ]]; then
    log "Missing checkpoint for fold_$fold: $checkpoint"
    missing_checkpoints=$((missing_checkpoints + 1))
  fi
done

if [[ "$missing_checkpoints" -gt 0 && "$FAIL_ON_MISSING_CHECKPOINTS" == "1" ]]; then
  echo "ERROR: $missing_checkpoints expected checkpoints are missing. Submit this job with an afterok dependency or adjust EVAL_FOLDS/FOLDS." >&2
  exit 1
fi

if [[ "$RUN_EVALUATION" == "1" ]]; then
  for fold in "${EVAL_FOLD_ARGS[@]}"; do
    validation_dir="$MODEL_DIR/fold_$fold/validation"
    summary_json="$validation_dir/summary.json"

    if [[ ! -d "$validation_dir" ]]; then
      echo "ERROR: missing validation directory for fold_$fold: $validation_dir" >&2
      exit 1
    fi

    if ! find "$validation_dir" -maxdepth 1 -name "*.nii*" -print -quit | grep -q .; then
      echo "ERROR: no validation predictions found for fold_$fold in $validation_dir" >&2
      exit 1
    fi

    if [[ -f "$summary_json" && "$FORCE_EVALUATION" != "1" ]]; then
      log "Keeping existing summary for fold_$fold: $summary_json"
      continue
    fi

    log "Evaluating fold_$fold validation predictions"
    nnUNetv2_evaluate_folder \
      "$LABELS_DIR" \
      "$validation_dir" \
      -djfile "$MODEL_DIR/dataset.json" \
      -pfile "$MODEL_DIR/plans.json" \
      -o "$summary_json"
  done
fi

if [[ "$RUN_EXPORT" == "1" ]]; then
  metrics_csv="$OUTPUT_DIR/nnunet_protocol_results_aug_seg_8.csv"
  log "Exporting protocol metrics to $metrics_csv"
  python3 "$PROJECT_DIR/src/export_nnunet_protocol_results.py" \
    --input "$NNUNET_RESULTS" \
    --output-file "$metrics_csv" \
    --segmentation-key "$PROJECT_DIR/data/SegmentationKey.csv" \
    --sequence-types "$PROJECT_DIR/data/SequenceTypes.csv" \
    --nnunet-raw-root "$NNUNET_RAW"

  metric_rows="$(($(wc -l < "$metrics_csv") - 1))"
  if [[ "$metric_rows" -le 0 ]]; then
    echo "ERROR: metrics export produced no data rows: $metrics_csv" >&2
    exit 1
  fi
  log "Exported $metric_rows metric rows"
fi

if [[ "$RUN_PREDICTION" == "1" ]]; then
  log "Writing batch predictions to $PREDICTIONS_DIR"
  mkdir -p "$PREDICTIONS_DIR"
  nnUNetv2_predict \
    -i "$IMAGES_DIR" \
    -o "$PREDICTIONS_DIR" \
    -d "$DATASET_ID" \
    -c "$CONFIGURATION" \
    -tr "$TRAINER" \
    -p "$PLANS" \
    -f "${PREDICT_FOLD_ARGS[@]}" \
    -chk "$PREDICT_CHECKPOINT"
else
  log "Skipping batch prediction; using existing predictions in $PREDICTIONS_DIR"
fi

if [[ "$RUN_VOLUMETRY" == "1" ]]; then
  dataset_dir="$NNUNET_RAW/$DATASET_NAME"
  if [[ "$(cd "$(dirname "$PREDICTIONS_DIR")" && pwd)" != "$(cd "$dataset_dir" && pwd)" ]]; then
    echo "ERROR: PREDICTIONS_DIR must be directly inside $dataset_dir for volumetry. Got: $PREDICTIONS_DIR" >&2
    exit 1
  fi

  log "Computing volumetry"
  python3 "$PROJECT_DIR/src/get_volumetry_mL.py" \
    --nnunet-raw "$NNUNET_RAW" \
    --output-dir "$OUTPUT_DIR" \
    --predictions-dir-name "$(basename "$PREDICTIONS_DIR")" \
    --datasets "$DATASET_NAME" \
    --fail-on-missing-predictions
fi

cat > "$OUTPUT_DIR/run_manifest.txt" <<EOF
run_id=$RUN_ID
project_dir=$PROJECT_DIR
dataset_id=$DATASET_ID
dataset_name=$DATASET_NAME
model_dir=$MODEL_DIR
eval_folds=$EVAL_FOLDS
predict_folds=$PREDICT_FOLDS
predictions_dir=$PREDICTIONS_DIR
output_dir=$OUTPUT_DIR
conda_env=$CONDA_ENV
EOF

log "Done. Outputs are in $OUTPUT_DIR"
