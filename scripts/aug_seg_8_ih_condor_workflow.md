# Augmented Couinaud 8-segment finish workflow on ih-condor

This workflow finishes `Dataset102_LiverSegmentsAug` entirely on IH. It waits
through Slurm dependencies, then exports validation metrics, generates a full
batch of predictions, and computes volumetry from those predictions.

The `ih-condor` login node must only submit jobs and run lightweight checks. All
model evaluation, prediction, and volumetry runs through `sbatch`.

## 1. Move to the project on IH

```bash
ssh <user>@ih-condor
cd /path/to/complete_mri_liver_segmentation_model
```

The scripts default to paths relative to this project:

```text
nnUNet_raw:        $PROJECT_DIR/nnUNet_raw
nnUNet_results:    $PROJECT_DIR/models/nnUNet_results_aug_seg_8
dataset:           Dataset102_LiverSegmentsAug
```

If the model folder uses a custom trainer name, for example
`nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres`, the job auto-detects it
when it is the only matching folder for the dataset/configuration.

Outputs are written under:

```text
/mnt/workspace/$USER/aug_seg_8_finish_outputs/<run_id>
```

## 2. Submit after the training job

If the augmented 8-segment training is still running, submit the finish workflow
with the training job ID or IDs:

```bash
TRAINING_JOB_IDS="123456 123457 123458 123459 123460" \
  bash scripts/submit_aug_seg_8_finish_condor.sh
```

This uses `afterok`, so the finish job starts only if training completes
successfully.

If training is already complete:

```bash
bash scripts/submit_aug_seg_8_finish_condor.sh
```

The job checks for `checkpoint_final.pth` before doing any heavy work.

## 3. Common overrides

Use a different conda environment:

```bash
CONDA_ENV=mariano bash scripts/submit_aug_seg_8_finish_condor.sh
```

Run only specific folds:

```bash
FOLDS="1 2" bash scripts/submit_aug_seg_8_finish_condor.sh
```

Force a specific model folder if auto-detection is not enough:

```bash
TRAINER_CONFIG_DIR=nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres \
bash scripts/submit_aug_seg_8_finish_condor.sh
```

Use existing predictions and only run metrics plus volumetry:

```bash
RUN_PREDICTION=0 \
PREDICTIONS_DIR_NAME=predictions \
bash scripts/submit_aug_seg_8_finish_condor.sh
```

Write outputs to a chosen directory:

```bash
OUTPUT_DIR=/mnt/workspace/$USER/aug_seg_8_finish_outputs/manual_run \
bash scripts/submit_aug_seg_8_finish_condor.sh
```

## 4. Outputs

Each run writes:

```text
nnunet_protocol_results_aug_seg_8.csv
volumetry_results_all_cases.csv
volumetry_summary.csv
run_manifest.txt
```

Prediction masks are written inside:

```text
nnUNet_raw/Dataset102_LiverSegmentsAug/predictions_<run_id>
```
