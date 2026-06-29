# Augmented Couinaud 8-segment finish workflow on ih-condor

This workflow finishes `Dataset102_LiverSegmentsAug` entirely on IH. It waits
through Slurm dependencies, then exports validation metrics, generates a full
batch of predictions, and computes volumetry from those predictions.

The `ih-condor` login node must only submit jobs and run lightweight checks. All
model evaluation, prediction, and volumetry runs through `sbatch`.

## 1. Move to the project on IH

```bash
ssh <user>@ih-condor
cd /home/mvaras/complete_mri_liver_segmentation_model
```

The scripts default to the server layout used by the previous jobs:

```text
project/code:      /home/mvaras/complete_mri_liver_segmentation_model
dataset base:      /mnt/researchers/julio-sotelo/datasets/mvarasr
nnUNet_raw:        /mnt/researchers/julio-sotelo/datasets/mvarasr/nnUNet_raw
nnUNet_preprocessed:
                   /mnt/researchers/julio-sotelo/datasets/mvarasr/nnUNet_preprocessed
nnUNet_results:    /mnt/researchers/julio-sotelo/datasets/mvarasr/nnUNet_results
dataset:           Dataset102_LiverSegmentsAug
trainer:           nnUNetTrainer_250epochs
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

## 3. Recover a fold that failed during final validation

If training finished but nnU-Net failed with a missing `gt_segmentations`
directory, submit a validation-only recovery job:

```bash
cd /home/mvaras/complete_mri_liver_segmentation_model
FOLD=3 sbatch scripts/validate_couinaud_fold_condor.sh
```

If the checkpoint was not completed and training must continue:

```bash
cd /home/mvaras/complete_mri_liver_segmentation_model
FOLD=3 RESUME_TRAINING=1 sbatch scripts/validate_couinaud_fold_condor.sh
```

## 4. Common overrides

Use a different conda environment:

```bash
CONDA_ENV=mariano bash scripts/submit_aug_seg_8_finish_condor.sh
```

Use a different dataset/model base:

```bash
BASE=/mnt/researchers/julio-sotelo/datasets/mvarasr \
bash scripts/submit_aug_seg_8_finish_condor.sh
```

Run only specific folds:

```bash
FOLDS="3" bash scripts/submit_aug_seg_8_finish_condor.sh
```

If `FOLDS` is not set, the job auto-detects folds that already have
`checkpoint_final.pth`.

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

## 5. Outputs

Each run writes:

```text
nnunet_protocol_results_aug_seg_8.csv
volumetry_results_all_cases.csv
volumetry_summary.csv
run_manifest.txt
```

Prediction masks are written inside:

```text
/mnt/researchers/julio-sotelo/datasets/mvarasr/nnUNet_raw/Dataset102_LiverSegmentsAug/predictions
```
