# Volumetry workflow on ih-condor

This workflow computes mL volumetry from nnU-Net predictions stored inside each
dataset folder under `nnUNet_raw`.

## 1. Log in and move to the project

```bash
ssh <user>@ih-condor
cd /mnt/researchers/julio-sotelo/datasets/mvarasr
```

The Slurm wrapper calls the volumetry script at:

```text
/home/mvarasr/get_volumetry_mL.py
```

That file must be the argument-based version that accepts `--nnunet-raw`,
`--output-dir`, `--predictions-dir-name`, and `--datasets`.

## 2. Confirm predictions exist

Run only lightweight shell checks on the login node:

```bash
for d in Dataset001_Liver Dataset101_LiverAug Dataset102_LiverSegmentsAug; do
  echo "$d"
  find "nnUNet_raw/$d/predictions" -maxdepth 1 -name '*.nii*' 2>/dev/null | wc -l
done
```

`Dataset002_LiverSegments` is intentionally not included in the default job
because it did not produce predictions.

## 3. Submit the Slurm job

```bash
mkdir -p /mnt/workspace/$USER/slurm-out
sbatch scripts/run_volumetry_condor.sh
```

The default output directory is:

```text
/mnt/workspace/$USER/volumetry_outputs/YYYYMMDD_HHMMSS
```

## 4. Optional overrides

Use a different conda env:

```bash
CONDA_ENV=mariano sbatch scripts/run_volumetry_condor.sh
```

Use a different dataset base:

```bash
BASE=/mnt/researchers/julio-sotelo/datasets/mvarasr sbatch scripts/run_volumetry_condor.sh
```

Use a different project path:

```bash
PROJECT_DIR=/path/to/complete_mri_liver_segmentation_model sbatch scripts/run_volumetry_condor.sh
```

Use a different volumetry script path:

```bash
VOLUMETRY_SCRIPT=/home/mvarasr/get_volumetry_mL.py sbatch scripts/run_volumetry_condor.sh
```

Run only specific datasets:

```bash
DATASETS="Dataset001_Liver Dataset101_LiverAug" sbatch scripts/run_volumetry_condor.sh
```

## 5. Outputs

The job writes:

```text
volumetry_results_all_cases.csv
volumetry_summary.csv
```

The first file has one row per case and label. The second file summarizes mean
and standard deviation by dataset, task, and label.
