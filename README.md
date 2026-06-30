# MRI Liver Segmentation with nnU-Net v2

This repository contains the final implementation of an academic MRI liver segmentation project based on nnU-Net v2. It supports whole-liver segmentation and Couinaud segment segmentation, with reproducible instructions for dataset preparation, model training, inference, evaluation, and volumetry.

The project is organized for publication and review. Public dataset instructions are provided, but medical image data and private Couinaud annotations are not redistributed in this repository.

## Repository Structure

```text
/home/mvarasr/complete_mri_liver_segmentation_model/
|-- README.md
|-- requirements.txt
|-- src/
|   |-- prepare_nnunet_dataset.py
|   |-- augment_nnunet_dataset.py
|   |-- export_nnunet_protocol_results.py
|   |-- get_volumetry_mL.py
|   `-- compute_volumetry.py
|-- scripts/
|   |-- train_couinaud.sh
|   |-- predict_couinaud.sh
|   |-- validate_couinaud_fold_condor.sh
|   |-- run_aug_seg_8_finish_condor.sh
|   |-- submit_aug_seg_8_finish_condor.sh
|   `-- run_volumetry_condor.sh
|-- data/
|   |-- SegmentationKey.csv
|   |-- SequenceTypes.csv
|   `-- README.md
|-- models/
|   `-- README.md
`-- results/
    `-- README.md
```


local nnU-Net model layout (not on the repository for file size issues):

```text
models/
├── nnUNet_results_seg/
│   └── Dataset001_Liver/
└── nnUNet_results_seg_8/
    └── Dataset002_LiverSegments/
```

The trained model directories must remain unchanged. If they are supplied separately, place or move them into `models/` exactly as shown above.

## System Requirements

- Python 3.10 or newer
- Linux, macOS, or Windows with PowerShell/Git Bash
- NVIDIA GPU with CUDA is strongly recommended for training 3D nnU-Net models
- Sufficient disk space for raw data, preprocessed nnU-Net data, model checkpoints, and predictions
- Git and a POSIX-compatible shell for the helper scripts

## Installation

Create and activate a clean Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch for your hardware first by following the official selector at <https://pytorch.org/get-started/locally/>.

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

## Python Dependencies

The core dependencies are listed in `requirements.txt`:

- `nnunetv2`
- `torch` and related PyTorch packages, installed with the build appropriate for the local GPU/CPU environment
- `numpy`, `pandas`, `scipy`
- `nibabel`, `SimpleITK`
- `medpy`, `scikit-image`
- `matplotlib`, `seaborn`
- `tqdm`, `pyyaml`

## nnU-Net Installation

For standard use:

```bash
pip install nnunetv2
```

For local development of nnU-Net itself:

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

nnU-Net v2 requires three environment variables:

```bash
export nnUNet_raw="$PWD/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnUNet_preprocessed"
export nnUNet_results="$PWD/models/nnUNet_results_seg"
```

For Couinaud segment models, point `nnUNet_results` to the corresponding results folder:

```bash
export nnUNet_results="$PWD/models/nnUNet_results_seg_8"
```

On Windows PowerShell:

```powershell
$Env:nnUNet_raw = "$PWD/nnUNet_raw"
$Env:nnUNet_preprocessed = "$PWD/nnUNet_preprocessed"
$Env:nnUNet_results = "$PWD/models/nnUNet_results_seg"
```

## Dataset Preparation

nnU-Net expects each dataset in `nnUNet_raw` to use the standard v2 format:

```text
nnUNet_raw/
└── Dataset001_Liver/
    ├── dataset.json
    ├── imagesTr/
    ├── labelsTr/
    └── imagesTs/
```

Whole-liver segmentation should use:

```text
nnUNet_raw/Dataset001_Liver/
```

Couinaud segmentation should use:

```text
nnUNet_raw/Dataset002_LiverSegments/
```

Labels should preserve image spacing and orientation. Use nearest-neighbor interpolation for masks and image interpolation appropriate for MRI volumes during any conversion or resampling.

## Duke Liver Dataset

The Duke Liver Dataset is a public MRI dataset with liver segmentation masks and series labels. Download it from Zenodo:

- <https://zenodo.org/records/7774566>

This repository includes `data/SegmentationKey.csv` and `data/SequenceTypes.csv` to help identify the relevant image series. The dataset itself is not redistributed here. After obtaining access and downloading the data, convert the selected DICOM series and masks into nnU-Net-compatible NIfTI files under `nnUNet_raw/Dataset001_Liver/`.

## Private Couinaud Dataset

The private Couinaud dataset contains segment-level liver annotations and cannot be publicly shared. Researchers reproducing this work must obtain the private dataset through the appropriate institutional, ethics, and data-use approvals.

Prepare it as:

```text
nnUNet_raw/Dataset002_LiverSegments/
├── dataset.json
├── imagesTr/
├── labelsTr/
└── imagesTs/
```

Use the following label convention unless the trained model metadata specifies otherwise:

```text
0 background
1 Couinaud I
2 Couinaud II
3 Couinaud III
4 Couinaud IV
5 Couinaud V
6 Couinaud VI
7 Couinaud VII
8 Couinaud VIII
```

## Training

Whole-liver training:

```bash
bash scripts/train_whole_liver.sh
```

Couinaud segment training:

```bash
bash scripts/train_couinaud.sh
```

The scripts call nnU-Net v2 planning, preprocessing, and fold training commands. They assume the nnU-Net raw datasets are already prepared.

## Inference

Whole-liver prediction:

```bash
bash scripts/predict_whole_liver.sh /path/to/imagesTs results/predictions/whole_liver
```

Couinaud prediction:

```bash
bash scripts/predict_couinaud.sh /path/to/imagesTs results/predictions/couinaud
```

The input directory must contain nnU-Net-formatted test images, for example `case_0000.nii.gz` for a single-channel MRI case.

## Evaluation

Run nnU-Net evaluation for a prediction folder and reference labels:

```bash
bash scripts/evaluate.sh \
  results/predictions/whole_liver \
  nnUNet_raw/Dataset001_Liver/labelsTs \
  results/evaluation/whole_liver \
  models/nnUNet_results_seg/Dataset001_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json \
  models/nnUNet_results_seg/Dataset001_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json
```

For academic reporting, compute and report metrics per dataset split and, where metadata are available, stratify by MRI contrast, scanner vendor, institution, and pathology group.

## Volumetry

Compute volumes from predicted or reference NIfTI masks:

```bash
python src/compute_volumetry.py \
  --masks results/predictions/couinaud \
  --labels "1:Couinaud I,2:Couinaud II,3:Couinaud III,4:Couinaud IV,5:Couinaud V,6:Couinaud VI,7:Couinaud VII,8:Couinaud VIII" \
  --output results/volumetry/couinaud_volumes.csv
```

For whole-liver masks:

```bash
python src/compute_volumetry.py \
  --masks results/predictions/whole_liver \
  --labels "1:Liver" \
  --output results/volumetry/whole_liver_volumes.csv
```

Volumes are computed in milliliters from the NIfTI voxel spacing.

## Results

Store publication artifacts in `results/` as csv files with data extracted from the `summary.json` files found in the
validation folder of each fold.

## Reproducibility Notes

- Keep train/validation/test splits at the patient level.
- Do not split different contrasts from the same patient across train and test.
- Preserve NIfTI affine, orientation, and spacing metadata.
- Record nnU-Net version, PyTorch version, CUDA version, GPU model, trainer,
  folds, and random seeds.
- Keep trained model folders intact after training.

## References

1. Macdonald JA, Zhu Z, Swensson J, et al. Duke Liver Dataset: A Publicly Available Liver MRI Dataset with Liver Segmentation Masks and Series Labels. *Radiology: Artificial Intelligence*. 2023. <https://doi.org/10.1148/ryai.220275>
2. Duke Liver Dataset (MRI) v2, Zenodo. <https://zenodo.org/records/7774566>
3. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*. 2021.
4. nnU-Net v2 documentation and source code. <https://github.com/MIC-DKFZ/nnUNet>
