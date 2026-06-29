# Dataset Access and Preparation

This repository does not redistribute medical imaging datasets. Only download and preparation instructions are provided.

## Duke Liver Dataset

The Duke Liver Dataset is publicly available through Zenodo:

<https://zenodo.org/records/7774566>

Download the dataset from Zenodo after reviewing the dataset terms. The included files `SegmentationKey.csv` and `SequenceTypes.csv` help identify the image series and contrast labels used for liver segmentation experiments.

Prepare the whole-liver dataset in nnU-Net v2 format:

```text
data/nnUNet_raw/Dataset001_Liver/
├── dataset.json
├── imagesTr/
├── labelsTr/
└── imagesTs/
```

Convert DICOM series and masks to NIfTI while preserving spacing, orientation, and affine metadata.

## Private Couinaud Dataset

The private Couinaud dataset contains segment-level liver annotations and cannot be publicly shared. Access requires the appropriate institutional permissions, ethics approval, and data-use agreements.

Prepare the Couinaud dataset in nnU-Net v2 format:

```text
data/nnUNet_raw/Dataset002_LiverSegments/
├── dataset.json
├── imagesTr/
├── labelsTr/
└── imagesTs/
```

Recommended label convention:

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

Keep all patient identifiers out of committed metadata files.
