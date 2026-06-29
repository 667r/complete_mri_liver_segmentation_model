# Trained Models

Place trained nnU-Net v2 model folders here.

Expected layout:

```text
models/
├── nnUNet_results_seg/
│   └── Dataset001_Liver/
└── nnUNet_results_seg_8/
    └── Dataset002_LiverSegments/
```

These directories are the nnU-Net results roots used by the helper scripts. Do not modify checkpoint files, plans files, folds, or trainer metadata after training. If the model folders are supplied outside Git, copy or move them into this directory exactly as shown.

For whole-liver inference:

```bash
export nnUNet_results="$PWD/models/nnUNet_results_seg"
```

For Couinaud inference:

```bash
export nnUNet_results="$PWD/models/nnUNet_results_seg_8"
```
