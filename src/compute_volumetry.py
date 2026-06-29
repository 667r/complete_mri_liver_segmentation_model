#!/usr/bin/env python
"""Compute label volumes from NIfTI segmentation masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def parse_labels(label_spec: str) -> dict[int, str]:
    """Parse labels formatted as '1:Liver,2:Couinaud II'."""
    labels: dict[int, str] = {}
    for item in label_spec.split(","):
        item = item.strip()
        if not item:
            continue
        value, name = item.split(":", 1)
        labels[int(value.strip())] = name.strip()
    return labels


def iter_nifti_files(mask_dir: Path) -> list[Path]:
    files = sorted(mask_dir.glob("*.nii")) + sorted(mask_dir.glob("*.nii.gz"))
    return [path for path in files if path.is_file()]


def compute_mask_volumes(mask_path: Path, labels: dict[int, str]) -> list[dict[str, object]]:
    image = nib.load(str(mask_path))
    mask = np.asarray(image.dataobj)
    voxel_volume_ml = float(np.prod(image.header.get_zooms()[:3])) / 1000.0

    rows: list[dict[str, object]] = []
    for label_value, label_name in labels.items():
        voxel_count = int(np.count_nonzero(mask == label_value))
        rows.append(
            {
                "case_id": mask_path.name.removesuffix(".nii.gz").removesuffix(".nii"),
                "label_value": label_value,
                "label_name": label_name,
                "voxel_count": voxel_count,
                "voxel_volume_ml": voxel_volume_ml,
                "volume_ml": voxel_count * voxel_volume_ml,
            }
        )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute per-label volumes in milliliters from NIfTI masks."
    )
    parser.add_argument("--masks", required=True, type=Path, help="Directory containing .nii or .nii.gz masks.")
    parser.add_argument(
        "--labels",
        required=True,
        help="Comma-separated label map such as '1:Liver' or '1:Couinaud I,2:Couinaud II'.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    labels = parse_labels(args.labels)
    mask_files = iter_nifti_files(args.masks)

    if not mask_files:
        raise FileNotFoundError(f"No NIfTI masks found in {args.masks}")

    # Compute one row per case and label so downstream statistics can group by either field.
    rows: list[dict[str, object]] = []
    for mask_path in mask_files:
        rows.extend(compute_mask_volumes(mask_path, labels))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
