from __future__ import annotations

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


DEFAULT_DATASETS = [
    "Dataset001_Liver",
    "Dataset002_LiverSegments",
    "Dataset101_LiverAug",
    "Dataset102_LiverSegmentsAug",
]


def get_case_id(path: Path) -> str:
    name = path.name
    return name.replace(".nii.gz", "").replace(".nii", "")


def list_nifti_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(set(path.glob("*.nii")) | set(path.glob("*.nii.gz")))


def compute_volumes(mask_path: Path, labels: list[int]) -> dict[int, float]:
    img = nib.load(str(mask_path))
    data = np.asanyarray(img.dataobj).astype(np.int16)

    spacing = img.header.get_zooms()[:3]  # mm
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]

    volumes = {}
    for label in labels:
        voxel_count = int(np.sum(data == label))
        volumes[label] = (voxel_count * voxel_volume_mm3) / 1000.0  # mL

    return volumes


def find_prediction(case_id: str, pred_dir: Path) -> Path | None:
    candidates = [
        pred_dir / f"{case_id}.nii.gz",
        pred_dir / f"{case_id}.nii",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def dataset_label_config(dataset_name: str) -> tuple[str, list[int]]:
    if "Segments" in dataset_name:
        return "Couinaud", list(range(1, 9))
    return "Whole liver", [1]


def collect_dataset_rows(
    nnunet_raw: Path,
    dataset_name: str,
    predictions_dir_name: str,
    fail_on_missing_predictions: bool,
) -> list[dict[str, object]]:
    dataset_dir = nnunet_raw / dataset_name
    labels_dir = dataset_dir / "labelsTr"
    pred_dir = dataset_dir / predictions_dir_name

    if not labels_dir.exists():
        print(f"[SKIP] Missing labelsTr: {labels_dir}")
        return []

    if not pred_dir.exists():
        print(f"[SKIP] Missing predictions dir: {pred_dir}")
        return []

    task_type, labels = dataset_label_config(dataset_name)
    gt_files = list_nifti_files(labels_dir)
    print(f"[INFO] {dataset_name}: labels={len(gt_files)} predictions_dir={pred_dir}")

    rows: list[dict[str, object]] = []
    missing_predictions: list[str] = []

    for gt_file in gt_files:
        case_id = get_case_id(gt_file)
        pred_file = find_prediction(case_id, pred_dir)

        if pred_file is None:
            missing_predictions.append(case_id)
            print(f"[WARN] Missing prediction for {dataset_name}/{case_id}")
            continue

        gt_vol = compute_volumes(gt_file, labels)
        pred_vol = compute_volumes(pred_file, labels)

        for label in labels:
            gt_ml = gt_vol[label]
            pred_ml = pred_vol[label]
            abs_error_ml = abs(pred_ml - gt_ml)
            rel_error_percent = (abs_error_ml / gt_ml * 100) if gt_ml > 0 else np.nan

            rows.append(
                {
                    "dataset": dataset_name,
                    "task": task_type,
                    "case": case_id,
                    "label": label,
                    "gt_volume_ml": gt_ml,
                    "pred_volume_ml": pred_ml,
                    "abs_error_ml": abs_error_ml,
                    "rel_error_percent": rel_error_percent,
                    "gt_file": str(gt_file),
                    "prediction_file": str(pred_file),
                }
            )

    if missing_predictions:
        print(f"[INFO] {dataset_name}: missing_predictions={len(missing_predictions)}")
        if fail_on_missing_predictions:
            preview = ", ".join(missing_predictions[:10])
            raise SystemExit(f"Missing predictions in {dataset_name}: {preview}")

    print(f"[INFO] {dataset_name}: volumetry_rows={len(rows)}")
    return rows


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "task", "label"])
        .agg(
            mean_gt_volume_ml=("gt_volume_ml", "mean"),
            mean_pred_volume_ml=("pred_volume_ml", "mean"),
            mean_abs_error_ml=("abs_error_ml", "mean"),
            mean_rel_error_percent=("rel_error_percent", "mean"),
            std_abs_error_ml=("abs_error_ml", "std"),
            std_rel_error_percent=("rel_error_percent", "std"),
            n_cases=("case", "count"),
        )
        .reset_index()
    )


def parse_args() -> argparse.Namespace:
    default_output = os.environ.get("VOLUMETRY_OUTPUT_DIR", "volumetry_outputs")
    parser = argparse.ArgumentParser(description="Compute liver/Couinaud volumetry from nnU-Net predictions.")
    parser.add_argument("--nnunet-raw", type=Path, default=Path("nnUNet_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path(default_output))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--predictions-dir-name", default="predictions")
    parser.add_argument("--case-output-name", default="volumetry_results_all_cases.csv")
    parser.add_argument("--summary-output-name", default="volumetry_summary.csv")
    parser.add_argument("--fail-on-missing-predictions", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    print(f"[INFO] nnUNet_raw: {args.nnunet_raw}")
    print(f"[INFO] output_dir: {args.output_dir}")
    print(f"[INFO] datasets: {', '.join(args.datasets)}")

    for dataset_name in args.datasets:
        all_rows.extend(
            collect_dataset_rows(
                nnunet_raw=args.nnunet_raw,
                dataset_name=dataset_name,
                predictions_dir_name=args.predictions_dir_name,
                fail_on_missing_predictions=args.fail_on_missing_predictions,
            )
        )

    if not all_rows:
        raise SystemExit("No volumetry rows were generated. Check dataset names and prediction folders.")

    df = pd.DataFrame(all_rows)
    results_csv = args.output_dir / args.case_output_name
    df.to_csv(results_csv, index=False)

    summary = build_summary(df)
    summary_csv = args.output_dir / args.summary_output_name
    summary.to_csv(summary_csv, index=False)

    print("Done.")
    print(f"Case-level results: {results_csv}")
    print(f"Summary: {summary_csv}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Legacy implementation kept for reference. The command-line implementation above
is the active entry point.

import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

NNUNET_RAW = Path("nnUNet_raw")

DATASETS = [
    "Dataset001_Liver",
    "Dataset002_LiverSegments",
    "Dataset101_LiverAug",
    "Dataset102_LiverSegmentsAug",
]

OUTPUT_DIR = Path("volumetry_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def get_case_id(path):
    name = path.name
    return name.replace(".nii.gz", "").replace(".nii", "")

def compute_volumes(mask_path, labels):
    img = nib.load(str(mask_path))
    data = img.get_fdata().astype(np.int16)

    spacing = img.header.get_zooms()[:3]  # mm
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]

    volumes = {}
    for label in labels:
        voxel_count = int(np.sum(data == label))
        volumes[label] = (voxel_count * voxel_volume_mm3) / 1000.0  # ml

    return volumes

def find_prediction(case_id, pred_dir):
    candidates = [
        pred_dir / f"{case_id}.nii.gz",
        pred_dir / f"{case_id}.nii",
    ]

    for c in candidates:
        if c.exists():
            return c

    return None

all_rows = []

for dataset_name in DATASETS:
    dataset_dir = NNUNET_RAW / dataset_name
    labels_dir = dataset_dir / "labelsTr"
    pred_dir = dataset_dir / "predictions"

    if not labels_dir.exists():
        print(f"[SKIP] No existe labelsTr en {dataset_name}")
        continue

    if not pred_dir.exists():
        print(f"[SKIP] No existe predictions en {dataset_name}")
        continue

    if "Segments" in dataset_name:
        labels = list(range(1, 9))   # Couinaud I-VIII
        task_type = "Couinaud"
    else:
        labels = [1]                 # hígado completo
        task_type = "Whole liver"

    gt_files = sorted(list(labels_dir.glob("*.nii")) + list(labels_dir.glob("*.nii.gz")))

    for gt_file in gt_files:
        case_id = get_case_id(gt_file)
        pred_file = find_prediction(case_id, pred_dir)

        if pred_file is None:
            print(f"[WARN] No se encontró predicción para {dataset_name}/{case_id}")
            continue

        gt_vol = compute_volumes(gt_file, labels)
        pred_vol = compute_volumes(pred_file, labels)

        for label in labels:
            gt_ml = gt_vol[label]
            pred_ml = pred_vol[label]
            abs_error_ml = abs(pred_ml - gt_ml)
            rel_error_percent = (abs_error_ml / gt_ml * 100) if gt_ml > 0 else np.nan

            all_rows.append({
                "dataset": dataset_name,
                "task": task_type,
                "case": case_id,
                "label": label,
                "gt_volume_ml": gt_ml,
                "pred_volume_ml": pred_ml,
                "abs_error_ml": abs_error_ml,
                "rel_error_percent": rel_error_percent,
            })

df = pd.DataFrame(all_rows)

results_csv = OUTPUT_DIR / "volumetry_results_all_cases.csv"
df.to_csv(results_csv, index=False)

summary = df.groupby(["dataset", "task", "label"]).agg(
    mean_gt_volume_ml=("gt_volume_ml", "mean"),
    mean_pred_volume_ml=("pred_volume_ml", "mean"),
    mean_abs_error_ml=("abs_error_ml", "mean"),
    mean_rel_error_percent=("rel_error_percent", "mean"),
    std_abs_error_ml=("abs_error_ml", "std"),
    std_rel_error_percent=("rel_error_percent", "std"),
    n_cases=("case", "count"),
).reset_index()

summary_csv = OUTPUT_DIR / "volumetry_summary.csv"
summary.to_csv(summary_csv, index=False)

print("Listo.")
print(f"Resultados por caso: {results_csv}")
print(f"Resumen: {summary_csv}")
print(summary)
"""
