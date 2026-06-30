"""Export nnU-Net validation summaries into the result csv tables.

The script scans one results folder or one summary.json file. For every
fold_*/validation/summary.json it writes fold-level, contrast-level and
case-level rows to a single CSV.

Example:
    python3 src/export_nnunet_protocol_results.py \
      --input models/nnUNet_results_aug_seg_8 \
      --output-file results/nnunet_protocol_results_aug_seg_8.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_PATTERN = re.compile(r"^(?P<case>pat(?P<patient>\d+)_ser(?P<series>\d+))(?:_aug\d+)?$")


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalize_patient(value: str) -> str:
    value = str(value).strip()
    if value.isdigit():
        return str(int(value))
    return value


def read_sequence_types(path: Path) -> dict[str, str]:
    return {
        row["Label"].strip(): row["Series Description"].strip()
        for row in read_csv_rows(path)
    }


def read_contrast_lookup(segmentation_key: Path, sequence_types: Path) -> dict[tuple[str, str], dict[str, str]]:
    sequence_lookup = read_sequence_types(sequence_types)
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv_rows(segmentation_key):
        patient_id = normalize_patient(row["DLDS"])
        series_id = str(row["Series"]).strip()
        contrast_label = row["Label"].strip()
        lookup[(patient_id, series_id)] = {
            "contrast_label": contrast_label,
            "contrast": sequence_lookup.get(contrast_label, ""),
        }
    return lookup


def find_summary_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.name != "summary.json":
            raise ValueError(f"Expected a summary.json file, got {input_path}")
        return [input_path]
    return sorted(input_path.glob("**/fold_*/validation/summary.json"))


def strip_nifti_suffix(path: str) -> str:
    name = Path(path).name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def parse_case_id(prediction_or_reference_file: str) -> str:
    return strip_nifti_suffix(prediction_or_reference_file)


def parse_patient_series(case_id: str, source_case_id: str | None = None) -> tuple[str, str]:
    for candidate in (source_case_id, case_id):
        if not candidate:
            continue
        match = CASE_PATTERN.match(candidate)
        if match:
            return normalize_patient(match.group("patient")), match.group("series")
    return "", ""


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def fmt(value: float | int | str | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    return f"{value:.10g}"


def derived_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    tp = safe_float(metrics.get("TP"))
    fp = safe_float(metrics.get("FP"))
    fn = safe_float(metrics.get("FN"))
    n_pred = safe_float(metrics.get("n_pred"))
    n_ref = safe_float(metrics.get("n_ref"))
    precision = divide(tp, (tp or 0) + (fp or 0))
    recall = divide(tp, (tp or 0) + (fn or 0))
    volume_error_pct = None
    if n_pred is not None and n_ref not in (None, 0):
        volume_error_pct = 100.0 * (n_pred - n_ref) / n_ref
    return {
        "precision": precision,
        "recall": recall,
        "volume_error_pct": volume_error_pct,
        "relative_volume_error_pct": volume_error_pct,
    }


def sum_metric(case_rows: list[dict[str, Any]], key: str) -> float | None:
    values = [safe_float(row.get(key)) for row in case_rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values)


def mean_metric(case_rows: list[dict[str, Any]], key: str) -> float | None:
    values = [safe_float(row.get(key)) for row in case_rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return mean(values)


def aggregate_case_rows(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sums = {
        "TP": sum_metric(case_rows, "TP"),
        "FP": sum_metric(case_rows, "FP"),
        "FN": sum_metric(case_rows, "FN"),
        "TN": sum_metric(case_rows, "TN"),
        "n_pred": sum_metric(case_rows, "n_pred"),
        "n_ref": sum_metric(case_rows, "n_ref"),
    }
    metrics = {
        "Dice": mean_metric(case_rows, "DSC"),
        "IoU": mean_metric(case_rows, "IoU"),
        **sums,
    }
    return {**metrics, **derived_metrics(metrics)}


def label_name(label_id: str, labels: dict[str, Any]) -> str:
    for name, value in labels.items():
        if str(value) == str(label_id):
            return name
    if label_id == "1":
        return "liver"
    return f"label_{label_id}"


def model_context(summary_path: Path) -> dict[str, str]:
    validation_dir = summary_path.parent
    fold_dir = validation_dir.parent
    model_dir = fold_dir.parent
    dataset_dir = model_dir.parent
    return {
        "dataset": dataset_dir.name,
        "trainer_configuration": model_dir.name,
        "fold": fold_dir.name,
        "test_setting": validation_dir.name,
        "model": f"{dataset_dir.name}/{model_dir.name}",
        "summary_file": str(summary_path),
    }


def load_dataset_labels(summary_path: Path, nnunet_raw_root: Path) -> dict[str, Any]:
    context = model_context(summary_path)
    candidates = [
        summary_path.parent.parent.parent / "dataset.json",
        summary_path.parent.parent.parent.parent / "dataset.json",
        nnunet_raw_root / context["dataset"] / "dataset.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            payload = read_json(candidate)
            labels = payload.get("labels", {})
            if isinstance(labels, dict):
                return labels
    return {"background": 0, "liver": 1}


def load_augmentation_sources(dataset_name: str, nnunet_raw_root: Path) -> dict[str, str]:
    manifest = nnunet_raw_root / dataset_name / "augmentation_manifest.csv"
    rows = read_csv_rows(manifest)
    return {
        row.get("case_id", ""): row.get("source_case_id", "")
        for row in rows
        if row.get("case_id") and row.get("source_case_id")
    }


def base_output_row(context: dict[str, str], row_type: str) -> dict[str, str]:
    return {
        "model": context["model"],
        "dataset": context["dataset"],
        "trainer_configuration": context["trainer_configuration"],
        "fold": context["fold"],
        "test_setting": context["test_setting"],
        "row_type": row_type,
        "case_id": "",
        "source_case_id": "",
        "patient_id": "",
        "series_id": "",
        "contrast_label": "",
        "contrast": "ALL",
        "segment": "",
        "n_cases": "",
        "volume_ml": "NA",
        "DSC": "NA",
        "IoU": "NA",
        "HD95_mm": "NA",
        "ASSD_mm": "NA",
        "volume_error_pct": "NA",
        "precision": "NA",
        "recall": "NA",
        "relative_volume_error_pct": "NA",
        "TP": "NA",
        "FP": "NA",
        "FN": "NA",
        "TN": "NA",
        "n_pred": "NA",
        "n_ref": "NA",
        "prediction_file": "",
        "reference_file": "",
        "summary_file": context["summary_file"],
    }


def fill_metric_columns(row: dict[str, str], metrics: dict[str, Any]) -> None:
    derived = derived_metrics(metrics)
    row["DSC"] = fmt(metrics.get("Dice"))
    row["IoU"] = fmt(metrics.get("IoU"))
    row["volume_error_pct"] = fmt(derived["volume_error_pct"])
    row["precision"] = fmt(derived["precision"])
    row["recall"] = fmt(derived["recall"])
    row["relative_volume_error_pct"] = fmt(derived["relative_volume_error_pct"])
    for key in ("TP", "FP", "FN", "TN", "n_pred", "n_ref"):
        row[key] = fmt(metrics.get(key))


def build_fold_mean_rows(
    summary: dict[str, Any],
    context: dict[str, str],
    labels: dict[str, Any],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    mean_by_label = summary.get("mean", {})
    if not isinstance(mean_by_label, dict):
        return rows
    for label_id, metrics in sorted(mean_by_label.items(), key=lambda item: str(item[0])):
        if str(label_id) == "0" or not isinstance(metrics, dict):
            continue
        row = base_output_row(context, "fold_mean")
        row["segment"] = label_name(str(label_id), labels)
        fill_metric_columns(row, metrics)
        rows.append(row)
    return rows


def build_case_rows(
    summary: dict[str, Any],
    context: dict[str, str],
    labels: dict[str, Any],
    contrast_lookup: dict[tuple[str, str], dict[str, str]],
    source_lookup: dict[str, str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case_entry in summary.get("metric_per_case", []):
        if not isinstance(case_entry, dict):
            continue
        prediction_file = str(case_entry.get("prediction_file", ""))
        reference_file = str(case_entry.get("reference_file", ""))
        case_id = parse_case_id(prediction_file or reference_file)
        source_case_id = source_lookup.get(case_id, "")
        patient_id, series_id = parse_patient_series(case_id, source_case_id)
        contrast_data = contrast_lookup.get((patient_id, series_id), {})
        metrics_by_label = case_entry.get("metrics", {})
        if not isinstance(metrics_by_label, dict):
            continue
        for label_id, metrics in sorted(metrics_by_label.items(), key=lambda item: str(item[0])):
            if str(label_id) == "0" or not isinstance(metrics, dict):
                continue
            row = base_output_row(context, "case")
            row["case_id"] = case_id
            row["source_case_id"] = source_case_id
            row["patient_id"] = patient_id
            row["series_id"] = series_id
            row["contrast_label"] = contrast_data.get("contrast_label", "")
            row["contrast"] = contrast_data.get("contrast", "UNKNOWN")
            row["segment"] = label_name(str(label_id), labels)
            row["prediction_file"] = prediction_file
            row["reference_file"] = reference_file
            fill_metric_columns(row, metrics)
            rows.append(row)
    return rows


def build_contrast_mean_rows(case_rows: list[dict[str, str]], context: dict[str, str]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in case_rows:
        grouped[(row["contrast_label"], row["contrast"], row["segment"])].append(row)

    output: list[dict[str, str]] = []
    for (contrast_label, contrast, segment), rows in sorted(grouped.items()):
        aggregate = aggregate_case_rows(rows)
        row = base_output_row(context, "contrast_mean")
        row["contrast_label"] = contrast_label
        row["contrast"] = contrast
        row["segment"] = segment
        row["n_cases"] = str(len(rows))
        row["DSC"] = fmt(aggregate.get("Dice"))
        row["IoU"] = fmt(aggregate.get("IoU"))
        row["volume_error_pct"] = fmt(aggregate.get("volume_error_pct"))
        row["precision"] = fmt(aggregate.get("precision"))
        row["recall"] = fmt(aggregate.get("recall"))
        row["relative_volume_error_pct"] = fmt(aggregate.get("relative_volume_error_pct"))
        for key in ("TP", "FP", "FN", "TN", "n_pred", "n_ref"):
            row[key] = fmt(aggregate.get(key))
        output.append(row)
    return output


def rows_for_summary(
    summary_path: Path,
    contrast_lookup: dict[tuple[str, str], dict[str, str]],
    nnunet_raw_root: Path,
) -> list[dict[str, str]]:
    summary = read_json(summary_path)
    context = model_context(summary_path)
    labels = load_dataset_labels(summary_path, nnunet_raw_root)
    source_lookup = load_augmentation_sources(context["dataset"], nnunet_raw_root)
    fold_rows = build_fold_mean_rows(summary, context, labels)
    case_rows = build_case_rows(summary, context, labels, contrast_lookup, source_lookup)
    contrast_rows = build_contrast_mean_rows(case_rows, context)
    return fold_rows + contrast_rows + case_rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(base_output_row({"model": "", "dataset": "", "trainer_configuration": "", "fold": "", "test_setting": "", "summary_file": ""}, "").keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("models"), help="summary.json file or directory to scan.")
    parser.add_argument("--output-file", type=Path, default=Path("data/reports/nnunet_protocol_results.csv"))
    parser.add_argument("--segmentation-key", type=Path, default=Path("data/SegmentationKey.csv"))
    parser.add_argument("--sequence-types", type=Path, default=Path("data/SequenceTypes.csv"))
    parser.add_argument("--nnunet-raw-root", type=Path, default=Path("nnUNet_raw"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_repo_path(args.input)
    output_file = resolve_repo_path(args.output_file)
    segmentation_key = resolve_repo_path(args.segmentation_key)
    sequence_types = resolve_repo_path(args.sequence_types)
    nnunet_raw_root = resolve_repo_path(args.nnunet_raw_root)

    contrast_lookup = read_contrast_lookup(segmentation_key, sequence_types)
    summary_files = find_summary_files(input_path)
    all_rows: list[dict[str, str]] = []
    for summary_path in summary_files:
        all_rows.extend(rows_for_summary(summary_path, contrast_lookup, nnunet_raw_root))

    write_rows(output_file, all_rows)
    print(f"Found summary files: {len(summary_files)}")
    print(f"Wrote rows: {len(all_rows)}")
    print(f"Output file: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
