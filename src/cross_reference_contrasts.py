"""Build contrast cross-reference reports for the local liver MRI dataset.

This script does not convert images or train models. It joins:

- data/SegmentationKey.csv: patient/series -> contrast label
- data/SequenceTypes.csv: contrast label -> human-readable name
- data/Segmentation/Segmentation: DICOM images and whole-liver masks
- data/8 segmentos: optional Couinaud NIfTI masks

Example:
    python src/cross_reference_contrasts.py --output-dir data/reports
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DICOM_PATTERN = "*.dicom"
COUINAUD_PATTERN = re.compile(r"^(?P<patient>\d+)_(?P<series>\d+)m?-label(?:\.nii(?:\.gz)?)$")


def normalize_patient(value: str) -> str:
    value = str(value).strip()
    if value.isdigit():
        return str(int(value))
    return value


def patient_dir_name(patient_id: str) -> str:
    if patient_id.isdigit():
        return f"{int(patient_id):04d}"
    return patient_id


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def read_sequence_types(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {
            row["Label"].strip(): row["Series Description"].strip()
            for row in csv.DictReader(f)
        }


def read_segmentation_key(path: Path, sequence_types: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            patient_id = normalize_patient(row["DLDS"])
            series_id = str(row["Series"]).strip()
            contrast_label = row["Label"].strip()
            rows.append(
                {
                    "patient_id": patient_id,
                    "patient_dir": patient_dir_name(patient_id),
                    "series_id": series_id,
                    "contrast_label": contrast_label,
                    "contrast_name": sequence_types.get(contrast_label, ""),
                }
            )
    return rows


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def build_whole_liver_rows(
    segmentation_root: Path,
    key_rows: list[dict[str, str]],
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for key_row in key_rows:
        case_root = segmentation_root / key_row["patient_dir"] / key_row["series_id"]
        images_dir = case_root / "images"
        masks_dir = case_root / "masks"
        image_count = count_files(images_dir, DICOM_PATTERN)
        mask_count = count_files(masks_dir, DICOM_PATTERN)
        rows.append(
            {
                "case_id": f"pat{key_row['patient_dir']}_ser{key_row['series_id']}",
                **key_row,
                "image_dicom_count": image_count,
                "mask_dicom_count": mask_count,
                "has_images": str(image_count > 0),
                "has_masks": str(mask_count > 0),
                "has_complete_pair": str(image_count > 0 and mask_count > 0),
                "image_mask_slice_count_match": str(image_count == mask_count),
                "images_dir": str(images_dir),
                "masks_dir": str(masks_dir),
            }
        )
    return rows


def parse_couinaud_label(path: Path) -> tuple[str, str] | None:
    match = COUINAUD_PATTERN.match(path.name)
    if not match:
        return None
    return normalize_patient(match.group("patient")), match.group("series")


def find_couinaud_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    nii_files = list(root.rglob("*.nii"))
    nii_gz_files = list(root.rglob("*.nii.gz"))
    return sorted(set(nii_files + nii_gz_files))


def build_couinaud_rows(
    couinaud_root: Path,
    segmentation_root: Path,
    key_lookup: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for label_path in find_couinaud_files(couinaud_root):
        parsed = parse_couinaud_label(label_path)
        if parsed is None:
            rows.append(
                {
                    "patient_id": "",
                    "patient_dir": "",
                    "series_id": "",
                    "contrast_label": "",
                    "contrast_name": "",
                    "image_dicom_count": 0,
                    "has_matching_dicom": "False",
                    "label_path": str(label_path),
                    "parse_status": "unparsed_filename",
                }
            )
            continue

        patient_id, series_id = parsed
        key_row = key_lookup.get((patient_id, series_id))
        patient_dir = patient_dir_name(patient_id)
        images_dir = segmentation_root / patient_dir / series_id / "images"
        image_count = count_files(images_dir, DICOM_PATTERN)
        rows.append(
            {
                "case_id": f"pat{patient_dir}_ser{series_id}",
                "patient_id": patient_id,
                "patient_dir": patient_dir,
                "series_id": series_id,
                "contrast_label": key_row["contrast_label"] if key_row else "",
                "contrast_name": key_row["contrast_name"] if key_row else "",
                "image_dicom_count": image_count,
                "has_matching_dicom": str(image_count > 0),
                "images_dir": str(images_dir),
                "label_path": str(label_path),
                "parse_status": "ok" if key_row else "missing_in_segmentation_key",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_whole_liver(rows: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    grouped: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["contrast_label"])].append(row)

    summary: list[dict[str, str | int]] = []
    for label in sorted(grouped):
        label_rows = grouped[label]
        complete_rows = [r for r in label_rows if r["has_complete_pair"] == "True"]
        patients = {str(r["patient_id"]) for r in complete_rows}
        summary.append(
            {
                "source": "whole_liver_dicom",
                "contrast_label": label,
                "contrast_name": str(label_rows[0]["contrast_name"]),
                "cases": len(complete_rows),
                "patients": len(patients),
                "image_slices": sum(int(r["image_dicom_count"]) for r in complete_rows),
                "mask_slices": sum(int(r["mask_dicom_count"]) for r in complete_rows),
            }
        )
    return summary


def summarize_couinaud(rows: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    counter = Counter(str(row["contrast_label"]) or "UNMAPPED" for row in rows)
    patients_by_label: dict[str, set[str]] = defaultdict(set)
    names_by_label: dict[str, str] = {}
    matching_dicom_by_label: Counter[str] = Counter()
    for row in rows:
        label = str(row["contrast_label"]) or "UNMAPPED"
        if row.get("patient_id"):
            patients_by_label[label].add(str(row["patient_id"]))
        names_by_label[label] = str(row.get("contrast_name", ""))
        if row.get("has_matching_dicom") == "True":
            matching_dicom_by_label[label] += 1

    return [
        {
            "source": "couinaud_nifti",
            "contrast_label": label,
            "contrast_name": names_by_label.get(label, ""),
            "cases": counter[label],
            "patients": len(patients_by_label[label]),
            "cases_with_matching_dicom": matching_dicom_by_label[label],
        }
        for label in sorted(counter)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-root", type=Path, default=Path("data/Segmentation/Segmentation"))
    parser.add_argument("--segmentation-key", type=Path, default=Path("data/SegmentationKey.csv"))
    parser.add_argument("--sequence-types", type=Path, default=Path("data/SequenceTypes.csv"))
    parser.add_argument("--couinaud-root", type=Path, default=Path("data/8 segmentos"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/reports"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.segmentation_root = resolve_repo_path(args.segmentation_root)
    args.segmentation_key = resolve_repo_path(args.segmentation_key)
    args.sequence_types = resolve_repo_path(args.sequence_types)
    args.couinaud_root = resolve_repo_path(args.couinaud_root)
    args.output_dir = resolve_repo_path(args.output_dir)

    sequence_types = read_sequence_types(args.sequence_types)
    key_rows = read_segmentation_key(args.segmentation_key, sequence_types)
    key_lookup = {
        (row["patient_id"], row["series_id"]): row
        for row in key_rows
    }

    whole_liver_rows = build_whole_liver_rows(args.segmentation_root, key_rows)
    couinaud_rows = build_couinaud_rows(args.couinaud_root, args.segmentation_root, key_lookup)
    summary_rows = summarize_whole_liver(whole_liver_rows) + summarize_couinaud(couinaud_rows)

    write_csv(args.output_dir / "whole_liver_contrast_reference.csv", whole_liver_rows)
    write_csv(args.output_dir / "couinaud_contrast_reference.csv", couinaud_rows)
    write_csv(args.output_dir / "contrast_summary.csv", summary_rows)

    whole_complete = sum(1 for row in whole_liver_rows if row["has_complete_pair"] == "True")
    whole_patients = {row["patient_id"] for row in whole_liver_rows if row["has_complete_pair"] == "True"}
    couinaud_patients = {row["patient_id"] for row in couinaud_rows if row.get("patient_id")}

    print(f"Whole-liver DICOM cases with image+mask: {whole_complete}")
    print(f"Whole-liver patients: {len(whole_patients)}")
    print(f"Couinaud NIfTI masks: {len(couinaud_rows)}")
    print(f"Couinaud patients: {len(couinaud_patients)}")
    print(f"Wrote reports to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
