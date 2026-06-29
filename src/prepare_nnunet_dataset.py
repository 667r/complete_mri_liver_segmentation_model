"""Prepare nnU-Net v2 datasets from the local liver MRI data.

The script is intentionally not executed by the protocol. Use it when you are
ready to convert data into an nnU-Net raw dataset.

Examples:
    python src/prepare_nnunet_dataset.py --task whole-liver --output-root nnUNet_raw --dry-run
    python src/prepare_nnunet_dataset.py --task whole-liver --output-root nnUNet_raw --dataset-id 1 --dataset-name Liver
    python src/prepare_nnunet_dataset.py --task couinaud --output-root nnUNet_raw --dataset-id 2 --dataset-name LiverSegments
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SIMPLEITK_INSTALL_HELP = """SimpleITK is required for DICOM/NIfTI conversion.
Install it in the same Python environment where you run this script:

  python3 -m pip install SimpleITK

If pip is missing in Ubuntu/WSL, install pip first:

  apt update
  apt install -y python3-pip
"""


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    patient_id: str
    patient_dir: str
    series_id: str
    contrast_label: str
    contrast_name: str
    image_dir: Path
    label_source: Path
    label_source_type: str
    split: str = "train"


def normalize_patient(value: str) -> str:
    value = str(value).strip()
    if value.isdigit():
        return str(int(value))
    return value


def patient_dir_name(patient_id: str) -> str:
    if patient_id.isdigit():
        return f"{int(patient_id):04d}"
    return patient_id


def dataset_dir_name(dataset_id: int, dataset_name: str) -> str:
    return f"Dataset{dataset_id:03d}_{dataset_name}"


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


def has_dicom_files(path: Path) -> bool:
    return path.exists() and any(path.glob("*.dicom"))


def build_whole_liver_cases(
    segmentation_root: Path,
    key_rows: list[dict[str, str]],
) -> list[CaseRecord]:
    cases: list[CaseRecord] = []
    for row in key_rows:
        case_root = segmentation_root / row["patient_dir"] / row["series_id"]
        image_dir = case_root / "images"
        mask_dir = case_root / "masks"
        if not has_dicom_files(image_dir) or not has_dicom_files(mask_dir):
            continue
        cases.append(
            CaseRecord(
                case_id=f"pat{row['patient_dir']}_ser{row['series_id']}",
                patient_id=row["patient_id"],
                patient_dir=row["patient_dir"],
                series_id=row["series_id"],
                contrast_label=row["contrast_label"],
                contrast_name=row["contrast_name"],
                image_dir=image_dir,
                label_source=mask_dir,
                label_source_type="dicom_whole_liver",
            )
        )
    return cases


def parse_couinaud_label(label_path: Path) -> tuple[str, str] | None:
    name = label_path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if not name.endswith("-label"):
        return None
    stem = name[: -len("-label")]
    parts = stem.split("_")
    if len(parts) != 2:
        return None
    patient_id = normalize_patient(parts[0])
    series_id = parts[1].removesuffix("m")
    if not patient_id or not series_id:
        return None
    return patient_id, series_id


def find_nifti_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(set(root.rglob("*.nii")) | set(root.rglob("*.nii.gz")))


def build_couinaud_cases(
    couinaud_root: Path,
    segmentation_root: Path,
    key_lookup: dict[tuple[str, str], dict[str, str]],
) -> list[CaseRecord]:
    cases: list[CaseRecord] = []
    for label_path in find_nifti_files(couinaud_root):
        parsed = parse_couinaud_label(label_path)
        if parsed is None:
            continue
        patient_id, series_id = parsed
        patient_dir = patient_dir_name(patient_id)
        image_dir = segmentation_root / patient_dir / series_id / "images"
        key_row = key_lookup.get((patient_id, series_id))
        if key_row is None or not has_dicom_files(image_dir):
            continue
        cases.append(
            CaseRecord(
                case_id=f"pat{patient_dir}_ser{series_id}",
                patient_id=patient_id,
                patient_dir=patient_dir,
                series_id=series_id,
                contrast_label=key_row["contrast_label"],
                contrast_name=key_row["contrast_name"],
                image_dir=image_dir,
                label_source=label_path,
                label_source_type="nifti_couinaud",
            )
        )
    return cases


def with_patient_splits(
    cases: list[CaseRecord],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[CaseRecord]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1.")

    patients = sorted({case.patient_id for case in cases})
    rng = random.Random(seed)
    rng.shuffle(patients)

    def split_count(ratio: float) -> int:
        if ratio == 0:
            return 0
        return max(1, round(len(patients) * ratio))

    n_test = split_count(test_ratio)
    n_val = split_count(val_ratio)
    test_patients = set(patients[:n_test])
    val_patients = set(patients[n_test : n_test + n_val])

    output: list[CaseRecord] = []
    for case in cases:
        if case.patient_id in test_patients:
            split = "test"
        elif case.patient_id in val_patients:
            split = "val"
        else:
            split = "train"
        output.append(CaseRecord(**{**case.__dict__, "split": split}))
    return output


def import_simpleitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise SystemExit(SIMPLEITK_INSTALL_HELP) from exc
    return sitk


def read_dicom_volume(directory: Path):
    sitk = import_simpleitk()
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(directory))
    if series_ids:
        file_names = reader.GetGDCMSeriesFileNames(str(directory), series_ids[0])
    else:
        file_names = [str(path) for path in sorted(directory.glob("*.dicom"))]
    if not file_names:
        raise FileNotFoundError(f"No DICOM files found in {directory}")
    reader.SetFileNames(file_names)
    return reader.Execute()


def same_geometry(left, right) -> bool:
    return (
        left.GetSize() == right.GetSize()
        and left.GetSpacing() == right.GetSpacing()
        and left.GetOrigin() == right.GetOrigin()
        and left.GetDirection() == right.GetDirection()
    )


def align_label_to_image(label, image, resample_labels: bool):
    sitk = import_simpleitk()
    if same_geometry(label, image):
        return label
    if label.GetSize() == image.GetSize() and not resample_labels:
        label = sitk.Image(label)
        label.CopyInformation(image)
        return label
    if resample_labels:
        return sitk.Resample(label, image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, label.GetPixelID())
    raise ValueError(
        "Label geometry does not match image geometry. Re-run with --resample-labels "
        "only after visually checking that the label belongs to this image."
    )


def read_label(case: CaseRecord, image, task: str, resample_labels: bool):
    sitk = import_simpleitk()
    if case.label_source_type == "dicom_whole_liver":
        label = read_dicom_volume(case.label_source)
        label = align_label_to_image(label, image, resample_labels)
        label = sitk.NotEqual(label, 0)
        return sitk.Cast(label, sitk.sitkUInt8)

    label = sitk.ReadImage(str(case.label_source))
    label = align_label_to_image(label, image, resample_labels)
    if task == "couinaud":
        return sitk.Cast(label, sitk.sitkUInt8)
    return sitk.Cast(label > 0, sitk.sitkUInt8)


def ensure_empty_or_overwrite(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if overwrite:
        return
    existing = any(path.iterdir())
    if existing:
        raise FileExistsError(f"{path} already exists and is not empty. Use --overwrite to write into it.")


def write_dataset_json(dataset_dir: Path, task: str, num_training: int) -> None:
    if task == "whole-liver":
        labels = {"background": 0, "liver": 1}
    else:
        labels = {
            "background": 0,
            "Couinaud_I": 1,
            "Couinaud_II": 2,
            "Couinaud_III": 3,
            "Couinaud_IV": 4,
            "Couinaud_V": 5,
            "Couinaud_VI": 6,
            "Couinaud_VII": 7,
            "Couinaud_VIII": 8,
        }
    payload = {
        "channel_names": {"0": "MRI"},
        "labels": labels,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_manifest(dataset_dir: Path, cases: list[CaseRecord]) -> None:
    fieldnames = [
        "case_id",
        "patient_id",
        "series_id",
        "contrast_label",
        "contrast_name",
        "split",
        "image_dir",
        "label_source",
        "label_source_type",
    ]
    with (dataset_dir / "case_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "patient_id": case.patient_id,
                    "series_id": case.series_id,
                    "contrast_label": case.contrast_label,
                    "contrast_name": case.contrast_name,
                    "split": case.split,
                    "image_dir": case.image_dir,
                    "label_source": case.label_source,
                    "label_source_type": case.label_source_type,
                }
            )


def write_patient_split_json(dataset_dir: Path, cases: list[CaseRecord]) -> None:
    train_cases = sorted(case.case_id for case in cases if case.split == "train")
    val_cases = sorted(case.case_id for case in cases if case.split == "val")
    if not val_cases:
        return
    payload = [{"train": train_cases, "val": val_cases}]
    (dataset_dir / "splits_patient_level.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def convert_cases(
    cases: list[CaseRecord],
    dataset_dir: Path,
    task: str,
    resample_labels: bool,
    copy_test_labels: bool,
) -> None:
    sitk = import_simpleitk()
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"

    for folder in (images_tr, labels_tr, images_ts):
        folder.mkdir(parents=True, exist_ok=True)
    if copy_test_labels:
        labels_ts.mkdir(parents=True, exist_ok=True)

    for index, case in enumerate(cases, start=1):
        image = read_dicom_volume(case.image_dir)
        label = read_label(case, image, task, resample_labels)
        image = sitk.Cast(image, sitk.sitkFloat32)

        if case.split == "test":
            image_path = images_ts / f"{case.case_id}_0000.nii.gz"
            label_path = labels_ts / f"{case.case_id}.nii.gz"
        else:
            image_path = images_tr / f"{case.case_id}_0000.nii.gz"
            label_path = labels_tr / f"{case.case_id}.nii.gz"

        sitk.WriteImage(image, str(image_path), True)
        if case.split != "test" or copy_test_labels:
            sitk.WriteImage(label, str(label_path), True)

        print(f"[{index}/{len(cases)}] wrote {case.case_id} ({case.contrast_label}, {case.split})")


def copy_existing_test_labels_note(dataset_dir: Path, cases: list[CaseRecord], copy_test_labels: bool) -> None:
    if copy_test_labels:
        return
    test_cases = [case for case in cases if case.split == "test"]
    if not test_cases:
        return
    note = (
        "Test labels were not copied because nnU-Net does not use labelsTs for inference. "
        "They remain referenced in case_manifest.csv.\n"
    )
    (dataset_dir / "labelsTs_README.txt").write_text(note, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("whole-liver", "couinaud"), default="whole-liver")
    parser.add_argument("--segmentation-root", type=Path, default=Path("data/Segmentation/Segmentation"))
    parser.add_argument("--segmentation-key", type=Path, default=Path("data/SegmentationKey.csv"))
    parser.add_argument("--sequence-types", type=Path, default=Path("data/SequenceTypes.csv"))
    parser.add_argument("--couinaud-root", type=Path, default=Path("data/8 segmentos"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int)
    parser.add_argument("--dataset-name")
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=402)
    parser.add_argument("--resample-labels", action="store_true")
    parser.add_argument("--copy-test-labels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.segmentation_root = resolve_repo_path(args.segmentation_root)
    args.segmentation_key = resolve_repo_path(args.segmentation_key)
    args.sequence_types = resolve_repo_path(args.sequence_types)
    args.couinaud_root = resolve_repo_path(args.couinaud_root)
    args.output_root = resolve_repo_path(args.output_root)

    dataset_id = args.dataset_id or (1 if args.task == "whole-liver" else 2)
    dataset_name = args.dataset_name or ("Liver" if args.task == "whole-liver" else "LiverSegments")
    dataset_dir = args.output_root / dataset_dir_name(dataset_id, dataset_name)

    sequence_types = read_sequence_types(args.sequence_types)
    key_rows = read_segmentation_key(args.segmentation_key, sequence_types)
    key_lookup = {(row["patient_id"], row["series_id"]): row for row in key_rows}

    if args.task == "whole-liver":
        cases = build_whole_liver_cases(args.segmentation_root, key_rows)
    else:
        cases = build_couinaud_cases(args.couinaud_root, args.segmentation_root, key_lookup)

    cases = with_patient_splits(cases, args.val_ratio, args.test_ratio, args.seed)
    train_or_val_cases = [case for case in cases if case.split != "test"]

    print(f"Task: {args.task}")
    print(f"Output dataset: {dataset_dir}")
    print(f"Cases found: {len(cases)}")
    print(f"Training/validation cases for imagesTr/labelsTr: {len(train_or_val_cases)}")
    print(f"Test holdout cases for imagesTs: {sum(1 for case in cases if case.split == 'test')}")

    if args.dry_run:
        print("Dry run only. No files were written.")
        return 0

    ensure_empty_or_overwrite(dataset_dir, args.overwrite)
    if args.overwrite and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    convert_cases(cases, dataset_dir, args.task, args.resample_labels, args.copy_test_labels)
    write_dataset_json(dataset_dir, args.task, num_training=len(train_or_val_cases))
    write_manifest(dataset_dir, cases)
    write_patient_split_json(dataset_dir, cases)
    copy_existing_test_labels_note(dataset_dir, cases, args.copy_test_labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
