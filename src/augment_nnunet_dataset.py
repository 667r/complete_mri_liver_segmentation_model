"""Create an augmented nnU-Net dataset with 3D rotations and scaling.

This implements the geometric augmentation:
small 3D rotations and isotropic scaling. It does not change intensities and
does not overwrite the input dataset.

Example:
    python src/augment_nnunet_dataset.py --dataset-dir nnUNet_raw/Dataset001_Liver --output-dir nnUNet_raw/Dataset101_LiverAug --dry-run
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
SIMPLEITK_INSTALL_HELP = """SimpleITK is required for 3D NIfTI augmentation.
Install it in the same Python environment where you run this script:

  python3 -m pip install SimpleITK

If pip is missing in Ubuntu/WSL, install pip first:

  apt update
  apt install -y python3-pip
"""


@dataclass(frozen=True)
class CaseFiles:
    case_id: str
    channel_files: list[Path]
    label_file: Path


def import_simpleitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise SystemExit(SIMPLEITK_INSTALL_HELP) from exc
    return sitk


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_dataset_json(dataset_dir: Path) -> dict:
    path = dataset_dir / "dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset.json in {dataset_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def split_image_name(path: Path, file_ending: str) -> tuple[str, str]:
    name = path.name
    if not name.endswith(file_ending):
        raise ValueError(f"{name} does not end with {file_ending}")
    stem = name[: -len(file_ending)]
    case_id, channel = stem.rsplit("_", 1)
    return case_id, channel


def collect_cases(dataset_dir: Path, file_ending: str) -> list[CaseFiles]:
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    if not images_tr.exists() or not labels_tr.exists():
        raise FileNotFoundError("Input dataset must contain imagesTr and labelsTr.")

    by_case: dict[str, list[tuple[str, Path]]] = {}
    for image_path in sorted(images_tr.glob(f"*{file_ending}")):
        case_id, channel = split_image_name(image_path, file_ending)
        by_case.setdefault(case_id, []).append((channel, image_path))

    cases: list[CaseFiles] = []
    for case_id, channel_paths in sorted(by_case.items()):
        label_file = labels_tr / f"{case_id}{file_ending}"
        if not label_file.exists():
            raise FileNotFoundError(f"Missing label for {case_id}: {label_file}")
        channel_files = [path for _, path in sorted(channel_paths)]
        cases.append(CaseFiles(case_id=case_id, channel_files=channel_files, label_file=label_file))
    return cases


def ensure_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} is not empty. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    (output_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (output_dir / "labelsTr").mkdir(parents=True, exist_ok=True)


def copy_originals(cases: list[CaseFiles], output_dir: Path, file_ending: str) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []
    for case in cases:
        for channel_index, image_path in enumerate(case.channel_files):
            destination = output_dir / "imagesTr" / f"{case.case_id}_{channel_index:04d}{file_ending}"
            shutil.copy2(image_path, destination)
        label_destination = output_dir / "labelsTr" / f"{case.case_id}{file_ending}"
        shutil.copy2(case.label_file, label_destination)
        manifest_rows.append(
            {
                "case_id": case.case_id,
                "source_case_id": case.case_id,
                "augmentation": "original",
                "rotation_x_degrees": "0",
                "rotation_y_degrees": "0",
                "rotation_z_degrees": "0",
                "scale": "1",
            }
        )
    return manifest_rows


def image_center_physical(image) -> tuple[float, float, float]:
    size = image.GetSize()
    center_index = [(dim - 1) / 2 for dim in size]
    return image.TransformContinuousIndexToPhysicalPoint(center_index)


def build_transform(reference_image, rx: float, ry: float, rz: float, scale: float):
    sitk = import_simpleitk()
    transform = sitk.AffineTransform(3)
    transform.SetCenter(image_center_physical(reference_image))
    transform.Rotate(1, 2, rx)
    transform.Rotate(0, 2, ry)
    transform.Rotate(0, 1, rz)
    transform.Scale((scale, scale, scale))
    return transform


def resample_image(image, reference, transform, is_label: bool):
    sitk = import_simpleitk()
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    default_value = 0
    return sitk.Resample(image, reference, transform, interpolator, default_value, image.GetPixelID())


def augment_case(
    case: CaseFiles,
    output_dir: Path,
    file_ending: str,
    aug_index: int,
    rotation_degrees: float,
    scale_range: tuple[float, float],
    rng: random.Random,
) -> dict[str, str]:
    sitk = import_simpleitk()
    channels = [sitk.ReadImage(str(path)) for path in case.channel_files]
    label = sitk.ReadImage(str(case.label_file))
    reference = channels[0]

    rx_deg = rng.uniform(-rotation_degrees, rotation_degrees)
    ry_deg = rng.uniform(-rotation_degrees, rotation_degrees)
    rz_deg = rng.uniform(-rotation_degrees, rotation_degrees)
    scale = rng.uniform(scale_range[0], scale_range[1])
    radians = 3.141592653589793 / 180
    transform = build_transform(reference, rx_deg * radians, ry_deg * radians, rz_deg * radians, scale)

    augmented_case_id = f"{case.case_id}_aug{aug_index:02d}"
    for channel_index, image in enumerate(channels):
        augmented = resample_image(image, reference, transform, is_label=False)
        out_path = output_dir / "imagesTr" / f"{augmented_case_id}_{channel_index:04d}{file_ending}"
        sitk.WriteImage(augmented, str(out_path), True)

    augmented_label = resample_image(label, reference, transform, is_label=True)
    label_out = output_dir / "labelsTr" / f"{augmented_case_id}{file_ending}"
    sitk.WriteImage(augmented_label, str(label_out), True)

    return {
        "case_id": augmented_case_id,
        "source_case_id": case.case_id,
        "augmentation": f"rotation_scale_{aug_index:02d}",
        "rotation_x_degrees": f"{rx_deg:.6f}",
        "rotation_y_degrees": f"{ry_deg:.6f}",
        "rotation_z_degrees": f"{rz_deg:.6f}",
        "scale": f"{scale:.6f}",
    }


def write_manifest(output_dir: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with (output_dir / "augmentation_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_dataset_json(input_json: dict, output_dir: Path, num_training: int) -> None:
    updated = dict(input_json)
    updated["numTraining"] = num_training
    (output_dir / "dataset.json").write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-augmentations", type=int, default=1)
    parser.add_argument("--rotation-degrees", type=float, default=10.0)
    parser.add_argument("--scale-range", nargs=2, type=float, default=(0.9, 1.1), metavar=("MIN", "MAX"))
    parser.add_argument("--seed", type=int, default=402)
    parser.add_argument("--include-originals", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.dataset_dir = resolve_repo_path(args.dataset_dir)
    args.output_dir = resolve_repo_path(args.output_dir)

    if args.num_augmentations < 0:
        raise ValueError("--num-augmentations must be non-negative.")
    scale_min, scale_max = args.scale_range
    if scale_min <= 0 or scale_max <= 0 or scale_min > scale_max:
        raise ValueError("--scale-range must be positive and ordered as MIN MAX.")

    dataset_json = load_dataset_json(args.dataset_dir)
    file_ending = dataset_json.get("file_ending", ".nii.gz")
    cases = collect_cases(args.dataset_dir, file_ending)
    originals = len(cases) if args.include_originals else 0
    generated = len(cases) * args.num_augmentations

    print(f"Input cases: {len(cases)}")
    print(f"Originals copied: {originals}")
    print(f"Augmented cases to generate: {generated}")
    print(f"Output dir: {args.output_dir}")
    if args.dry_run:
        print("Dry run only. No files were written.")
        return 0

    ensure_output(args.output_dir, args.overwrite)
    rng = random.Random(args.seed)
    manifest_rows: list[dict[str, str]] = []
    if args.include_originals:
        manifest_rows.extend(copy_originals(cases, args.output_dir, file_ending))

    for case_index, case in enumerate(cases, start=1):
        for aug_index in range(1, args.num_augmentations + 1):
            row = augment_case(
                case=case,
                output_dir=args.output_dir,
                file_ending=file_ending,
                aug_index=aug_index,
                rotation_degrees=args.rotation_degrees,
                scale_range=(scale_min, scale_max),
                rng=rng,
            )
            manifest_rows.append(row)
        print(f"[{case_index}/{len(cases)}] augmented {case.case_id}")

    write_manifest(args.output_dir, manifest_rows)
    write_dataset_json(dataset_json, args.output_dir, len(manifest_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
