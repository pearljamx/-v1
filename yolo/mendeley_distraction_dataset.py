"""
Prepare the Mendeley driver-distraction dataset for YOLO classification.

The Mendeley web page is public, but anonymous API/zip downloads can require
browser authorization. This script therefore supports the reliable workflow:

    1. Download the dataset from the Mendeley page with "Download All".
    2. Run this script with --zip or --source-dir.
    3. Train with python -m yolo.train_mendeley_distraction_cls.

The output layout is the standard Ultralytics classification layout:

datasets/mendeley_driver_distraction_yolo_cls/
├── train/<class_name>/*.jpg
├── val/<class_name>/*.jpg
└── test/<class_name>/*.jpg
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = BASE_DIR / "datasets" / "mendeley_driver_distractions_raw"
DEFAULT_OUTPUT = BASE_DIR / "datasets" / "mendeley_driver_distraction_yolo_cls"

SOURCE_URL = "https://data.mendeley.com/datasets/ykmr99nrsg/2"
SOURCE_DOI = "10.17632/ykmr99nrsg.2"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_ALIASES = {
    "safe_driving": [
        "c0",
        "safe",
        "safe driving",
        "normal",
        "normal driving",
        "not distracted",
        "notdistracted",
    ],
    "texting_right": ["c1", "texting right", "text right", "texting-right"],
    "phone_right": ["c2", "phone right", "calling right", "call right", "talking phone right"],
    "texting_left": ["c3", "texting left", "text left", "texting-left"],
    "phone_left": ["c4", "phone left", "calling left", "call left", "talking phone left"],
    "adjusting_radio": ["c5", "radio", "adjusting radio", "operating radio"],
    "drinking": ["c6", "drinking", "drink", "water bottle"],
    "reaching_behind": ["c7", "reaching behind", "reach behind", "reaching back"],
    "hair_makeup": ["c8", "hair", "makeup", "hair makeup", "hair or makeup"],
    "talking_to_passenger": ["c9", "talking passenger", "talking to passenger", "passenger"],
}


def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


NORMALIZED_ALIASES = {
    class_name: {normalize_text(alias) for alias in aliases}
    for class_name, aliases in CLASS_ALIASES.items()
}


def extract_zip(zip_path: Path, output_dir: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)
    return output_dir


def class_from_path(path: Path, root: Path) -> str | None:
    parts = list(path.relative_to(root).parts[:-1])
    # Prefer the nearest parent folder but allow dataset-specific nesting.
    for part in reversed(parts):
        norm = normalize_text(Path(part).stem)
        for class_name, aliases in NORMALIZED_ALIASES.items():
            if norm in aliases:
                return class_name
            if any(alias and alias in norm for alias in aliases):
                return class_name
    return None


def collect_images(source_dir: Path) -> dict[str, list[Path]]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    by_class: dict[str, list[Path]] = {class_name: [] for class_name in CLASS_ALIASES}
    unknown: list[str] = []
    for image_path in source_dir.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        class_name = class_from_path(image_path, source_dir)
        if class_name is None:
            unknown.append(str(image_path.relative_to(source_dir)))
            continue
        by_class[class_name].append(image_path)

    by_class = {key: sorted(paths) for key, paths in by_class.items() if paths}
    if not by_class:
        examples = ", ".join(unknown[:10])
        raise RuntimeError(
            "No classifiable images found. Check extracted folder names. "
            f"Unknown examples: {examples}"
        )
    return by_class


def split_paths(
    paths: list[Path],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
    max_per_class: int | None,
) -> dict[str, list[Path]]:
    shuffled = list(paths)
    rng.shuffle(shuffled)
    if max_per_class:
        shuffled = shuffled[:max_per_class]

    total = len(shuffled)
    if total < 3:
        raise RuntimeError(f"Need at least 3 images per class, got {total}")

    train_count = max(1, int(total * train_ratio))
    val_count = max(1, int(total * val_ratio))
    if train_count + val_count >= total:
        val_count = max(1, total - train_count - 1)
    test_count = total - train_count - val_count
    if test_count <= 0:
        train_count = max(1, train_count - 1)
        test_count = total - train_count - val_count

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count:train_count + val_count],
        "test": shuffled[train_count + val_count:],
    }


def prepare_dataset(
    source_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 2026,
    max_per_class: int | None = None,
) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    by_class = collect_images(source_dir)
    counts: dict[str, dict[str, int]] = {}

    for class_name, paths in by_class.items():
        split_map = split_paths(paths, train_ratio, val_ratio, rng, max_per_class)
        counts[class_name] = {}
        for split, split_paths_ in split_map.items():
            target_dir = output_dir / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            counts[class_name][split] = len(split_paths_)
            for idx, src in enumerate(split_paths_):
                suffix = src.suffix.lower() if src.suffix.lower() in IMAGE_EXTS else ".jpg"
                dst = target_dir / f"{class_name}_{idx:05d}{suffix}"
                shutil.copy2(src, dst)

    metadata = {
        "source": "Mendeley Data - Novel Driver Distractions Dataset With Low Lighting Support",
        "source_url": SOURCE_URL,
        "doi": SOURCE_DOI,
        "license": "CC BY-NC 3.0",
        "task": "Ultralytics YOLO classification",
        "output_classes": sorted(by_class.keys()),
        "counts": counts,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "seed": seed,
        "max_per_class": max_per_class,
    }
    (output_dir / "source_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Mendeley driver-distraction YOLO classification data")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip", type=Path, help="Mendeley Download All zip path")
    source.add_argument("--source-dir", type=Path, help="Already extracted Mendeley dataset directory")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Extraction directory for --zip")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output YOLO classification directory")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional cap per class for quick training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = extract_zip(args.zip, args.raw_dir) if args.zip else args.source_dir
    output = prepare_dataset(
        source_dir=source_dir,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_per_class=args.max_per_class,
    )
    print(f"Mendeley classification dataset ready: {output}")


if __name__ == "__main__":
    main()
