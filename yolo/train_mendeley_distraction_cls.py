"""
Train a YOLO classification model on the Mendeley driver-distraction dataset.

Prepare data first:
    python -m yolo.mendeley_distraction_dataset --zip D:\\path\\mendeley.zip

Then train:
    python -m yolo.train_mendeley_distraction_cls --quick
    python -m yolo.train_mendeley_distraction_cls
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from config import YOLO_DRIVER_CLASSIFIER_MODEL

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = BASE_DIR / "datasets" / "mendeley_driver_distraction_yolo_cls"


def ensure_dataset(dataset_dir: Path) -> None:
    required = [dataset_dir / "train", dataset_dir / "val"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Mendeley YOLO classification dataset is not ready. Run "
            "python -m yolo.mendeley_distraction_dataset --zip <downloaded.zip>; missing: "
            + ", ".join(missing)
        )
    train_classes = [p for p in (dataset_dir / "train").iterdir() if p.is_dir()]
    if len(train_classes) < 2:
        raise RuntimeError(f"Need at least two training classes under {dataset_dir / 'train'}")


def train(dataset_dir: Path = DEFAULT_DATASET, quick: bool = False):
    from ultralytics import YOLO

    ensure_dataset(dataset_dir)
    base_model = BASE_DIR / "yolov8n-cls.pt"
    model = YOLO(str(base_model) if base_model.exists() else "yolov8n-cls.pt")
    results = model.train(
        data=str(dataset_dir),
        epochs=2 if quick else 30,
        imgsz=160 if quick else 224,
        batch=8,
        workers=0,
        device="cpu",
        patience=2 if quick else 6,
        name="mendeley_distraction_cls_quick" if quick else "mendeley_distraction_cls",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
    )

    best_model = Path(results.save_dir) / "weights" / "best.pt"
    target = Path(YOLO_DRIVER_CLASSIFIER_MODEL)
    target.parent.mkdir(parents=True, exist_ok=True)
    if best_model.exists():
        shutil.copy2(best_model, target)
        print(f"Driver distraction classifier saved: {target}")
    else:
        print(f"Training finished but best.pt was not found: {best_model}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Mendeley driver-distraction YOLO classification model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET, help="YOLO classification dataset directory")
    parser.add_argument("--quick", action="store_true", help="Quick CPU smoke training")
    args = parser.parse_args()
    train(dataset_dir=args.data, quick=args.quick)


if __name__ == "__main__":
    main()
