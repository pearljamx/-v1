"""
YOLOv8 驾驶疲劳二分类模型训练脚本
================================
使用 Hugging Face 公开真实图像数据集的抽样子集训练 YOLO classification 模型。

运行:
    python -m yolo.hf_drowsiness_dataset --download
    python -m yolo.train_drowsiness_cls --quick
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = BASE_DIR / "datasets" / "hf_driver_drowsiness_yolo_cls"
TARGET_MODEL = BASE_DIR / "models_data" / "yolo_drowsiness_cls.pt"


def ensure_dataset(dataset_dir: Path) -> None:
    required = [
        dataset_dir / "train" / "drowsy",
        dataset_dir / "train" / "not_drowsy",
        dataset_dir / "val" / "drowsy",
        dataset_dir / "val" / "not_drowsy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "YOLO 分类数据集未准备好，请先运行 "
            "python -m yolo.hf_drowsiness_dataset --download；缺少: "
            + ", ".join(missing)
        )


def train(dataset_dir: Path = DEFAULT_DATASET, quick: bool = False):
    from ultralytics import YOLO

    ensure_dataset(dataset_dir)
    base_model = BASE_DIR / "yolov8n-cls.pt"
    model = YOLO(str(base_model) if base_model.exists() else "yolov8n-cls.pt")
    results = model.train(
        data=str(dataset_dir),
        epochs=2 if quick else 20,
        imgsz=128 if quick else 224,
        batch=8,
        workers=0,
        device="cpu",
        patience=2 if quick else 5,
        name="drowsiness_cls_quick" if quick else "drowsiness_cls",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
    )

    best_model = Path(results.save_dir) / "weights" / "best.pt"
    TARGET_MODEL.parent.mkdir(parents=True, exist_ok=True)
    if best_model.exists():
        shutil.copy2(best_model, TARGET_MODEL)
        print(f"YOLO 分类模型已保存: {TARGET_MODEL}")
    else:
        print(f"训练完成，但未找到 best.pt: {best_model}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 YOLOv8 驾驶疲劳分类模型")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET, help="YOLO 分类数据集目录")
    parser.add_argument("--quick", action="store_true", help="快速 CPU 训练")
    args = parser.parse_args()
    train(dataset_dir=args.data, quick=args.quick)


if __name__ == "__main__":
    main()
