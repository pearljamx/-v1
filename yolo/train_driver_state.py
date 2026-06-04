"""
YOLOv8 驾驶状态检测模型训练脚本
==============================
使用 yolo/roboflow_driver_dataset.py 准备的外部公开数据集训练模型。

运行:
    python -m yolo.train_driver_state --quick
    python -m yolo.train_driver_state
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from config import YOLO_DRIVER_STATE_MODEL

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_YAML = BASE_DIR / "dataset" / "roboflow_driver_state" / "data.yaml"


def resolve_data_yaml(data: str | None) -> Path:
    data_yaml = Path(data) if data else DEFAULT_DATA_YAML
    if not data_yaml.exists():
        candidates = list((BASE_DIR / "dataset" / "roboflow_driver_state").rglob("data.yaml"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(
            "未找到外部数据集 data.yaml。请先运行 "
            "python -m yolo.roboflow_driver_dataset 或使用 --data 指定。"
        )
    return data_yaml


def train(data: str | None = None, quick: bool = False):
    from ultralytics import YOLO

    data_yaml = resolve_data_yaml(data)
    model = YOLO(str(BASE_DIR / "yolov8n.pt") if (BASE_DIR / "yolov8n.pt").exists() else "yolov8n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=5 if quick else 50,
        imgsz=224 if quick else 320,
        batch=4 if quick else 8,
        workers=0,
        device="cpu",
        patience=3 if quick else 10,
        name="driver_state_quick" if quick else "driver_state_detection",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
    )

    best_model = Path(results.save_dir) / "weights" / "best.pt"
    target = Path(YOLO_DRIVER_STATE_MODEL)
    target.parent.mkdir(parents=True, exist_ok=True)
    if best_model.exists():
        shutil.copy2(best_model, target)
        print(f"模型已保存: {target}")
    else:
        print(f"训练完成，但未找到 best.pt: {best_model}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="训练驾驶状态 YOLOv8 模型")
    parser.add_argument("--data", help="外部数据集 data.yaml 路径")
    parser.add_argument("--quick", action="store_true", help="快速训练验证")
    args = parser.parse_args()
    train(data=args.data, quick=args.quick)


if __name__ == "__main__":
    main()
