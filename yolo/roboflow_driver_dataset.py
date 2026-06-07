"""
Roboflow 驾驶状态数据集导入工具
==============================
支持两种方式准备外部 YOLOv8 数据集：
1. 设置 ROBOFLOW_API_KEY 后自动下载 Driver fatigue and distraction；
2. 手动从 Roboflow 导出 YOLOv8 zip 后导入本地。

目标训练产物由 yolo/train_driver_state.py 保存为 models_data/yolo_driver_state.pt。
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "dataset" / "roboflow_driver_state"
TARGET_MODEL_NAME = "yolo_driver_state.pt"

ROBOFLOW_WORKSPACE = "mds-workspace-arqn1"
ROBOFLOW_PROJECT = "driver-fatigue-and-distraction-bned4"
ROBOFLOW_DEFAULT_VERSION = 1

CLASS_MAPPING = {
    "driver using phone": "phone",
    "driver smoking": "smoking",
    "driver eating": "eating",
    "driver turning": "turning",
    "driver drowsy": "drowsy",
    "driver sleeping": "drowsy",
    "driver awake": "normal",
}


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def download_from_roboflow(output_dir: Path, version: int) -> Path:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 ROBOFLOW_API_KEY；请设置后重试，或使用 --zip 导入手动下载的数据集。")

    from roboflow import Roboflow

    clean_dir(output_dir)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    # Roboflow SDK may mis-encode non-ASCII Windows paths. Download into an
    # ASCII-only temp directory first, then move the extracted files back.
    with tempfile.TemporaryDirectory(prefix="roboflow_driver_state_") as tmp:
        tmp_dir = Path(tmp)
        dataset = project.version(version).download("yolov8", location=str(tmp_dir), overwrite=True)
        downloaded = Path(dataset.location)
        if not downloaded.exists():
            downloaded = tmp_dir
        for item in downloaded.iterdir():
            target = output_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
    return output_dir


def import_zip(zip_path: Path, output_dir: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"找不到 zip 文件: {zip_path}")
    clean_dir(output_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    return output_dir


def find_data_yaml(dataset_dir: Path) -> Path:
    candidates = list(dataset_dir.rglob("data.yaml"))
    if not candidates:
        raise FileNotFoundError(f"未在 {dataset_dir} 中找到 data.yaml")
    return candidates[0]


def normalize_data_yaml(dataset_dir: Path) -> Path:
    data_yaml = find_data_yaml(dataset_dir)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = data_yaml.parent
    data["path"] = str(root.resolve()).replace("\\", "/")
    data["train"] = "train/images"
    data["val"] = "valid/images" if (root / "valid").exists() else "val/images"
    data["test"] = "test/images"
    data["driver_state_mapping"] = CLASS_MAPPING
    data["target_model"] = TARGET_MODEL_NAME
    data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="下载或导入 Roboflow 驾驶状态 YOLOv8 数据集")
    parser.add_argument("--zip", type=Path, help="手动下载的 YOLOv8 zip 路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出数据集目录")
    parser.add_argument("--version", type=int, default=ROBOFLOW_DEFAULT_VERSION, help="Roboflow 数据集版本")
    args = parser.parse_args()

    dataset_dir = import_zip(args.zip, args.output) if args.zip else download_from_roboflow(args.output, args.version)
    data_yaml = normalize_data_yaml(dataset_dir)
    print(f"数据集已就绪: {dataset_dir}")
    print(f"训练配置: {data_yaml}")
    print(f"目标模型: models_data/{TARGET_MODEL_NAME}")
    print("类别映射:")
    for src, dst in CLASS_MAPPING.items():
        print(f"  {src} -> {dst}")


if __name__ == "__main__":
    main()
