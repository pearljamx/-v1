"""
Hugging Face 真实驾驶疲劳分类数据集准备工具
========================================
从 n7i5x9/driver-drowsiness-dataset 下载 validation parquet 分片，
抽样导出为 YOLO classification 目录结构:

datasets/hf_driver_drowsiness_yolo_cls/
├── train/
│   ├── drowsy/
│   └── not_drowsy/
└── val/
    ├── drowsy/
    └── not_drowsy/

该数据集是公开 Hugging Face 数据集，未 gated，图像标签为 drowsy / not_drowsy。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import polars as pl
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WORK_DIR = BASE_DIR / "datasets" / "hf_driver_drowsiness"
DEFAULT_PARQUET = DEFAULT_WORK_DIR / "validation.parquet"
DEFAULT_OUTPUT = BASE_DIR / "datasets" / "hf_driver_drowsiness_yolo_cls"

DATASET_ID = "n7i5x9/driver-drowsiness-dataset"
PARQUET_URL = (
    "https://huggingface.co/datasets/"
    f"{DATASET_ID}/resolve/main/data/validation-00000-of-00001.parquet"
)
LABEL_NAMES = {0: "drowsy", 1: "not_drowsy"}


def download_parquet(target: Path = DEFAULT_PARQUET) -> Path:
    """Download the public validation parquet shard if it is missing."""
    if target.exists() and target.stat().st_size > 0:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(PARQUET_URL, stream=True, timeout=300) as response:
        response.raise_for_status()
        with target.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return target


def prepare_dataset(
    parquet_path: Path = DEFAULT_PARQUET,
    output_dir: Path = DEFAULT_OUTPUT,
    train_per_class: int = 160,
    val_per_class: int = 40,
) -> Path:
    """
    Convert a parquet shard into a small balanced YOLO classification dataset.

    The source shard has 2,311 real images. Defaults export 320 train + 80 val
    images, enough for a reproducible quick CPU training run.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"未找到 parquet 文件: {parquet_path}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val"):
        for class_name in LABEL_NAMES.values():
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(parquet_path)
    counts = {"train": {}, "val": {}}

    for label, class_name in LABEL_NAMES.items():
        class_df = df.filter(pl.col("label") == label).head(train_per_class + val_per_class)
        if class_df.height < train_per_class + val_per_class:
            raise RuntimeError(
                f"{class_name} 样本不足: {class_df.height}, "
                f"需要 {train_per_class + val_per_class}"
            )

        for idx, row in enumerate(class_df.iter_rows(named=True)):
            split = "train" if idx < train_per_class else "val"
            image_bytes = row["image"]["bytes"]
            filename = output_dir / split / class_name / f"{class_name}_{idx:04d}.jpg"
            filename.write_bytes(image_bytes)
            counts[split][class_name] = counts[split].get(class_name, 0) + 1

    metadata = {
        "source_dataset": DATASET_ID,
        "source_url": PARQUET_URL,
        "source_file": str(parquet_path),
        "task": "YOLO classification",
        "labels": LABEL_NAMES,
        "counts": counts,
    }
    (output_dir / "source_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 Hugging Face 真实驾驶疲劳 YOLO 分类数据集")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET, help="本地 parquet 路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 YOLO 分类数据集目录")
    parser.add_argument("--download", action="store_true", help="缺少 parquet 时自动下载公开分片")
    parser.add_argument("--train-per-class", type=int, default=160, help="每类训练样本数")
    parser.add_argument("--val-per-class", type=int, default=40, help="每类验证样本数")
    args = parser.parse_args()

    parquet = download_parquet(args.parquet) if args.download else args.parquet
    output = prepare_dataset(
        parquet_path=parquet,
        output_dir=args.output,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
    )
    print(f"真实数据集已导出: {output}")


if __name__ == "__main__":
    main()
