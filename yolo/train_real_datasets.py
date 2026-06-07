#!/usr/bin/env python3
"""
真实公开数据集 YOLO 训练统一入口。

本脚本用于课程答辩前说明并检查真实数据集训练链路，不会默认下载
Roboflow 或 Hugging Face 大文件。常用命令:

    python -m yolo.train_real_datasets --list
    python -m yolo.train_real_datasets --dry-run
    python -m yolo.train_real_datasets --run drowsiness-cls --prepare --quick
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


@dataclass(frozen=True)
class DatasetEntry:
    key: str
    title: str
    source: str
    license_note: str
    prepare_command: str
    train_command: str
    output_model: str


DATASETS = [
    DatasetEntry(
        key="driver-state",
        title="Roboflow Universe - Driver fatigue and distraction",
        source="https://universe.roboflow.com/mds-workspace-arqn1/driver-fatigue-and-distraction-bned4",
        license_note="Roboflow 页面标注 CC BY 4.0；自动下载需 ROBOFLOW_API_KEY，或手动导出 YOLOv8 zip。",
        prepare_command="python -m yolo.roboflow_driver_dataset",
        train_command="python -m yolo.train_driver_state --quick",
        output_model="models_data/yolo_driver_state.pt",
    ),
    DatasetEntry(
        key="mendeley-distraction-cls",
        title="Mendeley Data - Novel Driver Distractions Dataset With Low Lighting Support",
        source="https://data.mendeley.com/datasets/ykmr99nrsg/2",
        license_note="CC BY-NC 3.0；页面 Download All 后用 --zip 导入为 YOLO classification 目录。",
        prepare_command="python -m yolo.mendeley_distraction_dataset --zip /path/to/mendeley.zip",
        train_command="python -m yolo.train_mendeley_distraction_cls --quick",
        output_model="models_data/driver_distraction_cls.pt",
    ),
    DatasetEntry(
        key="drowsiness-cls",
        title="Hugging Face - n7i5x9/driver-drowsiness-dataset",
        source="https://huggingface.co/datasets/n7i5x9/driver-drowsiness-dataset",
        license_note="公开 Hugging Face 图像分类数据集；脚本抽样转换为 YOLO classification 目录。",
        prepare_command="python -m yolo.hf_drowsiness_dataset --download",
        train_command="python -m yolo.train_drowsiness_cls --quick",
        output_model="models_data/yolo_drowsiness_cls.pt",
    ),
    DatasetEntry(
        key="handheld",
        title="课程手持物/分心检测 YOLO 数据配置",
        source="dataset/data.yaml 或 Roboflow driver-state 导入后的 data.yaml",
        license_note="用于手机、抽烟、饮食等检测；建议优先由 Roboflow 真实数据集导入。",
        prepare_command="python -m yolo.roboflow_driver_dataset --zip /path/to/roboflow.zip",
        train_command="python -m yolo.train_handheld --quick",
        output_model="models_data/yolo_handheld_quick.pt",
    ),
]


def print_dataset_list() -> None:
    print("真实数据集与训练入口")
    print("=" * 72)
    for item in DATASETS:
        print(f"[{item.key}] {item.title}")
        print(f"  来源: {item.source}")
        print(f"  授权/说明: {item.license_note}")
        print(f"  准备: {item.prepare_command}")
        print(f"  训练: {item.train_command}")
        print(f"  产物: {item.output_model}")
        print()


def check_path(path: str) -> tuple[bool, str]:
    full = BASE_DIR / path
    if full.exists():
        return True, f"{path} 存在"
    return False, f"{path} 不存在"


def dry_run() -> int:
    checks: list[tuple[str, bool, str]] = []

    for script in [
        "yolo/roboflow_driver_dataset.py",
        "yolo/hf_drowsiness_dataset.py",
        "yolo/mendeley_distraction_dataset.py",
        "yolo/train_driver_state.py",
        "yolo/train_drowsiness_cls.py",
        "yolo/train_mendeley_distraction_cls.py",
        "yolo/train_handheld.py",
    ]:
        ok, detail = check_path(script)
        checks.append(("训练脚本", ok, detail))

    for data_config in [
        "dataset/data.yaml",
        "dataset/roboflow_driver_state/data.yaml",
        "datasets/hf_driver_drowsiness_yolo_cls",
        "datasets/mendeley_driver_distraction_yolo_cls",
    ]:
        ok, detail = check_path(data_config)
        checks.append(("数据配置", ok, detail))

    ok, detail = check_path("models_data")
    checks.append(("模型目录", ok, detail))

    for item in DATASETS:
        ok, detail = check_path(item.output_model)
        checks.append(("模型产物", ok, detail))

    try:
        importlib.import_module("ultralytics")
        checks.append(("依赖导入", True, "ultralytics 可导入"))
    except Exception as exc:  # noqa: BLE001 - dry-run should stay readable.
        checks.append(("依赖导入", False, f"ultralytics 不可导入: {exc}"))

    print("真实数据集训练 dry-run")
    print("=" * 72)
    hard_fail = False
    for group, ok, detail in checks:
        status = "PASS" if ok else "WARN"
        print(f"[{status}] {group}: {detail}")
        if group in {"训练脚本", "模型目录", "依赖导入"} and not ok:
            hard_fail = True
    print("=" * 72)
    print("说明: 数据配置缺失为 WARN，因为本入口默认不下载大文件；按 --list 中命令准备后即可训练。")
    return 1 if hard_fail else 0


def run_command(args: list[str]) -> int:
    print("+ " + " ".join(args))
    return subprocess.call(args, cwd=BASE_DIR)


def run_training(target: str, quick: bool, prepare: bool) -> int:
    if target == "driver-state":
        if prepare:
            code = run_command([PYTHON, "-m", "yolo.roboflow_driver_dataset"])
            if code != 0:
                return code
        cmd = [PYTHON, "-m", "yolo.train_driver_state"]
    elif target == "drowsiness-cls":
        if prepare:
            code = run_command([PYTHON, "-m", "yolo.hf_drowsiness_dataset", "--download"])
            if code != 0:
                return code
        cmd = [PYTHON, "-m", "yolo.train_drowsiness_cls"]
    elif target == "mendeley-distraction-cls":
        if prepare:
            print(
                "Mendeley API requires browser authorization in this environment. "
                "Download the zip from the dataset page, then run "
                "python -m yolo.mendeley_distraction_dataset --zip <zip>."
            )
            return 2
        cmd = [PYTHON, "-m", "yolo.train_mendeley_distraction_cls"]
    elif target == "handheld":
        cmd = [PYTHON, "-m", "yolo.train_handheld"]
    else:
        raise ValueError(f"未知训练目标: {target}")

    if quick:
        cmd.append("--quick")
    return run_command(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="真实公开数据集 YOLO 训练统一入口")
    parser.add_argument("--list", action="store_true", help="列出真实数据集来源、训练脚本和模型产物")
    parser.add_argument("--dry-run", action="store_true", help="只检查脚本/依赖/模型路径，不下载不训练")
    parser.add_argument("--run", choices=[item.key for item in DATASETS], help="执行指定训练任务")
    parser.add_argument("--quick", action="store_true", help="使用轻量训练参数")
    parser.add_argument("--prepare", action="store_true", help="训练前先执行对应数据准备脚本，可能需要网络或 API key")
    return parser.parse_args()


def main() -> int:
    os.chdir(BASE_DIR)
    args = parse_args()

    if args.list:
        print_dataset_list()
        return 0
    if args.dry_run:
        return dry_run()
    if args.run:
        return run_training(args.run, args.quick, args.prepare)

    print_dataset_list()
    print("下一步建议: python -m yolo.train_real_datasets --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
