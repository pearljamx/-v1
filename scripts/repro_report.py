#!/usr/bin/env python3
"""
Generate a local reproducibility report for the course delivery package.

This script is read-only except for writing the requested Markdown report. It
does not train models, download datasets, delete files, or run Git mutations.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "REPRODUCIBILITY_REPORT.md"

DEPENDENCIES = {
    "cv2": "opencv-contrib-python",
    "flask": "flask",
    "ultralytics": "ultralytics",
    "numpy": "numpy",
    "face_recognition": "face-recognition",
}

MODEL_PATHS = [
    "models_data/yolo_driver_state.pt",
    "models_data/yolo_drowsiness_cls.pt",
]

DELIVERY_FILES = [
    "README.md",
    "docs/DELIVERY_INDEX.md",
    "docs/SUBMISSION_NOTE.md",
    "docs/FINAL_FREEZE.md",
    "docs/FINAL_ACCEPTANCE.md",
    "docs/TEACHER_ACCEPTANCE_CHECKLIST.md",
    "docs/QUICK_START_CARD.md",
    "docs/POST_SUBMISSION_CHECK.md",
    "docs/DEMO_SCRIPT.md",
    "docs/TROUBLESHOOTING.md",
    "docs/REPRODUCIBILITY_REPORT.md",
    "docs/REQUIREMENT_TRACEABILITY.md",
    "docs/SUBMISSION_MANIFEST.md",
    "docs/ARCHIVE_DRY_RUN.md",
    "scripts/acceptance_check.py",
    "scripts/package_check.py",
    "scripts/repro_report.py",
    "scripts/pre_commit_check.py",
    "scripts/submission_manifest.py",
    "scripts/archive_dry_run.py",
    "yolo/train_real_datasets.py",
]

EXCLUDE_REMINDERS = [
    ".venv/",
    "datasets/",
    "uploads/",
    "outputs/",
    "runs/",
    "models_data/detection_history.db-shm",
    "models_data/detection_history.db-wal",
    "__pycache__/",
    "*.pyc",
]


def run_git(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = proc.stdout.rstrip()
    return output or "(no output)"


def dependency_version(module_name: str, package_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - report should stay readable.
        return "不可用"

    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def model_line(rel_path: str) -> str:
    path = ROOT / rel_path
    if not path.exists():
        return f"- `{rel_path}`：缺失；必须纳入最终课程交付包。"
    size_mb = path.stat().st_size / 1024 / 1024
    return f"- `{rel_path}`：存在，{size_mb:.1f} MB；必须纳入最终课程交付包。"


def delivery_file_line(rel_path: str) -> str:
    exists = (ROOT / rel_path).exists()
    return f"- `{rel_path}`：{'存在' if exists else '缺失'}。"


def fenced(text: str) -> str:
    return f"```text\n{text}\n```"


def build_report() -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    dependency_lines = [
        f"- `{module}`：{dependency_version(module, package)}"
        for module, package in DEPENDENCIES.items()
    ]
    model_lines = [model_line(path) for path in MODEL_PATHS]
    delivery_lines = [delivery_file_line(path) for path in DELIVERY_FILES]
    exclude_lines = [f"- `{item}`" for item in EXCLUDE_REMINDERS]

    status = run_git(["status", "--short"])
    diff_stat = run_git(["diff", "--stat"])

    return "\n".join(
        [
            "# 驾驶注意力检测系统可复现性报告",
            "",
            "## 1. 项目基础信息",
            "",
            "- 项目名称：基于计算机视觉的驾驶注意力检测系统。",
            f"- 生成时间：{generated_at}",
            f"- 当前工作目录：`{ROOT}`",
            f"- Python 版本：`{sys.version.splitlines()[0]}`",
            f"- Python 可执行文件：`{sys.executable}`",
            f"- 平台信息：`{platform.platform()}`",
            f"- 机器架构：`{platform.machine()}`",
            "",
            "## 2. 关键依赖版本",
            "",
            *dependency_lines,
            "",
            "## 3. 关键模型文件",
            "",
            *model_lines,
            "",
            "## 4. 关键文档与验收脚本",
            "",
            *delivery_lines,
            "",
            "## 5. Git 状态摘要",
            "",
            "### git status --short",
            "",
            fenced(status),
            "",
            "### git diff --stat",
            "",
            fenced(diff_stat),
            "",
            "## 6. 推荐验收命令",
            "",
            "```bash",
            "python scripts/repro_report.py",
            "python scripts/submission_manifest.py",
            "python scripts/archive_dry_run.py",
            "python scripts/package_check.py",
            "python scripts/acceptance_check.py",
            "python -m unittest discover -s tests -p \"test_*.py\" -v",
            "python -m yolo.train_real_datasets --dry-run",
            "git status --short",
            "```",
            "",
            "## 7. 打包排除项提醒",
            "",
            "最终课程交付包不应包含以下本地运行产物：",
            "",
            *exclude_lines,
            "",
            "两个课程证明模型虽然通常被 `.gitignore` 排除，但必须随最终课程包提供：",
            "",
            "- `models_data/yolo_driver_state.pt`",
            "- `models_data/yolo_drowsiness_cls.pt`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a reproducibility report.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Markdown report path. Defaults to docs/REPRODUCIBILITY_REPORT.md.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(), encoding="utf-8")
    print(f"[PASS] 可复现性报告已生成: {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
