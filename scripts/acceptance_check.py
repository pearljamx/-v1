#!/usr/bin/env python3
"""
课程设计本地验收脚本。

用于答辩前快速确认项目具备运行、演示、测试和模型证明能力。
建议先启动 Flask 服务:

    FLASK_PORT=5001 python app.py

再运行:

    python scripts/acceptance_check.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:5001"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


class AcceptanceReporter:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def pass_(self, name: str, detail: str) -> None:
        self.results.append(CheckResult(name, "PASS", detail))

    def warn(self, name: str, detail: str) -> None:
        self.results.append(CheckResult(name, "WARN", detail))

    def fail(self, name: str, detail: str) -> None:
        self.results.append(CheckResult(name, "FAIL", detail))

    def print_report(self) -> None:
        print("驾驶注意力检测系统 - 本地验收报告")
        print(f"项目目录: {ROOT}")
        print(f"Python: {sys.executable}")
        print(f"时间戳: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        for result in self.results:
            print(f"[{result.status}] {result.name}: {result.detail}")
        print()
        passed = sum(1 for item in self.results if item.status == "PASS")
        warned = sum(1 for item in self.results if item.status == "WARN")
        failed = sum(1 for item in self.results if item.status == "FAIL")
        print(f"汇总: PASS={passed} WARN={warned} FAIL={failed}")

    @property
    def has_failures(self) -> bool:
        return any(item.status == "FAIL" for item in self.results)


def run_check(reporter: AcceptanceReporter, name: str, fn: Callable[[], str | None]) -> None:
    try:
        detail = fn() or "ok"
    except Warning as exc:
        reporter.warn(name, str(exc))
    except Exception as exc:  # noqa: BLE001 - acceptance output should stay readable.
        reporter.fail(name, str(exc))
    else:
        reporter.pass_(name, detail)


def check_project_files() -> str:
    required = [
        "app.py",
        "config.py",
        "requirements.txt",
        "README.md",
        "templates/index.html",
        "templates/camera.html",
        "static/js/camera.js",
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
        "scripts/package_check.py",
        "scripts/repro_report.py",
        "scripts/pre_commit_check.py",
        "scripts/submission_manifest.py",
        "scripts/archive_dry_run.py",
        "web/routes_main.py",
        "web/routes_upload.py",
        "web/routes_detect.py",
        "detectors/fatigue.py",
        "detectors/distraction.py",
        "processors/frame_pipeline.py",
        "yolo/train_driver_state.py",
        "yolo/train_drowsiness_cls.py",
        "yolo/train_real_datasets.py",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    if missing:
        raise RuntimeError("缺少关键文件: " + ", ".join(missing))
    return f"关键文件齐全 ({len(required)} 项)"


def check_imports() -> str:
    modules = {
        "cv2": "opencv-contrib-python",
        "flask": "flask",
        "ultralytics": "ultralytics",
        "face_recognition": "face-recognition",
    }
    versions: list[str] = []
    for module_name, package_name in modules.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", UserWarning)
            module = importlib.import_module(module_name)
        try:
            version = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            version = getattr(module, "__version__", "unknown")
        versions.append(f"{module_name}={version}")
    return "; ".join(versions)


def check_opencv_contrib() -> str:
    import cv2

    if not hasattr(cv2, "face"):
        raise RuntimeError("未检测到 cv2.face，请确认安装 opencv-contrib-python")
    return "cv2.face 可用，OpenCV contrib 已生效"


def check_models() -> str:
    models = [
        ROOT / "models_data" / "yolo_driver_state.pt",
        ROOT / "models_data" / "yolo_drowsiness_cls.pt",
    ]
    missing_or_empty = [str(path.relative_to(ROOT)) for path in models if not path.exists() or path.stat().st_size == 0]
    if missing_or_empty:
        raise RuntimeError("模型文件缺失或为空: " + ", ".join(missing_or_empty))
    sizes = [f"{path.name}={path.stat().st_size / 1024 / 1024:.1f}MB" for path in models]
    return "; ".join(sizes)


def check_camera_delivery() -> str:
    camera_html = (ROOT / "templates" / "camera.html").read_text(encoding="utf-8")
    camera_js = (ROOT / "static" / "js" / "camera.js").read_text(encoding="utf-8")
    routes_demo_py = (ROOT / "web" / "routes_demo.py").read_text(encoding="utf-8")
    distraction_py = (ROOT / "detectors" / "distraction.py").read_text(encoding="utf-8")
    frame_pipeline_py = (ROOT / "processors" / "frame_pipeline.py").read_text(encoding="utf-8")

    required_markers = [
        ("camera.html", camera_html, "metric-hand-state"),
        ("camera.html", camera_html, "metric-head-turn-state"),
        ("camera.html", camera_html, "enable-camera-distraction"),
        ("camera.html", camera_html, "enable-demo-mode"),
        ("camera.html", camera_html, "camera-demo-fallback"),
        ("camera.js", camera_js, "drawVirtualSteeringWheel"),
        ("camera.js", camera_js, "hand_status"),
        ("camera.js", camera_js, "demo_mode"),
        ("camera.js", camera_js, "loadDemoSamples"),
        ("camera.js", camera_js, "/api/demo/samples"),
        ("routes_demo.py", routes_demo_py, "/api/demo/samples"),
        ("distraction.py", distraction_py, "VIRTUAL_WHEEL_CENTER"),
        ("distraction.py", distraction_py, "SINGLE_HAND_OFF_WHEEL_DURATION"),
        ("distraction.py", distraction_py, "DEMO_HAND_OFF_WHEEL_DURATION"),
        ("frame_pipeline.py", frame_pipeline_py, "virtual_wheel"),
        ("frame_pipeline.py", frame_pipeline_py, "head_turning"),
    ]
    missing = [f"{file}:{marker}" for file, content, marker in required_markers if marker not in content]
    if missing:
        raise RuntimeError("摄像头交付标记缺失: " + ", ".join(missing))
    return "虚拟方向盘、手部/转头状态、演示模式和无摄像头备用样例标记齐全"


def check_real_dataset_entry() -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "yolo.train_real_datasets", "--dry-run"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    output = proc.stdout.strip()
    if proc.returncode != 0:
        tail = "\n".join(output.splitlines()[-20:])
        raise RuntimeError(f"真实数据集入口 dry-run 失败，exit={proc.returncode}\n{tail}")
    pass_lines = [line for line in output.splitlines() if line.startswith("[PASS]")]
    warn_lines = [line for line in output.splitlines() if line.startswith("[WARN]")]
    return f"dry-run 可运行，PASS={len(pass_lines)} WARN={len(warn_lines)}"


def request_json(url: str, timeout: float) -> dict:
    with urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def check_health(base_url: str, timeout: float) -> str:
    try:
        heartbeat = request_json(f"{base_url}/api/heartbeat", timeout)
        models = request_json(f"{base_url}/api/models", timeout)
        demo_samples = request_json(f"{base_url}/api/demo/samples", timeout)
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法访问 {base_url}，请先启动服务: FLASK_PORT=5001 python app.py ({exc})") from exc

    if heartbeat.get("status") != "ok":
        raise RuntimeError(f"/api/heartbeat 状态异常: {heartbeat}")
    if "samples" not in demo_samples or "upload_url" not in demo_samples:
        raise RuntimeError(f"/api/demo/samples 返回异常: {demo_samples}")

    model_entries = models.get("models", models)
    available = []
    if isinstance(model_entries, dict):
        available = [name for name, item in model_entries.items() if isinstance(item, dict) and item.get("available")]
    if not available:
        raise RuntimeError(f"/api/models 未返回可用模型: {models}")
    return (
        f"heartbeat=ok; available_models={', '.join(sorted(available))}; "
        f"demo_samples={len(demo_samples.get('samples', []))}"
    )


def check_unittest() -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )
    output = proc.stdout.strip()
    if proc.returncode != 0:
        tail = "\n".join(output.splitlines()[-30:])
        raise RuntimeError(f"unittest 失败，exit={proc.returncode}\n{tail}")

    summary = "测试通过"
    for line in reversed(output.splitlines()):
        if line.startswith("Ran ") or line.strip() == "OK":
            summary = line.strip() if summary == "测试通过" else f"{line.strip()}; {summary}"
        if line.strip().startswith("Ran "):
            break
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="驾驶注意力检测系统本地验收脚本")
    parser.add_argument("--base-url", default=os.environ.get("ACCEPTANCE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--timeout", type=float, default=5.0, help="API 请求超时时间，单位秒")
    parser.add_argument("--skip-tests", action="store_true", help="跳过 unittest，仅做环境/API/模型检查")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reporter = AcceptanceReporter()

    run_check(reporter, "项目关键文件", check_project_files)
    run_check(reporter, "Python 依赖导入", check_imports)
    run_check(reporter, "OpenCV contrib", check_opencv_contrib)
    run_check(reporter, "YOLO 模型文件", check_models)
    run_check(reporter, "摄像头实时演示增强", check_camera_delivery)
    run_check(reporter, "真实数据集训练入口", check_real_dataset_entry)
    run_check(reporter, "Flask API 健康检查", lambda: check_health(args.base_url, args.timeout))
    if args.skip_tests:
        reporter.warn("unittest", "已按 --skip-tests 跳过")
    else:
        run_check(reporter, "unittest", check_unittest)

    reporter.print_report()
    return 1 if reporter.has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
