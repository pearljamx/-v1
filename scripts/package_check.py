#!/usr/bin/env python3
"""
Final packaging checklist for the driving attention detection project.

The script is read-only: it does not delete files, run training, stage files,
or create archives. It reports what must be included in the course delivery
package and what must be excluded from the final package.
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DELIVERY_PATHS = [
    "README.md",
    "requirements.txt",
    "app.py",
    "config.py",
    "templates",
    "static",
    "web",
    "detectors",
    "processors",
    "tests",
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
    "yolo/hf_drowsiness_dataset.py",
    "yolo/train_drowsiness_cls.py",
    "yolo/train_handheld.py",
    "vendor",
    "models_data/yolo_driver_state.pt",
    "models_data/yolo_drowsiness_cls.pt",
]

REQUIRED_MODEL_PATHS = [
    "models_data/yolo_driver_state.pt",
    "models_data/yolo_drowsiness_cls.pt",
]

EXCLUDED_EXACT_PATHS = [
    ".venv",
    "datasets",
    "uploads",
    "outputs",
    "runs",
    "models_data/detection_history.db-shm",
    "models_data/detection_history.db-wal",
]

SOURCE_SCAN_DIRS = [
    "detectors",
    "processors",
    "web",
    "yolo",
    "scripts",
    "tests",
    "utils",
    "models",
]

DELIVERY_CANDIDATE_PREFIXES = [
    "docs/",
    "scripts/",
    "vendor/",
    "web/routes_demo.py",
    "yolo/hf_drowsiness_dataset.py",
    "yolo/train_drowsiness_cls.py",
    "yolo/train_real_datasets.py",
]


@dataclass
class CheckResult:
    status: str
    name: str
    detail: str


class Reporter:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def pass_(self, name: str, detail: str) -> None:
        self.results.append(CheckResult("PASS", name, detail))

    def warn(self, name: str, detail: str) -> None:
        self.results.append(CheckResult("WARN", name, detail))

    def fail(self, name: str, detail: str) -> None:
        self.results.append(CheckResult("FAIL", name, detail))

    def print_report(self) -> None:
        print("驾驶注意力检测系统 - 最终打包清单校验")
        print(f"项目目录: {ROOT}")
        print()
        for result in self.results:
            print(f"[{result.status}] {result.name}: {result.detail}")
        print()
        counts = {
            "PASS": sum(item.status == "PASS" for item in self.results),
            "WARN": sum(item.status == "WARN" for item in self.results),
            "FAIL": sum(item.status == "FAIL" for item in self.results),
        }
        print(f"汇总: PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']}")

    @property
    def has_failures(self) -> bool:
        return any(item.status == "FAIL" for item in self.results)


def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def normalize_status_path(line: str) -> str:
    path = line[3:].strip()
    if " -> " in path:
        path = path.rsplit(" -> ", 1)[-1]
    return path


def git_status_lines() -> list[str]:
    proc = run_git(["status", "--short"])
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def is_ignored(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    proc = run_git(["check-ignore", "--quiet", "--no-index", rel])
    return proc.returncode == 0


def path_matches_excluded_status(path: str) -> bool:
    normalized = path.rstrip("/")
    if normalized in EXCLUDED_EXACT_PATHS:
        return True
    if any(normalized.startswith(item + "/") for item in EXCLUDED_EXACT_PATHS):
        return True
    if "__pycache__/" in path or path.endswith("/__pycache__"):
        return True
    return fnmatch.fnmatch(path, "*.pyc") or path.endswith(".pyc")


def find_python_cache_paths() -> list[Path]:
    matches: list[Path] = []
    for dirname in SOURCE_SCAN_DIRS:
        base = ROOT / dirname
        if not base.exists():
            continue
        matches.extend(path for path in base.rglob("__pycache__") if path.is_dir())
        matches.extend(path for path in base.rglob("*.pyc") if path.is_file())
    return sorted(set(matches))


def format_rel(paths: list[Path], limit: int = 8) -> str:
    rels = [path.relative_to(ROOT).as_posix() for path in paths]
    if len(rels) <= limit:
        return ", ".join(rels)
    return ", ".join(rels[:limit]) + f", ... (+{len(rels) - limit})"


def check_required_delivery_paths(reporter: Reporter) -> None:
    missing = [path for path in REQUIRED_DELIVERY_PATHS if not (ROOT / path).exists()]
    if missing:
        reporter.fail("必须纳入交付包的路径", "缺失: " + ", ".join(missing))
    else:
        reporter.pass_("必须纳入交付包的路径", f"全部存在 ({len(REQUIRED_DELIVERY_PATHS)} 项)")


def check_required_models(reporter: Reporter) -> None:
    failures: list[str] = []
    details: list[str] = []
    for rel in REQUIRED_MODEL_PATHS:
        path = ROOT / rel
        if not path.exists():
            failures.append(f"{rel} 缺失")
            continue
        size_mb = path.stat().st_size / 1024 / 1024
        if size_mb <= 1:
            failures.append(f"{rel} 文件过小 ({size_mb:.2f} MB)")
            continue
        details.append(f"{rel}={size_mb:.1f}MB，应纳入最终课程交付包")
    if failures:
        reporter.fail("证明模型文件", "; ".join(failures))
    else:
        reporter.pass_("证明模型文件", "; ".join(details))


def check_excluded_artifacts(reporter: Reporter, status_lines: list[str]) -> None:
    status_hits = [
        line for line in status_lines if path_matches_excluded_status(normalize_status_path(line))
    ]
    if status_hits:
        reporter.fail("运行副作用进入普通 Git 工作区", "发现: " + " | ".join(status_hits))
    else:
        reporter.pass_("运行副作用进入普通 Git 工作区", "普通 git status --short 未发现排除项")

    existing: list[Path] = [ROOT / rel for rel in EXCLUDED_EXACT_PATHS if (ROOT / rel).exists()]
    existing.extend(find_python_cache_paths())
    if not existing:
        reporter.pass_("本地排除项", "未发现本地运行产物")
        return

    ignored = [path for path in existing if is_ignored(path)]
    not_ignored = [path for path in existing if path not in ignored]
    if ignored:
        reporter.warn("本地排除项", "本地存在但不应打包: " + format_rel(ignored))
    if not_ignored:
        reporter.warn("未确认忽略的本地排除项", "请确认不纳入交付包: " + format_rel(not_ignored))


def check_git_delivery_candidates(reporter: Reporter, status_lines: list[str]) -> None:
    untracked = [normalize_status_path(line) for line in status_lines if line.startswith("?? ")]
    candidates = [
        path
        for path in untracked
        if any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in DELIVERY_CANDIDATE_PREFIXES)
    ]
    if not candidates:
        reporter.pass_("未跟踪交付候选", "当前没有未跟踪交付候选")
        return

    git_add_hint = (
        "git add docs scripts vendor web/routes_demo.py "
        "yolo/hf_drowsiness_dataset.py yolo/train_drowsiness_cls.py yolo/train_real_datasets.py"
    )
    reporter.warn(
        "未跟踪交付候选",
        "提交前需人工纳入: " + ", ".join(candidates) + f"；建议命令: {git_add_hint}",
    )


def main() -> int:
    reporter = Reporter()
    status_lines = git_status_lines()

    check_required_delivery_paths(reporter)
    check_required_models(reporter)
    check_excluded_artifacts(reporter, status_lines)
    check_git_delivery_candidates(reporter, status_lines)

    reporter.print_report()
    return 1 if reporter.has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
