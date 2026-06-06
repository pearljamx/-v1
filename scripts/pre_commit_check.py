#!/usr/bin/env python3
"""
Read-only pre-commit checklist for the course delivery package.

The script prints recommended Git staging paths and forbidden runtime
artifacts. It never runs git add, git commit, git clean, training, downloads,
or deletion.
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DELIVERY_PATHS = [
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
    "web/routes_demo.py",
    "yolo/train_real_datasets.py",
    "models_data/yolo_driver_state.pt",
    "models_data/yolo_drowsiness_cls.pt",
]

PROOF_MODEL_PATHS = [
    "models_data/yolo_driver_state.pt",
    "models_data/yolo_drowsiness_cls.pt",
]

FORBIDDEN_EXACT_PATHS = [
    ".venv",
    "datasets",
    "uploads",
    "outputs",
    "runs",
    "models_data/detection_history.db",
    "models_data/detection_history.db-shm",
    "models_data/detection_history.db-wal",
]

FORBIDDEN_DISPLAY_PATHS = [
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
        print("驾驶注意力检测系统 - 提交前 Git 清单核验")
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
        stderr=subprocess.STDOUT,
        check=False,
    )


def normalize_status_path(line: str) -> str:
    path = line[3:].strip()
    if " -> " in path:
        path = path.rsplit(" -> ", 1)[-1]
    return path


def git_status_lines(*args: str) -> list[str]:
    proc = run_git(["status", *args, "--short"])
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def path_matches_forbidden(path: str) -> bool:
    normalized = path.rstrip("/")
    if normalized in FORBIDDEN_EXACT_PATHS:
        return True
    if any(normalized.startswith(item + "/") for item in FORBIDDEN_EXACT_PATHS):
        return True
    if "__pycache__/" in path or path.endswith("/__pycache__"):
        return True
    return fnmatch.fnmatch(path, "*.pyc") or path.endswith(".pyc")


def compact_paths(paths: list[str], limit: int = 18) -> str:
    if not paths:
        return "(none)"
    unique = list(dict.fromkeys(path.rstrip("/") for path in paths))
    if len(unique) <= limit:
        return " ".join(unique)
    shown = " ".join(unique[:limit])
    return f"{shown} ... (+{len(unique) - limit})"


def recommended_stage_paths(status_lines: list[str]) -> list[str]:
    paths: list[str] = []
    for line in status_lines:
        path = normalize_status_path(line)
        if path_matches_forbidden(path):
            continue
        paths.append(path.rstrip("/"))
    return list(dict.fromkeys(paths))


def delivery_candidates(status_lines: list[str]) -> list[str]:
    paths = [normalize_status_path(line) for line in status_lines if line.startswith("?? ")]
    return [
        path
        for path in paths
        if any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in DELIVERY_CANDIDATE_PREFIXES)
    ]


def check_required_delivery_paths(reporter: Reporter) -> None:
    missing = [path for path in REQUIRED_DELIVERY_PATHS if not (ROOT / path).exists()]
    if missing:
        reporter.fail("关键交付路径", "缺失: " + ", ".join(missing))
    else:
        reporter.pass_("关键交付路径", f"全部存在 ({len(REQUIRED_DELIVERY_PATHS)} 项)")


def check_proof_models(reporter: Reporter) -> None:
    failures: list[str] = []
    details: list[str] = []
    for rel in PROOF_MODEL_PATHS:
        path = ROOT / rel
        if not path.exists():
            failures.append(f"{rel} 缺失")
            continue
        size_mb = path.stat().st_size / 1024 / 1024
        if size_mb <= 1:
            failures.append(f"{rel} 文件过小 ({size_mb:.2f} MB)")
            continue
        details.append(f"{rel}={size_mb:.1f}MB")
    if failures:
        reporter.fail("课程 YOLO 证明模型", "; ".join(failures))
    else:
        reporter.pass_("课程 YOLO 证明模型", "; ".join(details) + "；必须纳入最终交付包")


def check_git_status(reporter: Reporter, status_lines: list[str]) -> None:
    if not status_lines:
        reporter.pass_("Git 当前状态摘要", "工作区无普通变更")
        return
    modified = [normalize_status_path(line) for line in status_lines if not line.startswith("?? ")]
    untracked = [normalize_status_path(line) for line in status_lines if line.startswith("?? ")]
    detail = f"tracked={len(modified)} untracked={len(untracked)}"
    if modified:
        detail += f"; tracked: {compact_paths(modified, limit=10)}"
    if untracked:
        detail += f"; untracked: {compact_paths(untracked, limit=10)}"
    reporter.warn("Git 当前状态摘要", detail)


def check_forbidden_worktree(reporter: Reporter, status_lines: list[str]) -> None:
    hits = [line for line in status_lines if path_matches_forbidden(normalize_status_path(line))]
    if hits:
        reporter.fail("运行副作用进入普通 Git 工作区", "发现: " + " | ".join(hits))
    else:
        reporter.pass_("运行副作用进入普通 Git 工作区", "普通 git status --short 未发现 forbidden 路径")


def check_ignored_runtime_artifacts(reporter: Reporter) -> None:
    ignored_lines = git_status_lines("--ignored")
    hits = [
        normalize_status_path(line)
        for line in ignored_lines
        if line.startswith("!! ") and path_matches_forbidden(normalize_status_path(line))
    ]
    if hits:
        reporter.warn(
            "ignored 运行产物",
            "本地存在但普通 git add 不会加入: " + compact_paths(hits, limit=10),
        )
    else:
        reporter.pass_("ignored 运行产物", "未发现 forbidden ignored 路径")


def print_recommended_commands(status_lines: list[str]) -> None:
    print("推荐人工核验命令:")
    print()
    paths = recommended_stage_paths(status_lines)
    if paths:
        print("```bash")
        print("git add " + compact_paths(paths, limit=80))
        print("```")
    else:
        print("```bash")
        print("# 当前没有普通工作区变更需要 git add")
        print("```")
    print()
    print("必须纳入最终课程交付包的 YOLO 证明模型:")
    for rel in PROOF_MODEL_PATHS:
        print(f"- {rel}")
    print()
    candidates = delivery_candidates(status_lines)
    if candidates:
        print("未跟踪交付候选，请人工确认纳入:")
        for path in candidates:
            print(f"- {path}")
        print()
    print("禁止加入的本地运行产物:")
    for item in FORBIDDEN_DISPLAY_PATHS:
        print(f"- {item}")
    print()
    print("不要使用 `git add -f` 加入上述 forbidden 运行产物。")
    print()


def main() -> int:
    reporter = Reporter()
    status_lines = git_status_lines()

    print_recommended_commands(status_lines)

    check_required_delivery_paths(reporter)
    check_proof_models(reporter)
    check_git_status(reporter, status_lines)
    check_forbidden_worktree(reporter, status_lines)
    check_ignored_runtime_artifacts(reporter)

    reporter.print_report()
    return 1 if reporter.has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
