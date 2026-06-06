#!/usr/bin/env python3
"""
Generate a dry-run report for the final course archive.

The script is read-only except for writing the requested Markdown report. It
does not create zip/tar files, delete files, mutate Git state, download data,
or train models.
"""

from __future__ import annotations

import argparse
import fnmatch
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "ARCHIVE_DRY_RUN.md"

ARCHIVE_INCLUDE_PATHS = [
    "README.md",
    "requirements.txt",
    "app.py",
    "config.py",
    "dataset/data.yaml",
    "docs",
    "docs/DELIVERY_INDEX.md",
    "docs/SUBMISSION_NOTE.md",
    "docs/FINAL_FREEZE.md",
    "docs/TEACHER_ACCEPTANCE_CHECKLIST.md",
    "docs/QUICK_START_CARD.md",
    "docs/POST_SUBMISSION_CHECK.md",
    "scripts",
    "vendor",
    "web",
    "templates",
    "static",
    "detectors",
    "processors",
    "models",
    "models_data/yolo_driver_state.pt",
    "models_data/yolo_drowsiness_cls.pt",
    "tests",
    "yolo",
]

PROOF_MODEL_PATHS = [
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

EXCLUDED_DISPLAY_PATHS = [
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

PYTHON_CACHE_SCAN_DIRS = [
    "detectors",
    "processors",
    "web",
    "yolo",
    "scripts",
    "tests",
    "utils",
    "models",
]

ARCHIVE_COMMAND_PATHS = [
    "README.md",
    "requirements.txt",
    "app.py",
    "config.py",
    "dataset/data.yaml",
    "docs",
    "scripts",
    "vendor",
    "web",
    "templates",
    "static",
    "detectors",
    "processors",
    "models",
    "models_data/yolo_driver_state.pt",
    "models_data/yolo_drowsiness_cls.pt",
    "tests",
    "yolo",
]

ZIP_EXCLUDES = [
    ".venv/*",
    "datasets/*",
    "uploads/*",
    "outputs/*",
    "runs/*",
    "models_data/detection_history.db-shm",
    "models_data/detection_history.db-wal",
    "*/__pycache__/*",
    "*.pyc",
]


@dataclass
class ArchiveState:
    include_failures: list[str]
    proof_model_failures: list[str]
    ordinary_excluded_hits: list[str]
    ignored_excluded_hits: list[str]
    status_lines: list[str]

    @property
    def has_failures(self) -> bool:
        return bool(
            self.include_failures
            or self.proof_model_failures
            or self.ordinary_excluded_hits
        )


def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def git_output(args: list[str]) -> str:
    proc = run_git(args)
    output = proc.stdout.rstrip()
    if proc.returncode != 0:
        return f"(git {' '.join(args)} failed: {output or 'no output'})"
    return output or "(no output)"


def git_status_lines(*args: str) -> list[str]:
    proc = run_git(["status", *args, "--short"])
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def normalize_status_path(line: str) -> str:
    path = line[3:].strip()
    if " -> " in path:
        path = path.rsplit(" -> ", 1)[-1]
    return path


def path_matches_excluded(path: str) -> bool:
    normalized = path.rstrip("/")
    if normalized in EXCLUDED_EXACT_PATHS:
        return True
    if any(normalized.startswith(item + "/") for item in EXCLUDED_EXACT_PATHS):
        return True
    if "__pycache__/" in path or path.endswith("/__pycache__"):
        return True
    return fnmatch.fnmatch(path, "*.pyc") or path.endswith(".pyc")


def is_ignored(rel_path: str) -> bool:
    proc = run_git(["check-ignore", "--quiet", "--no-index", rel_path])
    return proc.returncode == 0


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / 1024 / 1024:.1f} MB"
    return f"{num_bytes / 1024 / 1024 / 1024:.2f} GB"


def compact(items: list[str], limit: int = 12) -> str:
    if not items:
        return "(none)"
    unique = list(dict.fromkeys(items))
    if len(unique) <= limit:
        return ", ".join(unique)
    return ", ".join(unique[:limit]) + f", ... (+{len(unique) - limit})"


def scan_python_cache_paths() -> list[str]:
    matches: list[Path] = []
    for dirname in PYTHON_CACHE_SCAN_DIRS:
        base = ROOT / dirname
        if not base.exists():
            continue
        matches.extend(path for path in base.rglob("__pycache__") if path.is_dir())
        matches.extend(path for path in base.rglob("*.pyc") if path.is_file())
    return sorted({path.relative_to(ROOT).as_posix() for path in matches})


def ignored_excluded_status_paths() -> list[str]:
    ignored_lines = git_status_lines("--ignored")
    hits = [
        normalize_status_path(line)
        for line in ignored_lines
        if line.startswith("!! ") and path_matches_excluded(normalize_status_path(line))
    ]
    hits.extend(scan_python_cache_paths())
    return sorted(set(hits))


def include_path_line(rel_path: str) -> tuple[str, str | None]:
    path = ROOT / rel_path
    if not path.exists():
        return f"- [FAIL] `{rel_path}`：缺失。", rel_path
    kind = "目录" if path.is_dir() else "文件"
    return f"- [PASS] `{rel_path}`：{kind}，{format_size(path_size_bytes(path))}。", None


def proof_model_line(rel_path: str) -> tuple[str, str | None]:
    path = ROOT / rel_path
    if not path.exists():
        return f"- [FAIL] `{rel_path}`：缺失；必须纳入最终课程交付包。", f"{rel_path} 缺失"
    size_mb = file_size_mb(path)
    if size_mb <= 1:
        return (
            f"- [FAIL] `{rel_path}`：{size_mb:.2f} MB，文件小于 1 MB；必须纳入最终课程交付包。",
            f"{rel_path} 文件过小 ({size_mb:.2f} MB)",
        )
    return (
        f"- [PASS] `{rel_path}`：存在，{size_mb:.1f} MB；必须纳入最终课程交付包。",
        None,
    )


def excluded_path_lines(state: ArchiveState) -> list[str]:
    lines: list[str] = []
    ordinary_paths = {normalize_status_path(line) for line in state.ordinary_excluded_hits}
    ignored_paths = set(state.ignored_excluded_hits)

    for rel_path in EXCLUDED_DISPLAY_PATHS:
        normalized = rel_path.rstrip("/")
        if rel_path in {"__pycache__/", "*.pyc"}:
            cache_hits = [path for path in ignored_paths if "__pycache__" in path or path.endswith(".pyc")]
            if cache_hits:
                lines.append(
                    f"- [WARN] `{rel_path}`：本地存在，但不应压缩/提交；示例: {compact(cache_hits, 6)}。"
                )
            else:
                lines.append(f"- [PASS] `{rel_path}`：未发现。")
            continue

        if any(path == normalized or path.startswith(normalized + "/") for path in ordinary_paths):
            lines.append(f"- [FAIL] `{rel_path}`：出现在普通 `git status --short` 中。")
        elif (ROOT / normalized).exists():
            ignored = is_ignored(normalized)
            if ignored:
                lines.append(f"- [WARN] `{rel_path}`：本地存在，但不应压缩/提交；已被 `.gitignore` 忽略。")
            else:
                lines.append(f"- [WARN] `{rel_path}`：本地存在，请确认最终压缩包排除。")
        else:
            lines.append(f"- [PASS] `{rel_path}`：未发现。")
    return lines


def suggested_zip_command() -> str:
    include_args = " ".join(ARCHIVE_COMMAND_PATHS)
    exclude_args = " ".join(f"-x '{item}'" for item in ZIP_EXCLUDES)
    return f"zip -r ../driving_attention_submission.zip {include_args} {exclude_args}"


def collect_state() -> ArchiveState:
    status_lines = git_status_lines()
    include_failures = [
        rel_path for rel_path in ARCHIVE_INCLUDE_PATHS if not (ROOT / rel_path).exists()
    ]
    proof_model_failures = [
        failure
        for _, failure in (proof_model_line(rel_path) for rel_path in PROOF_MODEL_PATHS)
        if failure
    ]
    ordinary_excluded_hits = [
        line for line in status_lines if path_matches_excluded(normalize_status_path(line))
    ]
    return ArchiveState(
        include_failures=include_failures,
        proof_model_failures=proof_model_failures,
        ordinary_excluded_hits=ordinary_excluded_hits,
        ignored_excluded_hits=ignored_excluded_status_paths(),
        status_lines=status_lines,
    )


def fenced(text: str, language: str = "text") -> str:
    return f"```{language}\n{text}\n```"


def build_report(state: ArchiveState) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    include_lines = [include_path_line(rel_path)[0] for rel_path in ARCHIVE_INCLUDE_PATHS]
    proof_lines = [proof_model_line(rel_path)[0] for rel_path in PROOF_MODEL_PATHS]
    excluded_lines = excluded_path_lines(state)
    verdict = (
        "[FAIL] 当前压缩包预检发现缺失项或运行产物进入普通 Git 工作区。"
        if state.has_failures
        else "[PASS] 当前压缩包预检通过；请按报告人工确认最终压缩包内容。"
    )

    return "\n".join(
        [
            "# 驾驶注意力检测系统最终压缩包 dry-run 预检",
            "",
            "## 1. 预检摘要",
            "",
            "- 说明：本报告只做 dry-run 预检，不会真正创建 zip/tar 压缩包。",
            f"- 生成时间：{generated_at}",
            f"- 当前工作目录：`{ROOT}`",
            f"- Python 版本：`{sys.version.splitlines()[0]}`",
            f"- 平台信息：`{platform.platform()}`",
            f"- 预检结论：{verdict}",
            "",
            "## 2. 建议压缩包包含项",
            "",
            *include_lines,
            "",
            "## 3. 课程证明模型检查",
            "",
            *proof_lines,
            "",
            "## 4. 不建议压缩项审计",
            "",
            *excluded_lines,
            "",
            "## 5. 建议压缩命令说明",
            "",
            "以下命令仅供人工参考，本脚本不会执行：",
            "",
            fenced(suggested_zip_command(), "bash"),
            "",
            "## 6. Git 状态摘要",
            "",
            "### git status --short",
            "",
            fenced(git_output(["status", "--short"])),
            "",
            "### git diff --stat",
            "",
            fenced(git_output(["diff", "--stat"])),
            "",
            "## 7. 推荐提交前检查顺序",
            "",
            "```bash",
            "python scripts/archive_dry_run.py",
            "python scripts/submission_manifest.py",
            "python scripts/pre_commit_check.py",
            "python scripts/package_check.py",
            "python scripts/acceptance_check.py",
            "python -m unittest discover -s tests -p \"test_*.py\" -v",
            "git status --short",
            "```",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a dry-run report for the final archive.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Markdown output path. Defaults to docs/ARCHIVE_DRY_RUN.md.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the generated report content after writing it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    state = collect_state()
    report = build_report(state)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    display_output = output.relative_to(ROOT).as_posix() if output.is_relative_to(ROOT) else output
    print(f"[PASS] 压缩包 dry-run 预检报告已生成: {display_output}")
    if state.has_failures:
        if state.include_failures:
            print("[FAIL] 建议压缩包包含项缺失: " + compact(state.include_failures))
        if state.proof_model_failures:
            print("[FAIL] 课程证明模型异常: " + compact(state.proof_model_failures))
        if state.ordinary_excluded_hits:
            print("[FAIL] 运行副作用进入普通 Git 工作区: " + compact(state.ordinary_excluded_hits))
    if state.ignored_excluded_hits:
        print("[WARN] 本地存在但不应压缩/提交: " + compact(state.ignored_excluded_hits))
    if args.stdout:
        print()
        print(report)
    return 1 if state.has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
