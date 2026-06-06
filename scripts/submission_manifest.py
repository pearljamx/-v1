#!/usr/bin/env python3
"""
Generate the final course submission manifest.

The script is read-only except for writing the requested Markdown manifest.
It does not run training, download datasets, mutate Git state, stage files,
delete files, or package archives.
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
DEFAULT_OUTPUT = ROOT / "docs" / "SUBMISSION_MANIFEST.md"

REQUIRED_TOP_LEVEL_PATHS = [
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
    "models_data",
    "tests",
    "yolo",
]

PROOF_MODEL_PATHS = [
    "models_data/yolo_driver_state.pt",
    "models_data/yolo_drowsiness_cls.pt",
]

KEY_DOC_PATHS = [
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
]

KEY_SCRIPT_PATHS = [
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


@dataclass
class ManifestState:
    status_lines: list[str]
    required_path_failures: list[str]
    proof_model_failures: list[str]
    ordinary_excluded_hits: list[str]
    ignored_excluded_hits: list[str]

    @property
    def has_failures(self) -> bool:
        return bool(
            self.required_path_failures
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


def compact(items: list[str], limit: int = 12) -> str:
    if not items:
        return "(none)"
    unique = list(dict.fromkeys(items))
    if len(unique) <= limit:
        return ", ".join(unique)
    return ", ".join(unique[:limit]) + f", ... (+{len(unique) - limit})"


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def status_tag(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"


def path_line(
    rel_path: str,
    *,
    must_include: bool = False,
    generated_output_rel: str | None = None,
) -> str:
    path = ROOT / rel_path
    exists = path.exists() or rel_path == generated_output_rel
    suffix = "；必须纳入最终课程交付包" if must_include else ""
    return f"- {status_tag(exists)} `{rel_path}`：{'存在' if exists else '缺失'}{suffix}。"


def proof_model_line(rel_path: str) -> tuple[str, str | None]:
    path = ROOT / rel_path
    if not path.exists():
        return (
            f"- [FAIL] `{rel_path}`：缺失；必须纳入最终课程交付包。",
            f"{rel_path} 缺失",
        )
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


def scan_local_python_cache_paths() -> list[str]:
    matches: list[Path] = []
    for dirname in SOURCE_SCAN_DIRS:
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
    hits.extend(scan_local_python_cache_paths())
    return sorted(set(hits))


def collect_state() -> ManifestState:
    status_lines = git_status_lines()
    required_path_failures = [
        rel_path for rel_path in REQUIRED_TOP_LEVEL_PATHS if not (ROOT / rel_path).exists()
    ]
    proof_model_failures = [
        failure
        for _, failure in (proof_model_line(rel_path) for rel_path in PROOF_MODEL_PATHS)
        if failure
    ]
    ordinary_excluded_hits = [
        line for line in status_lines if path_matches_excluded(normalize_status_path(line))
    ]
    ignored_excluded_hits = ignored_excluded_status_paths()
    return ManifestState(
        status_lines=status_lines,
        required_path_failures=required_path_failures,
        proof_model_failures=proof_model_failures,
        ordinary_excluded_hits=ordinary_excluded_hits,
        ignored_excluded_hits=ignored_excluded_hits,
    )


def fenced(text: str, language: str = "text") -> str:
    return f"```{language}\n{text}\n```"


def excluded_status_lines(state: ManifestState) -> list[str]:
    lines = [f"- `{item}`" for item in EXCLUDED_DISPLAY_PATHS]
    if state.ordinary_excluded_hits:
        lines.append("")
        lines.append(
            "- [FAIL] 普通 `git status --short` 中发现不应纳入提交包的路径："
            + compact(state.ordinary_excluded_hits)
            + "。"
        )
    else:
        lines.append("")
        lines.append("- [PASS] 普通 `git status --short` 未发现本地运行产物进入工作区。")

    if state.ignored_excluded_hits:
        lines.append(
            "- [WARN] 本地存在但不应打包："
            + compact(state.ignored_excluded_hits)
            + "。"
        )
    else:
        lines.append("- [PASS] 未发现本地 ignored 运行产物。")
    return lines


def build_manifest(state: ManifestState, generated_output_rel: str | None = None) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    proof_lines = [proof_model_line(rel_path)[0] for rel_path in PROOF_MODEL_PATHS]
    required_lines = [path_line(rel_path) for rel_path in REQUIRED_TOP_LEVEL_PATHS]
    doc_lines = [
        path_line(rel_path, generated_output_rel=generated_output_rel)
        for rel_path in KEY_DOC_PATHS
    ]
    script_lines = [path_line(rel_path) for rel_path in KEY_SCRIPT_PATHS]
    excluded_lines = excluded_status_lines(state)

    status = git_output(["status", "--short"])
    diff_stat = git_output(["diff", "--stat"])
    verdict = (
        "[FAIL] 当前提交包存在缺失或运行副作用进入普通 Git 工作区。"
        if state.has_failures
        else "[PASS] 当前提交包具备课程验收完整性；仍需按清单人工确认最终压缩包内容。"
    )

    return "\n".join(
        [
            "# 驾驶注意力检测系统最终提交包 Manifest",
            "",
            "## 1. 项目提交包摘要",
            "",
            "- 项目名称：基于计算机视觉的驾驶注意力检测系统。",
            f"- 生成时间：{generated_at}",
            f"- 当前工作目录：`{ROOT}`",
            f"- Python 版本：`{sys.version.splitlines()[0]}`",
            f"- Python 可执行文件：`{sys.executable}`",
            f"- 平台信息：`{platform.platform()}`",
            f"- 机器架构：`{platform.machine()}`",
            f"- Manifest 结论：{verdict}",
            "",
            "## 2. 必须包含的顶层目录/文件",
            "",
            *required_lines,
            "",
            "## 3. 必须包含的课程证明模型",
            "",
            *proof_lines,
            "",
            "## 4. 关键文档 Manifest",
            "",
            *doc_lines,
            "",
            "## 5. 关键脚本 Manifest",
            "",
            *script_lines,
            "",
            "## 6. 不应纳入提交包的路径提醒",
            "",
            "最终课程交付包不应包含以下本地运行产物：",
            "",
            *excluded_lines,
            "",
            "## 7. Git 状态摘要",
            "",
            "### git status --short",
            "",
            fenced(status),
            "",
            "### git diff --stat",
            "",
            fenced(diff_stat),
            "",
            "## 8. 推荐提交前命令",
            "",
            "```bash",
            "python scripts/submission_manifest.py",
            "python scripts/archive_dry_run.py",
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
    parser = argparse.ArgumentParser(description="Generate the final submission manifest.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Markdown output path. Defaults to docs/SUBMISSION_MANIFEST.md.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the generated manifest content after writing it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    state = collect_state()
    rel_output = output.relative_to(ROOT).as_posix() if output.is_relative_to(ROOT) else None
    manifest = build_manifest(state, generated_output_rel=rel_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(manifest, encoding="utf-8")
    display_output = rel_output if rel_output else output
    print(f"[PASS] 提交包 Manifest 已生成: {display_output}")
    if state.has_failures:
        if state.required_path_failures:
            print("[FAIL] 必须包含路径缺失: " + compact(state.required_path_failures))
        if state.proof_model_failures:
            print("[FAIL] 课程证明模型异常: " + compact(state.proof_model_failures))
        if state.ordinary_excluded_hits:
            print("[FAIL] 运行副作用进入普通 Git 工作区: " + compact(state.ordinary_excluded_hits))
    if state.ignored_excluded_hits:
        print("[WARN] 本地存在但不应打包: " + compact(state.ignored_excluded_hits))
    if args.stdout:
        print()
        print(manifest)
    return 1 if state.has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
