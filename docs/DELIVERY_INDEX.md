# 最终交付文档与检查脚本总索引

本文用于答辩、验收、提交和打包前快速判断“先看哪个文件、先运行哪个脚本”。
它只整理交付入口，不替代各个具体文档和检查脚本。

## 1. 文档阅读顺序

建议按用途从上到下阅读：

1. `docs/QUICK_START_CARD.md`：答辩现场一分钟启动。
2. `docs/SUBMISSION_NOTE.md`：最终提交给教师/助教的简短说明。
3. `docs/TEACHER_ACCEPTANCE_CHECKLIST.md`：教师/助教快速验收。
4. `docs/FINAL_ACCEPTANCE.md`：最终课程要求与验收证据。
5. `docs/DEMO_SCRIPT.md`：5 到 8 分钟答辩讲稿。
6. `docs/REQUIREMENT_TRACEABILITY.md`：课程要求到代码、接口和模型的映射。
7. `docs/TROUBLESHOOTING.md`：现场故障排查。
8. `docs/REPRODUCIBILITY_REPORT.md`：本机环境、依赖版本和模型大小。
9. `docs/SUBMISSION_MANIFEST.md`：最终提交包 Manifest。
10. `docs/ARCHIVE_DRY_RUN.md`：压缩包 dry-run 预检。
11. `docs/POST_SUBMISSION_CHECK.md`：提交/打包后复查。
12. `docs/FINAL_FREEZE.md`：最终冻结确认，确认进入提交/打包状态。

## 2. 脚本执行顺序

最终提交前建议按以下顺序执行：

```bash
python scripts/pre_commit_check.py
python scripts/package_check.py
python scripts/submission_manifest.py
python scripts/archive_dry_run.py
python scripts/repro_report.py
python scripts/acceptance_check.py
python -m unittest discover -s tests -p "test_*.py" -v
```

脚本用途：

- `scripts/pre_commit_check.py`：提交前 Git 清单核验。
- `scripts/package_check.py`：最终包必须/禁止项检查。
- `scripts/submission_manifest.py`：生成提交包 Manifest。
- `scripts/archive_dry_run.py`：生成压缩包 dry-run 报告。
- `scripts/repro_report.py`：生成环境复现报告。
- `scripts/acceptance_check.py`：课程交付一键验收。

## 3. 必须纳入最终包的关键项

```text
README.md
requirements.txt
docs/
scripts/
vendor/
web/
templates/
static/
detectors/
processors/
models/
tests/
yolo/
models_data/yolo_driver_state.pt
models_data/yolo_drowsiness_cls.pt
```

两个 `.pt` 文件是课程 YOLO 训练证明模型，即使被 `.gitignore` 排除，
也必须随最终课程交付包提供。

## 4. 不应纳入最终包的运行产物

```text
.venv/
datasets/
uploads/
outputs/
runs/
models_data/detection_history.db-shm
models_data/detection_history.db-wal
__pycache__/
*.pyc
```

## 5. 最终状态判断

最终交付前应满足：

- `scripts/acceptance_check.py` 输出 `FAIL=0`。
- `scripts/package_check.py` 输出 `FAIL=0`。
- `scripts/pre_commit_check.py` 普通 Git 工作区未出现 forbidden 路径。
- `models_data/yolo_driver_state.pt` 和 `models_data/yolo_drowsiness_cls.pt` 存在且均大于 1 MB。
- unittest 全部通过。
- 如果没有摄像头，使用 `/camera` 页面中的固定样例备用演示。
