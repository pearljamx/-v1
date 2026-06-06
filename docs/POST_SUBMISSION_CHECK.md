# 提交/打包后复查清单

本文用于最终提交或打包之后复查，确认交付包没有漏文件、没有漏模型，
也没有混入虚拟环境、缓存或运行产物。
提交包内文档索引见 `docs/DELIVERY_INDEX.md`。

## 1. 使用场景

- `git commit` 后复查。
- 压缩包生成后复查。
- 答辩前从提交包重新解压后的复查。
- 教师/助教验收前快速定位验收入口。

## 2. 提交后推荐复查命令

```bash
python scripts/pre_commit_check.py
python scripts/package_check.py
python scripts/submission_manifest.py
python scripts/archive_dry_run.py
python scripts/acceptance_check.py
python -m unittest discover -s tests -p "test_*.py" -v
```

API 快速复查：

```bash
curl http://127.0.0.1:5001/api/heartbeat
curl http://127.0.0.1:5001/api/models
curl http://127.0.0.1:5001/api/demo/samples
```

## 3. 必须存在的交付文档

```text
README.md
docs/DELIVERY_INDEX.md
docs/SUBMISSION_NOTE.md
docs/FINAL_FREEZE.md
docs/FINAL_ACCEPTANCE.md
docs/DEMO_SCRIPT.md
docs/TROUBLESHOOTING.md
docs/REPRODUCIBILITY_REPORT.md
docs/REQUIREMENT_TRACEABILITY.md
docs/SUBMISSION_MANIFEST.md
docs/ARCHIVE_DRY_RUN.md
docs/QUICK_START_CARD.md
docs/TEACHER_ACCEPTANCE_CHECKLIST.md
docs/POST_SUBMISSION_CHECK.md
```

## 4. 必须存在的脚本

```text
scripts/acceptance_check.py
scripts/package_check.py
scripts/repro_report.py
scripts/pre_commit_check.py
scripts/submission_manifest.py
scripts/archive_dry_run.py
yolo/train_real_datasets.py
```

## 5. 必须存在的模型

```text
models_data/yolo_driver_state.pt
models_data/yolo_drowsiness_cls.pt
```

- 两个模型均必须大于 1 MB。
- 它们是课程 YOLO 训练证明。
- 不要为了减小压缩包体积删除。

## 6. 不应出现在最终包中的内容

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

## 7. 通过标准

- `acceptance_check.py` 输出 `FAIL=0`。
- `package_check.py` 输出 `FAIL=0`。
- `archive_dry_run.py` 结论为 `[PASS]`。
- `submission_manifest.py` 结论为 `[PASS]`。
- unittest 全部通过。
- `/api/heartbeat`、`/api/models`、`/api/demo/samples` 可访问。
- 两个证明模型存在且大小合格。
