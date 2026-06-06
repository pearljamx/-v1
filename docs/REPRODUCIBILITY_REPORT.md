# 驾驶注意力检测系统可复现性报告

## 1. 项目基础信息

- 项目名称：基于计算机视觉的驾驶注意力检测系统。
- 生成时间：2026-06-06 20:16:02
- 当前工作目录：`/Volumes/Data/02_课程/计算机视觉/实验/课程设计`
- Python 版本：`3.9.6 (default, Apr 30 2025, 02:07:17) `
- Python 可执行文件：`/Volumes/Data/02_课程/计算机视觉/实验/课程设计/.venv/bin/python`
- 平台信息：`macOS-15.3.1-arm64-arm-64bit`
- 机器架构：`arm64`

## 2. 关键依赖版本

- `cv2`：4.13.0.92
- `flask`：3.1.3
- `ultralytics`：8.4.60
- `numpy`：2.0.2
- `face_recognition`：1.3.0

## 3. 关键模型文件

- `models_data/yolo_driver_state.pt`：存在，5.9 MB；必须纳入最终课程交付包。
- `models_data/yolo_drowsiness_cls.pt`：存在，2.8 MB；必须纳入最终课程交付包。

## 4. 关键文档与验收脚本

- `README.md`：存在。
- `docs/DELIVERY_INDEX.md`：存在。
- `docs/SUBMISSION_NOTE.md`：存在。
- `docs/FINAL_FREEZE.md`：存在。
- `docs/FINAL_ACCEPTANCE.md`：存在。
- `docs/TEACHER_ACCEPTANCE_CHECKLIST.md`：存在。
- `docs/QUICK_START_CARD.md`：存在。
- `docs/POST_SUBMISSION_CHECK.md`：存在。
- `docs/DEMO_SCRIPT.md`：存在。
- `docs/TROUBLESHOOTING.md`：存在。
- `docs/REPRODUCIBILITY_REPORT.md`：存在。
- `docs/REQUIREMENT_TRACEABILITY.md`：存在。
- `docs/SUBMISSION_MANIFEST.md`：存在。
- `docs/ARCHIVE_DRY_RUN.md`：存在。
- `scripts/acceptance_check.py`：存在。
- `scripts/package_check.py`：存在。
- `scripts/repro_report.py`：存在。
- `scripts/pre_commit_check.py`：存在。
- `scripts/submission_manifest.py`：存在。
- `scripts/archive_dry_run.py`：存在。
- `yolo/train_real_datasets.py`：存在。

## 5. Git 状态摘要

### git status --short

```text
 M .gitignore
 M README.md
 M config.py
 M dataset/data.yaml
 M detectors/distraction.py
 M detectors/head_pose.py
 M processors/frame_pipeline.py
 M processors/realtime_processor.py
 M requirements.txt
 M static/css/style.css
 M static/js/camera.js
 M templates/camera.html
 M tests/test_performance_stability.py
 M web/__init__.py
 M web/routes_camera.py
 M web/routes_main.py
 M yolo/train_handheld.py
?? docs/
?? scripts/
?? vendor/
?? web/routes_demo.py
?? yolo/hf_drowsiness_dataset.py
?? yolo/train_drowsiness_cls.py
?? yolo/train_real_datasets.py
```

### git diff --stat

```text
 .gitignore                          |   1 +
 README.md                           | 193 ++++++++++++++++-
 config.py                           |  22 +-
 dataset/data.yaml                   |   2 +-
 detectors/distraction.py            | 158 +++++++++++---
 detectors/head_pose.py              |  52 ++++-
 processors/frame_pipeline.py        |  58 +++++-
 processors/realtime_processor.py    |   7 +-
 requirements.txt                    |   7 +-
 static/css/style.css                | 405 +++++++++++++++++++++++++++++++++++-
 static/js/camera.js                 | 246 +++++++++++++++++++++-
 templates/camera.html               | 253 ++++++++++++++--------
 tests/test_performance_stability.py | 106 +++++++++-
 web/__init__.py                     |   1 +
 web/routes_camera.py                |  34 ++-
 web/routes_main.py                  |  48 ++++-
 yolo/train_handheld.py              |   7 +
 17 files changed, 1451 insertions(+), 149 deletions(-)
```

## 6. 推荐验收命令

```bash
python scripts/repro_report.py
python scripts/submission_manifest.py
python scripts/archive_dry_run.py
python scripts/package_check.py
python scripts/acceptance_check.py
python -m unittest discover -s tests -p "test_*.py" -v
python -m yolo.train_real_datasets --dry-run
git status --short
```

## 7. 打包排除项提醒

最终课程交付包不应包含以下本地运行产物：

- `.venv/`
- `datasets/`
- `uploads/`
- `outputs/`
- `runs/`
- `models_data/detection_history.db-shm`
- `models_data/detection_history.db-wal`
- `__pycache__/`
- `*.pyc`

两个课程证明模型虽然通常被 `.gitignore` 排除，但必须随最终课程包提供：

- `models_data/yolo_driver_state.pt`
- `models_data/yolo_drowsiness_cls.pt`
