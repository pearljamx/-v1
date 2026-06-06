# 驾驶注意力检测系统最终提交包 Manifest

## 1. 项目提交包摘要

- 项目名称：基于计算机视觉的驾驶注意力检测系统。
- 生成时间：2026-06-06 20:16:01
- 当前工作目录：`/Volumes/Data/02_课程/计算机视觉/实验/课程设计`
- Python 版本：`3.9.6 (default, Apr 30 2025, 02:07:17) `
- Python 可执行文件：`/Volumes/Data/02_课程/计算机视觉/实验/课程设计/.venv/bin/python`
- 平台信息：`macOS-15.3.1-arm64-arm-64bit`
- 机器架构：`arm64`
- Manifest 结论：[PASS] 当前提交包具备课程验收完整性；仍需按清单人工确认最终压缩包内容。

## 2. 必须包含的顶层目录/文件

- [PASS] `README.md`：存在。
- [PASS] `requirements.txt`：存在。
- [PASS] `app.py`：存在。
- [PASS] `config.py`：存在。
- [PASS] `dataset/data.yaml`：存在。
- [PASS] `docs`：存在。
- [PASS] `scripts`：存在。
- [PASS] `vendor`：存在。
- [PASS] `web`：存在。
- [PASS] `templates`：存在。
- [PASS] `static`：存在。
- [PASS] `detectors`：存在。
- [PASS] `processors`：存在。
- [PASS] `models`：存在。
- [PASS] `models_data`：存在。
- [PASS] `tests`：存在。
- [PASS] `yolo`：存在。

## 3. 必须包含的课程证明模型

- [PASS] `models_data/yolo_driver_state.pt`：存在，5.9 MB；必须纳入最终课程交付包。
- [PASS] `models_data/yolo_drowsiness_cls.pt`：存在，2.8 MB；必须纳入最终课程交付包。

## 4. 关键文档 Manifest

- [PASS] `docs/DELIVERY_INDEX.md`：存在。
- [PASS] `docs/SUBMISSION_NOTE.md`：存在。
- [PASS] `docs/FINAL_FREEZE.md`：存在。
- [PASS] `docs/FINAL_ACCEPTANCE.md`：存在。
- [PASS] `docs/TEACHER_ACCEPTANCE_CHECKLIST.md`：存在。
- [PASS] `docs/QUICK_START_CARD.md`：存在。
- [PASS] `docs/POST_SUBMISSION_CHECK.md`：存在。
- [PASS] `docs/DEMO_SCRIPT.md`：存在。
- [PASS] `docs/TROUBLESHOOTING.md`：存在。
- [PASS] `docs/REPRODUCIBILITY_REPORT.md`：存在。
- [PASS] `docs/REQUIREMENT_TRACEABILITY.md`：存在。
- [PASS] `docs/SUBMISSION_MANIFEST.md`：存在。
- [PASS] `docs/ARCHIVE_DRY_RUN.md`：存在。

## 5. 关键脚本 Manifest

- [PASS] `scripts/acceptance_check.py`：存在。
- [PASS] `scripts/package_check.py`：存在。
- [PASS] `scripts/repro_report.py`：存在。
- [PASS] `scripts/pre_commit_check.py`：存在。
- [PASS] `scripts/submission_manifest.py`：存在。
- [PASS] `scripts/archive_dry_run.py`：存在。
- [PASS] `yolo/train_real_datasets.py`：存在。
- [PASS] `yolo/hf_drowsiness_dataset.py`：存在。
- [PASS] `yolo/train_drowsiness_cls.py`：存在。
- [PASS] `yolo/train_handheld.py`：存在。

## 6. 不应纳入提交包的路径提醒

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

- [PASS] 普通 `git status --short` 未发现本地运行产物进入工作区。
- [WARN] 本地存在但不应打包：.venv/, datasets/, detectors/__pycache__, detectors/__pycache__/__init__.cpython-39.pyc, detectors/__pycache__/distraction.cpython-39.pyc, detectors/__pycache__/face_detector.cpython-39.pyc, detectors/__pycache__/face_recognizer.cpython-39.pyc, detectors/__pycache__/fatigue.cpython-39.pyc, detectors/__pycache__/gaze.cpython-39.pyc, detectors/__pycache__/head_pose.cpython-39.pyc, detectors/__pycache__/physiological.cpython-39.pyc, models/__pycache__, ... (+41)。

## 7. Git 状态摘要

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

## 8. 推荐提交前命令

```bash
python scripts/submission_manifest.py
python scripts/archive_dry_run.py
python scripts/pre_commit_check.py
python scripts/package_check.py
python scripts/acceptance_check.py
python -m unittest discover -s tests -p "test_*.py" -v
git status --short
```
