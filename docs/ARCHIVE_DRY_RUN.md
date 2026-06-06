# 驾驶注意力检测系统最终压缩包 dry-run 预检

## 1. 预检摘要

- 说明：本报告只做 dry-run 预检，不会真正创建 zip/tar 压缩包。
- 生成时间：2026-06-06 20:16:01
- 当前工作目录：`/Volumes/Data/02_课程/计算机视觉/实验/课程设计`
- Python 版本：`3.9.6 (default, Apr 30 2025, 02:07:17) `
- 平台信息：`macOS-15.3.1-arm64-arm-64bit`
- 预检结论：[PASS] 当前压缩包预检通过；请按报告人工确认最终压缩包内容。

## 2. 建议压缩包包含项

- [PASS] `README.md`：文件，10.9 KB。
- [PASS] `requirements.txt`：文件，809 B。
- [PASS] `app.py`：文件，7.8 KB。
- [PASS] `config.py`：文件，7.0 KB。
- [PASS] `dataset/data.yaml`：文件，292 B。
- [PASS] `docs`：目录，57.3 KB。
- [PASS] `docs/DELIVERY_INDEX.md`：文件，2.8 KB。
- [PASS] `docs/SUBMISSION_NOTE.md`：文件，1.8 KB。
- [PASS] `docs/FINAL_FREEZE.md`：文件，3.0 KB。
- [PASS] `docs/TEACHER_ACCEPTANCE_CHECKLIST.md`：文件，3.8 KB。
- [PASS] `docs/QUICK_START_CARD.md`：文件，1.4 KB。
- [PASS] `docs/POST_SUBMISSION_CHECK.md`：文件，2.2 KB。
- [PASS] `scripts`：目录，61.1 KB。
- [PASS] `vendor`：目录，214 B。
- [PASS] `web`：目录，73.1 KB。
- [PASS] `templates`：目录，55.0 KB。
- [PASS] `static`：目录，119.5 KB。
- [PASS] `detectors`：目录，224.9 KB。
- [PASS] `processors`：目录，64.8 KB。
- [PASS] `models`：目录，43.9 KB。
- [PASS] `models_data/yolo_driver_state.pt`：文件，5.9 MB。
- [PASS] `models_data/yolo_drowsiness_cls.pt`：文件，2.8 MB。
- [PASS] `tests`：目录，54.6 KB。
- [PASS] `yolo`：目录，106.7 KB。

## 3. 课程证明模型检查

- [PASS] `models_data/yolo_driver_state.pt`：存在，5.9 MB；必须纳入最终课程交付包。
- [PASS] `models_data/yolo_drowsiness_cls.pt`：存在，2.8 MB；必须纳入最终课程交付包。

## 4. 不建议压缩项审计

- [WARN] `.venv/`：本地存在，但不应压缩/提交；已被 `.gitignore` 忽略。
- [WARN] `datasets/`：本地存在，但不应压缩/提交；已被 `.gitignore` 忽略。
- [WARN] `uploads/`：本地存在，但不应压缩/提交；已被 `.gitignore` 忽略。
- [WARN] `outputs/`：本地存在，但不应压缩/提交；已被 `.gitignore` 忽略。
- [WARN] `runs/`：本地存在，但不应压缩/提交；已被 `.gitignore` 忽略。
- [WARN] `models_data/detection_history.db-shm`：本地存在，但不应压缩/提交；已被 `.gitignore` 忽略。
- [WARN] `models_data/detection_history.db-wal`：本地存在，但不应压缩/提交；已被 `.gitignore` 忽略。
- [WARN] `__pycache__/`：本地存在，但不应压缩/提交；示例: yolo/__pycache__/train_handheld.cpython-39.pyc, yolo/__pycache__, detectors/__pycache__/head_pose.cpython-39.pyc, models/__pycache__, processors/__pycache__/video_processor.cpython-39.pyc, utils/__pycache__/signal_processing.cpython-39.pyc, ... (+37)。
- [WARN] `*.pyc`：本地存在，但不应压缩/提交；示例: yolo/__pycache__/train_handheld.cpython-39.pyc, yolo/__pycache__, detectors/__pycache__/head_pose.cpython-39.pyc, models/__pycache__, processors/__pycache__/video_processor.cpython-39.pyc, utils/__pycache__/signal_processing.cpython-39.pyc, ... (+37)。

## 5. 建议压缩命令说明

以下命令仅供人工参考，本脚本不会执行：

```bash
zip -r ../driving_attention_submission.zip README.md requirements.txt app.py config.py dataset/data.yaml docs scripts vendor web templates static detectors processors models models_data/yolo_driver_state.pt models_data/yolo_drowsiness_cls.pt tests yolo -x '.venv/*' -x 'datasets/*' -x 'uploads/*' -x 'outputs/*' -x 'runs/*' -x 'models_data/detection_history.db-shm' -x 'models_data/detection_history.db-wal' -x '*/__pycache__/*' -x '*.pyc'
```

## 6. Git 状态摘要

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

## 7. 推荐提交前检查顺序

```bash
python scripts/archive_dry_run.py
python scripts/submission_manifest.py
python scripts/pre_commit_check.py
python scripts/package_check.py
python scripts/acceptance_check.py
python -m unittest discover -s tests -p "test_*.py" -v
git status --short
```
