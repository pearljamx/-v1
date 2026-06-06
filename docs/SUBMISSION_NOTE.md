# 最终提交摘要与评审说明

本文可作为最终提交时给教师/助教的简短说明，帮助评审者快速了解项目、
运行入口、模型证明和推荐验收顺序。

## 1. 提交摘要

项目名称：基于计算机视觉的驾驶注意力检测系统。

本项目使用 Python、OpenCV-python/contrib、Flask、Bootstrap 和 YOLO，
实现图像检测、视频检测、实时摄像头检测、疲劳检测、分心检测和生理状态展示。
系统已提供答辩演示模式，并在没有摄像头时提供固定样例备用演示路径。

## 2. 运行入口

在项目根目录执行：

```bash
source .venv/bin/activate
FLASK_PORT=5001 python app.py
```

浏览器入口：

```text
http://127.0.0.1:5001
http://127.0.0.1:5001/camera
```

API 验收入口：

```text
http://127.0.0.1:5001/api/heartbeat
http://127.0.0.1:5001/api/models
http://127.0.0.1:5001/api/demo/samples
```

## 3. 课程模型证明

最终课程包必须包含：

```text
models_data/yolo_driver_state.pt
models_data/yolo_drowsiness_cls.pt
```

- 两个模型是课程 YOLO 训练证明。
- 两个模型必须包含在最终包中。
- `yolo_steering_hand.pt` 是可选增强模型，不作为课程必要证明。

## 4. 推荐先读文档

```text
docs/DELIVERY_INDEX.md
docs/QUICK_START_CARD.md
docs/TEACHER_ACCEPTANCE_CHECKLIST.md
docs/FINAL_ACCEPTANCE.md
docs/DEMO_SCRIPT.md
```

## 5. 推荐验收命令

```bash
python scripts/acceptance_check.py
python scripts/package_check.py
python scripts/archive_dry_run.py
python -m unittest discover -s tests -p "test_*.py" -v
```

## 6. 打包注意事项

最终提交包不要包含以下本地运行产物：

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
