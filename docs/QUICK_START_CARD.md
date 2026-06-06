# 驾驶注意力检测系统答辩快速启动卡

这张卡片用于答辩现场一分钟启动和备用演示。完整讲稿见
`docs/DEMO_SCRIPT.md`，最终验收见 `docs/FINAL_ACCEPTANCE.md`。
教师/助教逐项验收可查看 `docs/TEACHER_ACCEPTANCE_CHECKLIST.md`。
提交/打包后复查可查看 `docs/POST_SUBMISSION_CHECK.md`。
完整文档阅读顺序见 `docs/DELIVERY_INDEX.md`。

## 1. 一分钟启动命令

```bash
cd /Volumes/Data/02_课程/计算机视觉/实验/课程设计
source .venv/bin/activate
FLASK_PORT=5001 python app.py
```

## 2. 浏览器入口

```text
http://127.0.0.1:5001
http://127.0.0.1:5001/camera
http://127.0.0.1:5001/api/heartbeat
http://127.0.0.1:5001/api/models
```

## 3. 答辩前 3 条检查

```bash
python scripts/acceptance_check.py
python scripts/package_check.py
python scripts/archive_dry_run.py
```

## 4. 摄像头不可用时备用路径

- 打开 `/camera`。
- 使用“无摄像头备用演示”。
- 点击“固定样例”。
- 说明固定样例来自本地 `dataset/` 目录。

## 5. 必须带上的模型

```text
models_data/yolo_driver_state.pt
models_data/yolo_drowsiness_cls.pt
```

这两个文件是课程 YOLO 训练证明模型，必须纳入最终课程交付包。

## 6. 禁止打包项

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
