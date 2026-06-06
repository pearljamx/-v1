# 教师/助教验收清单

本文面向课程验收人员，用于快速确认项目是否满足课程要求、能否运行演示、
模型证明是否齐全，以及测试命令是否通过。更详细的要求映射见
`docs/REQUIREMENT_TRACEABILITY.md`。
提交包复查可查看 `docs/POST_SUBMISSION_CHECK.md`。
完整文档阅读顺序见 `docs/DELIVERY_INDEX.md`。

## 1. 项目基本信息

- 项目名称：基于计算机视觉的驾驶注意力检测系统。
- 技术栈：Python、OpenCV-python/contrib、Flask、Bootstrap、YOLO。
- 本地默认端口：`5001`。
- 入口地址：`http://127.0.0.1:5001`。
- 实时摄像头页：`http://127.0.0.1:5001/camera`。

## 2. 课程要求逐项验收

| 课程要求 | 是否覆盖 | 验收证据 | 验收命令/页面 |
| --- | --- | --- | --- |
| Python | 是 | `app.py`、`detectors/`、`processors/`、`web/`、`scripts/`、`yolo/` | `python scripts/acceptance_check.py` |
| OpenCV-python + contrib | 是 | `detectors/fatigue.py`、`detectors/head_pose.py`、`processors/frame_pipeline.py` | `python scripts/acceptance_check.py`，查看 OpenCV contrib 检查 |
| Flask 后端 | 是 | `app.py`、`web/routes_*.py` | `curl http://127.0.0.1:5001/api/heartbeat` |
| Bootstrap 前端 | 是 | `templates/`、`static/css/style.css`、`static/js/camera.js` | 打开 `http://127.0.0.1:5001` 和 `/camera` |
| 模块化结构 | 是 | `detectors/`、`processors/`、`web/`、`models/`、`yolo/` | `python scripts/package_check.py` |
| 图像检测 | 是 | 首页上传和 `/detect/auto` | 首页上传图片并查看结果页 |
| 视频检测 | 是 | `processors/video_processor.py`、上传/检测路由 | 上传视频或运行 unittest |
| 摄像头检测 | 是 | `/camera`、`/camera/frame`、实时仪表盘 | 打开 `http://127.0.0.1:5001/camera` |
| 疲劳检测 | 是 | EAR、眨眼率、MAR、头姿点头/低头 | 查看摄像头页指标或验收脚本 |
| 分心检测 | 是 | 手机/吸烟/饮食、视线偏移、转头、转身、手离方向盘 | 摄像头页演示模式或 unittest |
| 生理状态展示 | 是 | rPPG 心率估计与血压趋势字段 | 首页/结果页查看生理状态输出 |
| 至少使用 YOLO 训练一个分类或目标检测模型 | 是 | `models_data/yolo_driver_state.pt`、`models_data/yolo_drowsiness_cls.pt`、`yolo/train_real_datasets.py` | `python -m yolo.train_real_datasets --dry-run` |

## 3. 推荐验收顺序

1. 启动服务。
2. 打开首页。
3. 查看图像/视频检测入口。
4. 打开实时摄像头页。
5. 如果无摄像头，使用固定样例备用演示。
6. 运行一键验收脚本。

## 4. 关键命令

```bash
cd /Volumes/Data/02_课程/计算机视觉/实验/课程设计
source .venv/bin/activate
FLASK_PORT=5001 python app.py
python scripts/acceptance_check.py
python scripts/package_check.py
python scripts/archive_dry_run.py
python -m unittest discover -s tests -p "test_*.py" -v
```

API 快速检查：

```bash
curl http://127.0.0.1:5001/api/heartbeat
curl http://127.0.0.1:5001/api/models
curl http://127.0.0.1:5001/api/demo/samples
```

## 5. 必须存在的模型

```text
models_data/yolo_driver_state.pt
models_data/yolo_drowsiness_cls.pt
```

- 它们是课程 YOLO 训练证明。
- 必须纳入最终交付包。
- `yolo_steering_hand.pt` 是可选增强，不是课程验收必要项。

## 6. 不应打包内容

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

- `scripts/acceptance_check.py` 输出 `FAIL=0`。
- unittest 全部通过。
- `/api/heartbeat` 返回 `status=ok`。
- `/api/models` 中两个必要 YOLO 模型可用。
- `/api/demo/samples` 能返回固定样例。
- 两个 `.pt` 模型存在且大于 1 MB。
- 普通 `git status --short` 不出现运行副作用文件。
