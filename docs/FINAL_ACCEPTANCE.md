# 驾驶注意力检测系统最终交付检查

生成日期：2026-06-06

## 1. 课程要求覆盖情况

课程要求逐项映射见 `docs/REQUIREMENT_TRACEABILITY.md`。

| 课程要求 | 覆盖状态 | 项目证据 |
| --- | --- | --- |
| Python 编程语言 | 已覆盖 | `app.py`、`detectors/`、`processors/`、`web/` |
| OpenCV-python | 已覆盖 | 图像/视频处理、几何计算、PnP 头姿、rPPG ROI |
| OpenCV-python-Contrib | 已覆盖 | `scripts/acceptance_check.py` 检查 `cv2.face` |
| Flask 后端 | 已覆盖 | `app.py`、`web/routes_*.py` |
| Bootstrap 前端 | 已覆盖 | `templates/`、`static/css/style.css`、Bootstrap 图标与按钮风格 |
| 模块化程序设计 | 已覆盖 | `detectors/`、`processors/`、`web/`、`utils/`、`models/`、`yolo/` |
| 图像文件检测 | 已覆盖 | 首页上传、`/upload`、`/detect/auto`、`ImageProcessor` |
| 视频文件检测 | 已覆盖 | 视频上传、逐帧处理、`VideoProcessor` |
| 摄像头实时检测 | 已覆盖 | `/camera`、`/camera/frame`、`RealtimeProcessor` |
| 疲劳驾驶检测 | 已覆盖 | EAR、MAR、眨眼率、哈欠、点头/低头 |
| 分心驾驶检测 | 已覆盖 | 手机/吸烟/饮食、视线偏移、转头、转身、手离方向盘 |
| 生理状态展示 | 已覆盖 | rPPG 心率估计、血压趋势展示入口 |
| 至少一个 YOLO 训练模型 | 已覆盖 | `models_data/yolo_driver_state.pt`、`models_data/yolo_drowsiness_cls.pt` |

## 2. 当前可证明模型

| 模型文件 | 用途 | 证明意义 |
| --- | --- | --- |
| `models_data/yolo_driver_state.pt` | 驾驶状态目标检测 | 使用真实驾驶状态数据集训练得到，用于证明 YOLO 检测训练流程 |
| `models_data/yolo_drowsiness_cls.pt` | 驾驶疲劳二分类 | 使用 Hugging Face 公开真实数据集训练得到，用于证明 YOLO 分类训练流程 |
| `models_data/yolov8n-pose.pt` | 人体姿态关键点 | 支撑手腕关键点、虚拟方向盘 ROI、转身检测 |
| `models_data/yolo_handheld.pt` | 手持物品检测 | 作为手机、吸烟、饮食等分心行为检测模型 |

可选模型 `models_data/yolo_steering_hand.pt` 当前不是必需项。没有该模型时，系统使用 YOLO pose 手腕关键点 + 虚拟方向盘 ROI 完成手离把判断。

## 3. 关键演示入口

| 演示项 | 地址或操作 | 说明 |
| --- | --- | --- |
| 首页 | `http://127.0.0.1:5001` | 上传入口、结果入口、项目概览 |
| 图像检测 | 首页上传图片后自动检测 | 展示疲劳/分心/生理状态报告 |
| 视频检测 | 首页上传视频后自动检测 | 展示逐帧处理与报告 |
| 实时摄像头检测 | `http://127.0.0.1:5001/camera` | 展示实时仪表盘、虚拟方向盘、告警面板 |
| 演示模式 | 摄像头页勾选“演示模式” | 只缩短持续时间阈值，便于课堂快速触发 |
| 无摄像头固定样例 | 摄像头页“固定样例”按钮 | 调用 `/api/demo/samples`，无摄像头时走上传备用演示 |

## 4. 阈值对照

| 检测项 | 默认现实阈值 | 演示模式阈值 | 说明 |
| --- | --- | --- | --- |
| 双手离开方向盘 | 5s 后 danger | 1.5s 后 danger | 虚拟方向盘 ROI 和手腕关键点逻辑不变 |
| 单手离开方向盘 | 8s 后 warning | 2s 后 warning | 单手短暂操作允许更长容忍 |
| 持续转头 | yaw > 35° 持续 2s | yaw > 35° 持续 1s | yaw >= 55° 时升级 danger |
| 躯干转身 | 角度 > 45° 持续 2s | 角度 > 45° 持续 1s | 肩部关键点和躯干角度逻辑不变 |
| 视线偏离 | 角度 > 30° 持续 2s | 不变 | 避免正常扫视误报 |
| 闭眼 | EAR < 0.2 持续 3s | 不变 | 符合疲劳检测要求 |
| 哈欠 | MAR > 0.5 持续 3s | 不变 | 符合疲劳检测要求 |

## 5. 最终验收命令

在项目目录执行：

```bash
cd /Volumes/Data/02_课程/计算机视觉/实验/课程设计
source .venv/bin/activate
```

启动服务：

```bash
FLASK_PORT=5001 python app.py
```

另开终端执行：

```bash
python -m py_compile $(rg --files -g '*.py')
python -m unittest discover -s tests -p "test_*.py" -v
python -m yolo.train_real_datasets --dry-run
python scripts/acceptance_check.py
```

API 验证地址：

```text
http://127.0.0.1:5001/api/heartbeat
http://127.0.0.1:5001/api/models
http://127.0.0.1:5001/api/demo/samples
```

## 6. 最终打包清单

最终课程交付包必须包含：

- 源码目录：`app.py`、`config.py`、`detectors/`、`processors/`、`web/`、`templates/`、`static/`、`models/`、`utils/`。
- 说明与验收材料：`README.md`、`docs/`、`scripts/`、`tests/`。
- 最终交付总索引：`docs/DELIVERY_INDEX.md`，建议答辩、验收和打包前优先查看。
- 最终提交简短说明：`docs/SUBMISSION_NOTE.md`，用于给教师/助教快速说明运行入口和模型证明。
- 最终冻结确认：`docs/FINAL_FREEZE.md`，用于确认项目已进入提交/打包状态，不再建议继续改功能、算法、模型或 UI。
- 教师/助教快速验收清单：`docs/TEACHER_ACCEPTANCE_CHECKLIST.md`，用于不阅读完整 README 时快速核对课程要求。
- 答辩现场快速启动卡片：`docs/QUICK_START_CARD.md`。
- 提交/打包后复查清单：`docs/POST_SUBMISSION_CHECK.md`。
- 可复现性报告：`docs/REPRODUCIBILITY_REPORT.md`，由 `python scripts/repro_report.py` 生成。
- 课程要求映射：`docs/REQUIREMENT_TRACEABILITY.md`，用于逐项核对课程要求到代码、接口和模型证据。
- 最终提交包 Manifest：`docs/SUBMISSION_MANIFEST.md`，由 `python scripts/submission_manifest.py` 生成，用于核对目录、模型文件和排除项。
- 压缩包 dry-run 预检：`docs/ARCHIVE_DRY_RUN.md`，由 `python scripts/archive_dry_run.py` 生成，只输出应包含/排除项，不真正创建压缩包。
- 提交前 Git 清单核验：`scripts/pre_commit_check.py`，用于输出推荐 `git add` 清单和禁止加入项。
- YOLO 真实数据集训练入口：`yolo/train_real_datasets.py`、`yolo/hf_drowsiness_dataset.py`、`yolo/train_drowsiness_cls.py`、`yolo/train_handheld.py`。
- macOS 兼容文件：`vendor/`。
- 课程训练证明模型：`models_data/yolo_driver_state.pt`、`models_data/yolo_drowsiness_cls.pt`。

最终课程交付包不应包含：

- `.venv/`
- `datasets/`
- `uploads/`
- `outputs/`
- `runs/`
- SQLite WAL/SHM 临时文件，例如 `models_data/detection_history.db-shm`、`models_data/detection_history.db-wal`
- Python 缓存，例如 `__pycache__/`、`*.pyc`

推荐最终检查命令：

```bash
python scripts/repro_report.py
python scripts/submission_manifest.py
python scripts/archive_dry_run.py
python scripts/pre_commit_check.py
python scripts/package_check.py
python scripts/acceptance_check.py
git status --short
```

注意：模型文件通常被 `.gitignore` 排除，但 `models_data/yolo_driver_state.pt`
和 `models_data/yolo_drowsiness_cls.pt` 必须随最终课程包提供。

## 7. 当前已验证结果

- `py_compile`：通过。
- `unittest`：26 项测试通过。
- `yolo.train_real_datasets --dry-run`：可运行，训练入口和模型证明状态可检查。
- `scripts/acceptance_check.py`：8 项检查通过，包含模型、API、demo samples 和 unittest。
- Chrome 页面验证：`/camera` 可打开，演示模式可切换，固定样例可展开 6 张图片。

## 8. 风险说明

- 当前环境没有可用摄像头，因此真机摄像头中的手腕点位、虚拟方向盘覆盖效果、转头动态提示仍建议答辩前人工确认一次。
- `yolo_steering_hand.pt` 是可选模型，不影响当前交付；答辩时建议表述为“基于 YOLO pose 手腕关键点与虚拟方向盘 ROI 判断手部离把”。
- 不建议答辩前重新训练模型或下载大数据集，避免破坏当前稳定环境。
- 不应提交或打包运行副作用文件，例如 `dataset/*/labels.cache`、`datasets/`、`uploads/`、`outputs/`、`runs/`、SQLite WAL/SHM、`__pycache__/`、`*.pyc`。
