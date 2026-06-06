# 课程要求映射与文件证据索引

生成日期：2026-06-06

本文用于答辩或最终验收时快速说明：课程要求分别由哪些代码、页面、接口、模型和脚本覆盖。
快速验收版见 `docs/TEACHER_ACCEPTANCE_CHECKLIST.md`。

## 1. 课程要求覆盖矩阵

| 课程要求 | 项目实现 | 证据文件/接口 | 验收方式 |
| --- | --- | --- | --- |
| Python | 后端、检测算法、训练脚本和验收脚本均使用 Python 实现 | `app.py`、`detectors/`、`processors/`、`web/`、`yolo/`、`scripts/` | `python -m py_compile $(rg --files -g '*.py')` |
| OpenCV-python + contrib | 图像/视频帧处理、头姿 PnP、EAR/MAR、rPPG ROI；contrib 通过 `cv2.face` 检查 | `detectors/fatigue.py`、`detectors/head_pose.py`、`processors/frame_pipeline.py`、`scripts/acceptance_check.py` | `python scripts/acceptance_check.py` 中的 OpenCV contrib 检查 |
| Flask 后端 | 应用入口、上传、检测、模型状态、摄像头和 demo API | `app.py`、`web/routes_main.py`、`web/routes_upload.py`、`web/routes_detect.py`、`web/routes_camera.py`、`web/routes_demo.py` | 访问 `/api/heartbeat`、`/api/models`、`/api/demo/samples` |
| Bootstrap 前端 | 首页、检测页和摄像头页使用 Bootstrap 布局、按钮、图标和响应式样式 | `templates/index.html`、`templates/camera.html`、`static/css/style.css`、`static/js/camera.js` | 浏览器访问 `http://127.0.0.1:5001` 和 `/camera` |
| 模块化结构 | 检测、处理、Web 路由、模型、工具和训练脚本按功能拆分 | `detectors/`、`processors/`、`web/`、`models/`、`utils/`、`yolo/` | 查看目录结构，运行 `python scripts/package_check.py` |
| 图像检测 | 首页上传图片后自动调用 `ImageProcessor` 并生成结果 | `/upload`、`/detect/auto`、`processors/video_processor.py`、`processors/frame_pipeline.py` | 上传一张图片，检查结果页和输出报告 |
| 视频检测 | 视频上传后由 `VideoProcessor` 逐帧处理，输出检测摘要和报告 | `processors/video_processor.py`、`web/routes_upload.py`、`web/routes_detect.py` | 上传视频或运行现有 unittest |
| 摄像头检测 | 实时摄像头页按帧调用处理器，展示仪表盘、告警和状态 | `/camera`、`/camera/frame`、`processors/realtime_processor.py`、`templates/camera.html`、`static/js/camera.js` | 访问 `http://127.0.0.1:5001/camera` |
| 疲劳检测 | EAR 闭眼、眨眼率、MAR 哈欠、头姿点头/低头 | `detectors/fatigue.py`、`detectors/head_pose.py`、`config.py` | 查看摄像头页指标，运行 `python scripts/acceptance_check.py` |
| 分心检测 | 手机/吸烟/饮食、视线偏移、转头、转身、手离方向盘 | `detectors/distraction.py`、`detectors/gaze.py`、`detectors/head_pose.py`、`processors/frame_pipeline.py` | 摄像头页演示模式，或检查 `tests/test_performance_stability.py` |
| 生理状态展示 | rPPG 心率估计与血压趋势展示入口 | `detectors/physiological.py`、`models/lstm_bp.py`、`models_data/bp_lstm.pt` | 首页/结果页查看生理状态字段 |
| 至少使用 YOLO 训练一个分类或目标检测模型 | 已保留真实数据集训练入口与两个课程证明模型 | `yolo/train_real_datasets.py`、`yolo/train_drowsiness_cls.py`、`models_data/yolo_driver_state.pt`、`models_data/yolo_drowsiness_cls.pt` | `python -m yolo.train_real_datasets --dry-run`、`python scripts/package_check.py` |

## 2. 核心功能证据索引

| 文件或目录 | 作用 | 证据说明 |
| --- | --- | --- |
| `app.py` | Flask 应用入口 | 读取配置、初始化 Web 应用和端口 |
| `config.py` | 系统阈值与路径配置 | 包含 EAR、MAR、视线、转头、转身、虚拟方向盘和手离把阈值 |
| `web/` | Web 页面与 API 路由 | 覆盖首页、上传、检测、摄像头、模型状态和 demo samples |
| `processors/` | 图像、视频、实时帧处理 | 将单帧检测结果组织成前端和报告可消费的数据 |
| `detectors/` | 计算机视觉检测模块 | 疲劳、分心、头姿、视线、生理状态等核心检测逻辑 |
| `templates/` | Flask HTML 模板 | 首页、结果页和实时摄像头页面 |
| `static/` | 前端样式与交互脚本 | Bootstrap 风格美化、摄像头仪表盘、虚拟方向盘绘制和 demo 样例加载 |
| `models_data/yolo_driver_state.pt` | YOLO 驾驶状态检测证明模型 | 必须纳入最终课程交付包 |
| `models_data/yolo_drowsiness_cls.pt` | YOLO 驾驶疲劳分类证明模型 | 必须纳入最终课程交付包 |
| `yolo/train_real_datasets.py` | 真实公开数据集训练统一入口 | 支持 dry-run，便于答辩时证明训练流程 |
| `scripts/acceptance_check.py` | 本地一键验收 | 检查关键文件、依赖、模型、API、训练入口和 unittest |
| `scripts/package_check.py` | 最终打包清单校验 | 检查必要文件、证明模型和不应打包的本地运行产物 |
| `scripts/repro_report.py` | 可复现性报告生成 | 输出本机环境、依赖版本、模型大小和 Git 摘要 |

## 3. API 证据索引

| API | 验收用途 | 推荐检查 |
| --- | --- | --- |
| `/api/heartbeat` | 证明 Flask 后端正在运行，服务状态正常 | `curl http://127.0.0.1:5001/api/heartbeat` |
| `/api/models` | 证明必要模型文件可被后端识别，并展示模型可用状态 | `curl http://127.0.0.1:5001/api/models` |
| `/api/demo/samples` | 无摄像头时提供固定样例，支撑课堂备用演示路径 | `curl http://127.0.0.1:5001/api/demo/samples` |

## 4. 答辩时推荐展示顺序

1. 首页：说明系统目标、上传入口和模型状态。
2. 图像/视频检测入口：展示文件检测与报告生成能力。
3. 实时摄像头页：展示实时仪表盘、告警面板和虚拟方向盘。
4. 演示模式：快速触发手离把、转头或分心状态，说明只缩短持续时间。
5. 无摄像头固定样例：通过 `/api/demo/samples` 说明备用演示路径。
6. YOLO 训练证明：展示 `models_data/yolo_driver_state.pt`、`models_data/yolo_drowsiness_cls.pt` 和 `yolo/train_real_datasets.py --dry-run`。
7. 一键验收脚本：运行 `python scripts/acceptance_check.py`，证明环境、API、模型和测试通过。

## 5. 注意事项

- `models_data/yolo_steering_hand.pt` 是可选增强模型，不作为课程必要证明。
- 两个必要证明模型必须纳入最终包：`models_data/yolo_driver_state.pt`、`models_data/yolo_drowsiness_cls.pt`。
- 最终包不应包含 `.venv/`、`datasets/`、`runs/`、`uploads/`、`outputs/`、SQLite WAL/SHM 文件、`__pycache__/` 或 `*.pyc`。
- ChatGPT 只参与规划和验收评审；本地代码、测试、运行和回传由 Codex 执行。
