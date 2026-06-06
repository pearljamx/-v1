# 驾驶注意力检测系统故障排查

本文用于答辩或验收现场快速定位问题。优先按顺序检查，不建议在答辩前临时重新训练模型或下载大数据集。

## 1. Flask 服务无法访问

推荐启动命令：

```bash
cd /Volumes/Data/02_课程/计算机视觉/实验/课程设计
source .venv/bin/activate
FLASK_PORT=5001 python app.py
```

检查 API：

```bash
curl http://127.0.0.1:5001/api/heartbeat
curl http://127.0.0.1:5001/api/models
curl http://127.0.0.1:5001/api/demo/samples
```

排查要点：

- 如果 `5001` 被占用，换成其他端口，例如 `FLASK_PORT=5002 python app.py`，浏览器地址也同步改为 `http://127.0.0.1:5002`。
- 如果 `curl` 没有返回，先确认终端中的 Flask 服务仍在运行。
- 如果页面能打开但 API 失败，先运行 `python scripts/acceptance_check.py` 查看失败项。

## 2. 摄像头页面没有画面

打开：

```text
http://127.0.0.1:5001/camera
```

排查要点：

- 使用 Chrome，并确认地址是 `localhost` 或 `127.0.0.1`。
- 浏览器弹出摄像头权限时选择允许。
- 如果权限曾被拒绝，在 Chrome 地址栏左侧的网站设置里重新允许摄像头。
- 如果本机没有摄像头，使用页面中的“固定样例”按钮走无摄像头备用演示。
- 实时检测未能验证时，答辩时按 `docs/DEMO_SCRIPT.md` 的 5 分钟检查清单说明这是硬件环境限制。

## 3. 看不到虚拟方向盘或演示模式

排查要点：

- 刷新 `/camera` 页面，确认加载的是最新前端文件。
- 页面应包含实时画面区、虚拟方向盘辅助线、手部方向盘状态、转头/转身状态和演示模式开关。
- 运行 `python scripts/acceptance_check.py`，其中“摄像头实时演示增强”应为 `[PASS]`。

## 4. 模型文件缺失

关键模型：

- `models_data/yolo_driver_state.pt`
- `models_data/yolo_drowsiness_cls.pt`
- `models_data/yolo_handheld.pt`
- `models_data/yolov8n-pose.pt`

检查命令：

```bash
python scripts/acceptance_check.py
curl http://127.0.0.1:5001/api/models
```

说明：

- `yolo_steering_hand.pt` 是可选模型。当前手离把判断主要由 YOLO pose 手腕关键点与虚拟方向盘 ROI 完成。
- 不建议答辩前临时替换或重新训练模型，避免破坏当前可验收状态。

## 5. 打包时模型或脚本缺失

最终打包前先运行：

```bash
python scripts/package_check.py
python scripts/submission_manifest.py
python scripts/archive_dry_run.py
```

排查要点：

- 如果最终包里缺少 `models_data/yolo_driver_state.pt` 或 `models_data/yolo_drowsiness_cls.pt`，`/api/models` 会显示对应模型不可用，课程中的 YOLO 训练证明也会变弱。
- `docs/`、`scripts/`、`vendor/`、`web/routes_demo.py`、`yolo/train_real_datasets.py`、`yolo/hf_drowsiness_dataset.py`、`yolo/train_drowsiness_cls.py` 都属于交付增强内容，提交或打包前要确认已经纳入。
- 不要为了减小压缩包体积删除两个证明模型：`models_data/yolo_driver_state.pt` 和 `models_data/yolo_drowsiness_cls.pt`。
- `.venv/`、`datasets/`、`uploads/`、`outputs/`、`runs/`、SQLite WAL/SHM 和 Python 缓存只属于本地运行产物，不应进入最终课程包。

## 6. face_recognition 或 dlib 安装问题

排查要点：

- macOS 环境优先使用项目中的 `vendor/dlib_metadata_shim` 配合 `dlib-bin`。
- 如果 `face-recognition` 安装时尝试编译源码版 `dlib` 并失败，先安装兼容依赖，再执行：

```bash
pip install face-recognition --no-deps
```

系统会优先使用 face_recognition/dlib 68 点后端；不可用时自动回退到 MediaPipe/Yunet，不会阻断基础演示。

## 7. Roboflow 或真实数据集下载问题

查看训练入口：

```bash
python -m yolo.train_real_datasets --list
python -m yolo.train_real_datasets --dry-run
```

排查要点：

- Roboflow 自动下载需要合法的 `ROBOFLOW_API_KEY`。
- 没有 API key 时，不要绕过认证；可以使用网页手动导出的 YOLOv8 zip，或使用 Hugging Face 驾驶疲劳分类数据集训练入口。
- `--dry-run` 不下载、不训练，只证明训练脚本、模型路径和依赖完整，适合答辩前展示。

## 8. Git 状态出现运行副作用文件

答辩或提交前检查：

```bash
python scripts/pre_commit_check.py
git status --short
git status --short | rg 'labels\.cache|detection_history\.db|datasets/|__pycache__|\.pyc$' || true
```

不应新增提交：

- `dataset/*/labels.cache`
- `models_data/detection_history.db`
- `datasets/`
- `__pycache__/`
- `*.pyc`

说明：

- 仓库中已有基线跟踪文件不要随意删除。
- 如果只是运行产生的临时文件，保持 `.gitignore` 规则生效即可。

## 9. 提交时漏加文件或误加运行产物

提交或打包后复查见 `docs/POST_SUBMISSION_CHECK.md`。

提交前先运行：

```bash
python scripts/pre_commit_check.py
```

排查要点：

- 按脚本输出人工执行 `git add`，脚本不会替你执行 Git 修改。
- 不要使用 `git add -f` 加入 ignored 的 `datasets/`、`.venv/`、SQLite WAL/SHM 或缓存目录。
- 两个课程证明模型 `models_data/yolo_driver_state.pt` 和 `models_data/yolo_drowsiness_cls.pt` 必须纳入最终课程交付包。
- 提交或打包后再次运行 `python scripts/package_check.py` 和 `python scripts/acceptance_check.py`。

## 10. 提交包内容不完整

如果老师或助教要求核对最终提交包内容，先生成 Manifest：

```bash
python scripts/submission_manifest.py
```

排查要点：

- 打开 `docs/SUBMISSION_MANIFEST.md`，确认必须包含的顶层目录、关键文档、关键脚本均为 `[PASS]`。
- 检查两个 `.pt` 课程证明模型是否存在且大于 1 MB：`models_data/yolo_driver_state.pt`、`models_data/yolo_drowsiness_cls.pt`。
- 确认 `docs/`、`scripts/`、`vendor/`、`web/routes_demo.py`、`yolo/` 新增训练脚本都在 Manifest 中。
- 不要为了减小体积删除两个课程证明模型。
- 如果 Manifest 显示 `[WARN] 本地存在但不应打包`，只要这些路径没有进入普通 `git status --short` 或最终压缩包即可。

## 11. 压缩包体积异常或缺少模型

最终打包前先运行 dry-run 预检：

```bash
python scripts/archive_dry_run.py
```

排查要点：

- 如果两个 `.pt` 课程证明模型缺失或小于 1 MB，不能提交最终包。
- 如果最终压缩包包含 `.venv/`、`datasets/`、`runs/`、`uploads/`、`outputs/`，应重新打包。
- 如果本地存在 SQLite WAL/SHM、`__pycache__/` 或 `*.pyc`，只要没有进入最终压缩包即可。
- 不要为了缩小体积删除两个课程证明模型。

## 12. 最快恢复演示流程

快速命令见 `docs/QUICK_START_CARD.md`。

如果现场时间紧，按下面顺序恢复：

1. 启动服务：`FLASK_PORT=5001 python app.py`。
2. 打开首页：`http://127.0.0.1:5001`。
3. 打开摄像头页：`http://127.0.0.1:5001/camera`。
4. 若摄像头不可用，点击“固定样例”。
5. 回到首页上传样例图，展示图像检测结果。
6. 运行 `python scripts/acceptance_check.py` 展示最终验收通过。
