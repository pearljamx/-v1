# 驾驶注意力检测系统

这是一个基于 Flask、OpenCV、MediaPipe 和 YOLO 的课程设计项目，支持图像、视频和摄像头实时驾驶状态检测。

## 交付文档入口

答辩或验收时建议先查看这些入口：

- `docs/DELIVERY_INDEX.md`：最终交付文档与检查脚本总索引，建议优先阅读。
- `docs/SUBMISSION_NOTE.md`：最终提交给教师/助教的简短说明。
- `docs/FINAL_FREEZE.md`：最终冻结确认，说明项目已进入提交/打包状态。
- `docs/FINAL_ACCEPTANCE.md`：课程要求覆盖情况、模型证明、API 验证和最终风险说明。
- `docs/TEACHER_ACCEPTANCE_CHECKLIST.md`：教师/助教快速验收清单。
- `docs/QUICK_START_CARD.md`：答辩现场一分钟启动与备用演示卡片。
- `docs/DEMO_SCRIPT.md`：5 到 8 分钟答辩演示讲稿，以及答辩前 5 分钟检查清单。
- `docs/TROUBLESHOOTING.md`：现场故障排查，覆盖端口、摄像头、模型、依赖、数据集和 Git 副作用。
- `docs/POST_SUBMISSION_CHECK.md`：提交/打包后复查清单。
- `docs/REPRODUCIBILITY_REPORT.md`：本机环境、依赖版本、模型大小和 Git 摘要，由 `scripts/repro_report.py` 生成。
- `docs/REQUIREMENT_TRACEABILITY.md`：课程要求到代码、接口、页面和模型的证据映射表。
- `docs/SUBMISSION_MANIFEST.md`：最终提交包目录、模型文件和排除项 Manifest，由 `scripts/submission_manifest.py` 生成。
- `docs/ARCHIVE_DRY_RUN.md`：压缩包预检报告，由 `scripts/archive_dry_run.py` 生成，不真正创建压缩包。
- `scripts/acceptance_check.py`：本地一键验收脚本，验证依赖、模型、API、摄像头演示增强和 unittest。
- `scripts/package_check.py`：最终打包清单校验，确认必要模型和新增交付文件已纳入、运行产物未误打包。
- `scripts/repro_report.py`：提交前可复现性报告生成脚本。
- `scripts/pre_commit_check.py`：提交前 Git 清单核验，只读输出推荐 `git add` 清单和禁止加入项。
- `scripts/submission_manifest.py`：最终提交包 Manifest 生成脚本，只读汇总必须包含、必须排除和当前 Git 状态。
- `scripts/archive_dry_run.py`：最终压缩包 dry-run 审计脚本，只读检查应包含/应排除路径和建议打包命令。

## 环境要求

- Windows 10/11 或 macOS
- Python 3.9 或兼容版本
- 推荐在虚拟环境中安装依赖

## 安装依赖

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 `face-recognition` 安装时尝试编译源码版 `dlib` 并失败，先确认 `dlib-bin`
和 `vendor/dlib_metadata_shim` 已安装，再执行：

```powershell
pip install face-recognition --no-deps
```

系统会优先使用 face_recognition/dlib 68 点后端；不可用时自动回退到 MediaPipe/Yunet。

## 模型文件

项目默认从 `models_data/` 读取模型：

- `models_data/yolo_handheld.pt`：手机、吸烟、饮水、手离方向盘检测模型
- `models_data/yolo_driver_state.pt`：外部公开数据集训练的驾驶状态检测模型
- `models_data/driver_distraction_cls.pt`：Mendeley 真实驾驶分心图片数据集训练的整帧分类模型
- `models_data/yolo_steering_hand.pt`：可选方向盘/手部真实数据集检测模型
- `models_data/yolov8n-pose.pt`：人体姿态模型
- `models_data/face_landmarker.task`：MediaPipe 面部关键点模型
- `models_data/face_detection_yunet_2023mar.onnx`：Yunet 人脸检测模型
- `models_data/bp_lstm.pt`：血压趋势预测模型

模型权重、数据库、上传文件和训练输出默认由 `.gitignore` 排除。交付给别人运行时，需要单独确认 `models_data/` 中必要模型已随包提供。
其中 `models_data/yolo_driver_state.pt`、`models_data/driver_distraction_cls.pt` 和 `models_data/yolo_drowsiness_cls.pt`
是课程训练证明模型，最终课程交付包必须包含这两个文件，不能因为它们被 `.gitignore`
排除而漏打包。

最终打包前建议执行：

```bash
python scripts/package_check.py
python scripts/repro_report.py
python scripts/submission_manifest.py
python scripts/archive_dry_run.py
python scripts/pre_commit_check.py
python scripts/acceptance_check.py
git status --short
```

`package_check.py`、`pre_commit_check.py`、`submission_manifest.py` 和 `archive_dry_run.py`
只做只读检查或生成报告，不删除文件、不执行 `git add`、不训练模型、不真正创建压缩包。

## 外部数据集与 YOLO 训练

统一查看真实数据集来源、训练脚本和当前模型证明状态：

```bash
python -m yolo.train_real_datasets --list
python -m yolo.train_real_datasets --dry-run
```

`--dry-run` 不下载数据、不训练模型，只检查训练脚本、数据配置、`ultralytics`
依赖、输出目录和已存在模型文件，适合答辩前快速证明训练入口完整。

推荐使用 Roboflow Universe 的 Driver fatigue and distraction 数据集：

```text
https://universe.roboflow.com/mds-workspace-arqn1/driver-fatigue-and-distraction-bned4
```

该数据集约 6.4k 图像，包含 `driver awake`、`driver drowsy`、`driver eating`、`driver sleeping`、`driver smoking`、`driver turning`、`driver using phone` 等类别，许可证为 CC BY 4.0。

自动下载需要 Roboflow API key：

```bash
export ROBOFLOW_API_KEY="你的API_KEY"
python -m yolo.roboflow_driver_dataset
python -m yolo.train_driver_state --quick
```

也可以在网页手动导出 YOLOv8 zip 后导入：

```bash
python -m yolo.roboflow_driver_dataset --zip /path/to/roboflow.zip
python -m yolo.train_driver_state
```

训练完成后最佳权重会复制为 `models_data/yolo_driver_state.pt`。如果该模型不存在，系统会继续使用 `models_data/yolo_handheld.pt` 作为 fallback。
无 CUDA 的机器建议先运行 `--quick` 验证数据格式和训练链路；完整训练可去掉 `--quick`，但 CPU 训练耗时会明显增加。

### Mendeley 驾驶分心整帧分类数据集

为了补充分心行为的整帧分类判断，可使用 Mendeley Data 的真实图片数据集
`Novel Driver Distractions Dataset With Low Lighting Support`：

```text
https://data.mendeley.com/datasets/ykmr99nrsg/2
```

该数据集为 10 类驾驶分心图片数据，包含安全驾驶、左右手打电话/发短信、
调收音机、饮水、回头取物、整理头发/化妆、与乘客交谈等类别，页面标注
许可证为 CC BY-NC 3.0。当前环境下 Mendeley 匿名 API 下载会返回未授权，
因此稳定流程是先在网页点击 Download All 获取 zip，再导入：

```bash
python -m yolo.mendeley_distraction_dataset --zip /path/to/mendeley.zip
python -m yolo.train_mendeley_distraction_cls --quick
```

完整训练可去掉 `--quick`。训练完成后最佳权重会复制为
`models_data/driver_distraction_cls.pt`，运行时会与 Roboflow YOLO 检测模型并行：
YOLO 检测模型负责目标框和局部告警，Mendeley 分类模型负责整帧驾驶行为分类。

如果暂时没有 Roboflow API key，也可以使用 Hugging Face 公开真实数据集
`n7i5x9/driver-drowsiness-dataset` 的 validation 分片训练 YOLOv8 驾驶疲劳二分类模型：

```bash
python -m yolo.hf_drowsiness_dataset --download
python -m yolo.train_drowsiness_cls --quick
```

该流程会导出平衡抽样的 YOLO classification 数据集，并将最佳权重复制为
`models_data/yolo_drowsiness_cls.pt`。本模型用于证明真实公开数据集上的 YOLO 分类训练流程；
若需要更高精度，可增大样本数量或使用完整数据集训练。

也可以通过统一入口启动轻量训练：

```bash
python -m yolo.train_real_datasets --run driver-state --quick
python -m yolo.train_real_datasets --run drowsiness-cls --prepare --quick
```

其中 `--prepare` 可能访问网络或要求 `ROBOFLOW_API_KEY`，不会在默认 dry-run 中自动执行。

## 实时摄像头演示增强

摄像头页 `/camera` 面向答辩演示做了独立实时仪表盘：

- 虚拟方向盘：画面下方使用椭圆方向盘，中心默认 `x=0.50`、`y=0.78`，半径默认 `rx=0.23`、`ry=0.16`。
- 手部检测：使用 YOLO pose 的 COCO 手腕关键点判断左右手是否靠近方向盘区域；双手离把持续 `5s` 触发 danger，单手离把持续 `8s` 触发 warning。
- 转头检测：头部 yaw 超过 `35°` 并持续 `2s` 触发 warning，超过 `55°` 且持续时升级 danger，避免短暂看后视镜造成误报。
- 转身检测：肩部/躯干扭转角超过 `45°` 并持续 `2s` 才触发 warning。
- 页面展示：右侧指标面板展示疲劳、视线、手部方向盘状态、转头/转身、光照和生理状态入口。

### 答辩演示模式与无摄像头备用路径

摄像头页默认使用现实阈值：双手离把 `5s`、单手离把 `8s`、转头 `2s`、
转身 `2s`。勾选“演示模式”后只缩短持续时间，便于课堂现场快速触发：
双手离把 `1.5s`、单手离把 `2s`、转头 `1s`、转身 `1s`，角度阈值和检测
逻辑保持不变。

如果答辩现场没有摄像头，可进入 `/camera` 使用“固定样例”按钮查看
`/api/demo/samples` 返回的样例图片，再回到首页 `/#upload-section` 上传检测。
该路径用于证明图像/视频处理链路可运行，不替代实时摄像头检测。

## 启动

默认端口为 5000：

```powershell
python app.py
```

当前 macOS 答辩演示环境建议使用 5001 端口，避免与系统服务冲突：

```bash
FLASK_PORT=5001 python app.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

可选环境变量：

- `SECRET_KEY`：Flask 会话密钥，未设置时启动时自动生成
- `FLASK_DEBUG`：是否开启调试模式，默认 `true`
- `FLASK_HOST`：监听地址，默认 `127.0.0.1`
- `FLASK_PORT`：监听端口，默认 `5000`
- `CORS_ORIGINS`：允许跨域来源，默认 `*`

## macOS 本地验收 / 答辩演示流程

当前课程设计目录：

```bash
cd /Volumes/Data/02_课程/计算机视觉/实验/课程设计
```

激活虚拟环境：

```bash
source .venv/bin/activate
```

启动 Flask 服务：

```bash
FLASK_PORT=5001 python app.py
```

浏览器访问：

```text
http://127.0.0.1:5001
```

API 验证：

```bash
curl http://127.0.0.1:5001/api/heartbeat
curl http://127.0.0.1:5001/api/models
curl http://127.0.0.1:5001/api/demo/samples
```

运行单元测试：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

运行一键验收脚本：

```bash
python scripts/acceptance_check.py
```

验收脚本会检查关键文件、Python 依赖、OpenCV contrib、YOLO 模型文件、Flask API
和现有 unittest，输出 `[PASS]`、`[WARN]`、`[FAIL]` 状态；出现失败时返回非 0
退出码，便于答辩前快速判断交付状态。

三个真实数据模型文件的证明意义：

- `models_data/yolo_driver_state.pt`：使用驾驶状态目标检测数据集训练得到，用于证明外部数据集上的 YOLO 检测训练流程。
- `models_data/driver_distraction_cls.pt`：使用 Mendeley 驾驶分心图片数据集训练得到，用于证明整帧驾驶行为分类流程。
- `models_data/yolo_drowsiness_cls.pt`：使用 Hugging Face 公开真实数据集训练得到，用于证明驾驶疲劳二分类 YOLO 训练流程。

演示时可依次展示：

- 图像上传检测：在首页上传图片后执行自动检测并查看结果。
- 视频文件检测：上传视频后生成逐帧检测结果和报告。
- 摄像头检测：进入摄像头页面，开启实时监测。
- 无摄像头备用：进入摄像头页面点击固定样例，再回到首页上传样例图检测。
- 疲劳检测：展示 EAR、MAR、头姿和闭眼/哈欠告警逻辑。
- 分心检测：展示 YOLO 手机/吸烟/饮食、视线偏移和姿态判断。
- 生理状态展示：展示 rPPG 心率估计和血压趋势模块说明。

## 常用验证

```powershell
python -m compileall .
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/acceptance_check.py
```

## 目录说明

- `app.py`：Flask 应用入口
- `web/`：页面和 API 路由
- `processors/`：图像、视频和实时帧处理
- `detectors/`：疲劳、分心、头姿、视线和生理信号检测
- `utils/`：通用工具、可视化和报告生成
- `models/`：数据库与 LSTM 模型封装
- `yolo/`：YOLO 数据准备、训练和模型管理脚本
