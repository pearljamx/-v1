# 驾驶注意力检测系统

这是一个基于 Flask、OpenCV、MediaPipe 和 YOLO 的课程设计项目，支持图像、视频和摄像头实时驾驶状态检测。

## 环境要求

- Windows 10/11
- Python 3.9 或兼容版本
- 推荐在虚拟环境中安装依赖

## 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 `face-recognition` 安装时尝试编译源码版 `dlib` 并失败，先确认 `dlib-bin` 已安装，再执行：

```powershell
pip install face-recognition --no-deps
```

系统会优先使用 face_recognition/dlib 68 点后端；不可用时自动回退到 MediaPipe/Yunet。

## 模型文件

项目默认从 `models_data/` 读取模型：

- `models_data/yolo_handheld.pt`：手机、吸烟、饮水、手离方向盘检测模型
- `models_data/yolo_driver_state.pt`：外部公开数据集训练的驾驶状态检测模型
- `models_data/yolov8n-pose.pt`：人体姿态模型
- `models_data/face_landmarker.task`：MediaPipe 面部关键点模型
- `models_data/face_detection_yunet_2023mar.onnx`：Yunet 人脸检测模型
- `models_data/bp_lstm.pt`：血压趋势预测模型

模型权重、数据库、上传文件和训练输出默认由 `.gitignore` 排除。交付给别人运行时，需要单独确认 `models_data/` 中必要模型已随包提供。

## 外部数据集与 YOLO 训练

推荐使用 Roboflow Universe 的 Driver fatigue and distraction 数据集：

```text
https://universe.roboflow.com/mds-workspace-arqn1/driver-fatigue-and-distraction-bned4
```

该数据集约 6.4k 图像，包含 `driver awake`、`driver drowsy`、`driver eating`、`driver sleeping`、`driver smoking`、`driver turning`、`driver using phone` 等类别，许可证为 CC BY 4.0。

自动下载需要 Roboflow API key：

```powershell
$env:ROBOFLOW_API_KEY="你的API_KEY"
python -m yolo.roboflow_driver_dataset
python -m yolo.train_driver_state --quick
```

也可以在网页手动导出 YOLOv8 zip 后导入：

```powershell
python -m yolo.roboflow_driver_dataset --zip D:\path\to\roboflow.zip
python -m yolo.train_driver_state
```

训练完成后最佳权重会复制为 `models_data/yolo_driver_state.pt`。如果该模型不存在，系统会继续使用 `models_data/yolo_handheld.pt` 作为 fallback。

## 启动

```powershell
python app.py
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

## 常用验证

```powershell
python -m compileall .
python -m unittest discover -s tests -p "test_*.py" -v
```

## 目录说明

- `app.py`：Flask 应用入口
- `web/`：页面和 API 路由
- `processors/`：图像、视频和实时帧处理
- `detectors/`：疲劳、分心、头姿、视线和生理信号检测
- `utils/`：通用工具、可视化和报告生成
- `models/`：数据库与 LSTM 模型封装
- `yolo/`：YOLO 数据准备、训练和模型管理脚本
