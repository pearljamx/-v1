"""
全局配置模块
============
包含疲劳检测、分心检测、生理检测的所有阈值常量、路径配置和摄像机参数。
所有数值均为推荐默认值，可根据实际场景调整。
"""

import numpy as np
from pathlib import Path

# ============================================================================
# 项目根目录
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent  # 本 config.py 所在目录即为项目根目录


# ============================================================================
# 模型文件路径
# ============================================================================
YOLO_HANDHELD_MODEL = str(BASE_DIR / "models_data" / "yolo_handheld.pt")   # YOLO 手持物（手机）检测模型
YOLO_DRIVER_STATE_MODEL = str(BASE_DIR / "models_data" / "yolo_driver_state.pt")  # 外部数据集训练的驾驶状态检测模型
YOLO_DRIVER_CLASSIFIER_MODEL = str(BASE_DIR / "models_data" / "driver_distraction_cls.pt")  # Mendeley/StateFarm 风格整帧分心分类模型
YOLO_STEERING_HAND_MODEL = str(BASE_DIR / "models_data" / "yolo_steering_hand.pt")  # 可选：真实方向盘/手部检测模型
YOLO_POSE_MODEL     = str(BASE_DIR / "models_data" / "yolov8n-pose.pt")   # YOLOv8 人体姿态估计模型（nano 版）
BP_LSTM_MODEL       = str(BASE_DIR / "models_data" / "bp_lstm.pt")        # LSTM 血压预测模型


# ============================================================================
# 摄像机参数（近似值，非严格标定，仅用于视线估计等几何计算）
# ============================================================================

# 摄像机内参矩阵 K（3×3）
# 假设传感器尺寸约 1/2.8''，焦距约 600 px（640×480 分辨率下的典型值）
CAMERA_MATRIX = np.array([
    [600.0,   0.0, 320.0],  # fx,   0,  cx（光心 x = 图像宽度 / 2）
    [  0.0, 600.0, 240.0],  #  0,  fy,  cy（光心 y = 图像高度 / 2）
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

# 畸变系数（k1, k2, p1, p2, k3），假设为普通广角畸变（近似为 0，未标定）
DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)


# ============================================================================
# 疲劳检测相关阈值
# ============================================================================

# --- 眼睛纵横比（Eye Aspect Ratio, EAR）---
EAR_THRESHOLD = 0.2        # EAR 低于此值判定为闭眼
                           # 参考 Soukupová & Čech (2016) 的推荐值
EAR_DURATION  = 3.0        # 闭眼持续超过此秒数触发疲劳告警（PERCLOS 逻辑）

# --- 眨眼频率（Blink Rate）---
BLINK_RATE_LOW        = 10  # 眨眼频率低于 10 次/分钟 → 可能疲劳/注意力下降
BLINK_RATE_NORMAL_MIN = 15  # 正常眨眼频率下限（次/分钟）
BLINK_RATE_NORMAL_MAX = 20  # 正常眨眼频率上限（次/分钟）

# --- 嘴部纵横比（Mouth Aspect Ratio, MAR）---
MAR_THRESHOLD = 0.5        # MAR 高于此值判定为打哈欠
MAR_DURATION  = 3.0        # 哈欠持续超过此秒数触发疲劳告警


# ============================================================================
# 分心检测相关阈值
# ============================================================================

# --- 头部姿态（Pitch：俯仰角，单位：度）---
PITCH_THRESHOLD = 15.0     # |pitch| 超过 15° 判定为低头/仰头（低头看手机或打瞌睡点头）
PITCH_DURATION  = 2.0      # 异常俯仰持续超过此秒数触发分心告警

# --- 转头（Yaw：偏航角，单位：度）---
HEAD_TURN_YAW_THRESHOLD = 35.0       # 头部左右偏航超过 35° 视为明显转头
HEAD_TURN_DANGER_YAW = 55.0          # 超过 55° 且持续时升级为 danger
HEAD_TURN_DURATION = 2.0             # 转头持续超过 2 秒才报警，避免正常扫视误报

# --- 视线方向（Gaze Angle，单位：度）---
GAZE_ANGLE_THRESHOLD = 30.0  # 视线偏离正前方超过 30° 判定为视线离开路面
GAZE_DURATION        = 2.0   # 视线偏转持续超过此秒数触发分心告警

# --- 手部离把检测 ---
HAND_OFF_WHEEL_DURATION = 5.0         # 双手均未检测到在方向盘区域持续超过此秒数触发 danger
SINGLE_HAND_OFF_WHEEL_DURATION = 8.0  # 单手离把允许更长容忍时间，持续后 warning
HAND_KEYPOINT_CONFIDENCE = 0.3        # COCO 手腕关键点置信度下限
VIRTUAL_WHEEL_CENTER = (0.50, 0.78)   # 虚拟方向盘中心，归一化坐标
VIRTUAL_WHEEL_RADIUS = (0.23, 0.16)   # 虚拟方向盘椭圆半径，归一化坐标
VIRTUAL_WHEEL_GRIP_TOLERANCE = 0.18   # 方向盘握持区域容差，数值越大越宽松

# --- YOLO 分心行为确认 ---
YOLO_OBJECT_CONFIDENCE = 0.25      # 低阈值捕获手机/抽烟/饮食候选
YOLO_OBJECT_CONFIRM_DURATION = 1.8 # 候选持续 1.5~2 秒后再报警
YOLO_CLASSIFIER_CONFIDENCE = 0.42  # 整帧分心分类的最低置信度，低于该值只展示不告警

# --- 躯干姿态 ---
SHOULDER_YAW_THRESHOLD = 45.0  # 肩部偏航角超过 45° 判定为身体大幅扭转
BODY_TURN_ANGLE        = 45.0  # 躯干整体旋转角超过 45° 判定为侧身（如回头取物）
BODY_TURN_DURATION     = 2.0   # 躯干扭转持续超过此秒数才报警

# --- 答辩演示模式 ---
# 默认关闭，不改变现实检测阈值。开启后只缩短持续时间，便于课堂现场快速展示告警。
DEMO_MODE_DEFAULT = False
DEMO_HAND_OFF_WHEEL_DURATION = 1.5
DEMO_SINGLE_HAND_OFF_WHEEL_DURATION = 2.0
DEMO_HEAD_TURN_DURATION = 1.0
DEMO_BODY_TURN_DURATION = 1.0


# ============================================================================
# 系统运行参数
# ============================================================================
FPS_TARGET = 15  # 目标处理帧率（Hz），降低计算负载同时保证检测时效性


# ============================================================================
# 生理信号（PPG / rPPG）检测相关阈值
# ============================================================================

PPG_WINDOW = 150  # PPG 滑动窗口长度（帧），约 150 / FPS_TARGET = 10 秒的历史信号

# --- 带通滤波器截止频率（Hz）---
PPG_BANDPASS_LOW  = 0.7   # 下限 0.7 Hz ≈ 42 BPM，滤除呼吸、身体晃动等低频噪声
PPG_BANDPASS_HIGH = 4.0   # 上限 4.0 Hz ≈ 240 BPM，滤除工频干扰和混叠高频

# --- 正常心率范围（BPM）---
HEART_RATE_NORMAL = (60, 100)  # 成人静息/低活动量状态下的正常心率区间
# 注意：驾驶场景下心率可能偏高（75-110 BPM），可根据实际情况放宽

# ============================================================================
# 模块开关（默认启用的检测模块）
# ============================================================================
ENABLE_FATIGUE = True
ENABLE_POSE = True
ENABLE_GAZE = True
ENABLE_DISTRACTION = True
ENABLE_PHYSIO = False  # 默认关闭生理信号检测（需要较长视频）

# ============================================================================
# 上传文件配置
# ============================================================================
ALLOWED_EXTENSIONS = {
    'image': {'jpg', 'jpeg', 'png', 'bmp', 'webp'},
    'video': {'mp4', 'avi', 'mov', 'webm', 'mkv'},
}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB
