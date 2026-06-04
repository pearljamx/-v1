"""
分心驾驶检测模块
=================
使用YOLO检测手持物品（手机、吸烟、喝水等）和人体姿态,
通过手腕关键点与方向盘ROI的空间关系判断手离方向盘,
通过肩部关键点水平位移判断转身取物。

所有阈值和模型路径从 config.py 读取。
"""

import os
import time
import logging
from collections import deque
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    YOLO_HANDHELD_MODEL, YOLO_DRIVER_STATE_MODEL, YOLO_POSE_MODEL,
    HAND_OFF_WHEEL_DURATION, SHOULDER_YAW_THRESHOLD, BODY_TURN_ANGLE,
    YOLO_OBJECT_CONFIDENCE, YOLO_OBJECT_CONFIRM_DURATION
)

logger = logging.getLogger(__name__)
_YOLO_MODEL_CACHE = {}

# ---------------------------------------------------------------------------
# 手持物品类别映射
#  0: normal       — 正常（无手持物）
#  1: phone        — 使用手机
#  2: smoking      — 吸烟
#  3: drinking     — 喝水/进食
#  4: hands_off    — 手离方向盘
# ---------------------------------------------------------------------------
HANDHELD_CLASSES = {
    0: "normal",
    1: "phone",
    2: "smoking",
    3: "drinking",
    4: "hands_off",
    5: "eating",
    6: "turning",
    7: "drowsy",
}

DRIVER_STATE_LABELS = {
    "driver using phone": 1,
    "using phone": 1,
    "phone": 1,
    "driver smoking": 2,
    "smoking": 2,
    "driver drinking": 3,
    "drinking": 3,
    "driver eating": 5,
    "eating": 5,
    "driver turning": 6,
    "turning": 6,
    "driver drowsy": 7,
    "driver sleeping": 7,
    "drowsy": 7,
    "sleeping": 7,
    "driver awake": 0,
    "awake": 0,
    "normal": 0,
}

# 需要触发告警的类别 ID（排除 "normal"）
ALERT_CLASS_IDS = {1, 2, 3, 4, 5, 6, 7}

# 类别 → 告警类型映射
CLASS_ALERT_TYPE = {
    1: "phone_usage",
    2: "smoking",
    3: "drinking",
    4: "hands_off",
    5: "eating",
    6: "body_turn",
    7: "drowsy",
}

# COCO 姿态关键点索引（仅列出本模块用到的）
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_WRIST     = 9
KP_RIGHT_WRIST    = 10


def _load_cached_yolo_model(model_path, model_name, missing_message):
    """加载并缓存 YOLO 模型，避免多个检测器重复占用初始化成本。"""
    if not os.path.exists(model_path):
        logger.warning("%s文件不存在: %s, %s", model_name, model_path, missing_message)
        _YOLO_MODEL_CACHE[model_path] = None
        return None

    if model_path in _YOLO_MODEL_CACHE:
        return _YOLO_MODEL_CACHE[model_path]

    try:
        model = YOLO(model_path)
        _YOLO_MODEL_CACHE[model_path] = model
        logger.info("%s加载成功: %s", model_name, model_path)
        return model
    except Exception as e:
        logger.error("%s加载失败 (%s): %s", model_name, model_path, e)
        _YOLO_MODEL_CACHE[model_path] = None
        return None


class DistractionDetector:
    """分心驾驶检测器。

    同时运行两个 YOLO 模型:
      - handheld_model : 手持物品分类/检测
      - pose_model     : 人体姿态估计（COCO 17 关键点）

    任一模型不可用时自动降级, 对应检测返回空/None。
    """

    def __init__(self):
        # --- 模型 ---
        self.handheld_model = None
        self.driver_state_model = None
        self.pose_model = None
        self._init_models()

        # --- 手离方向盘状态追踪 ---
        self.single_hand_off_start = None # 单手离开方向盘的起始时间戳（秒）
        self.hands_off_start = None       # 双手离开方向盘的起始时间戳（秒）
        self.hands_off_active = False     # 当前是否处于手离方向盘告警激活状态

        # --- 转身检测状态追踪 ---
        self.shoulder_positions = deque(maxlen=30)  # 肩部中心点历史坐标
        self.body_turn_active = False
        self.body_turn_info = {}
        self.object_confirm_starts = {}

        # --- 方向盘 ROI（归一化坐标） ---
        # (x, y, w, h) → 图像下半部分中心区域，约 25%-75% 宽度，60%-95% 高度
        self.wheel_roi = (0.25, 0.6, 0.5, 0.35)

    # ========================================================================
    # 模型初始化
    # ========================================================================

    def _init_models(self):
        """初始化 YOLO 模型, 任一模型文件不存在时优雅降级并记录警告。"""
        self._try_load_driver_state()
        self._try_load_handheld()
        self._try_load_pose()

    def _try_load_driver_state(self):
        """尝试加载外部数据集训练的驾驶员状态检测模型。"""
        self.driver_state_model = _load_cached_yolo_model(
            YOLO_DRIVER_STATE_MODEL,
            "驾驶状态检测模型",
            "外部数据集模型不可用，将回退到手持物检测模型。",
        )

    def _try_load_handheld(self):
        """尝试加载手持物品检测模型。"""
        self.handheld_model = _load_cached_yolo_model(
            YOLO_HANDHELD_MODEL,
            "手持物品检测模型",
            "手持物检测功能将禁用。如需启用, 请将训练好的模型放到该路径。",
        )

    def _try_load_pose(self):
        """尝试加载人体姿态估计模型。"""
        self.pose_model = _load_cached_yolo_model(
            YOLO_POSE_MODEL,
            "姿态估计模型",
            "姿态检测功能将禁用。如需启用, 请将 YOLOv8-pose 模型放到该路径。",
        )

    @property
    def handheld_available(self) -> bool:
        """手持物品检测是否可用。"""
        return self.handheld_model is not None

    @property
    def driver_state_available(self) -> bool:
        """驾驶状态检测是否可用。"""
        return self.driver_state_model is not None

    @property
    def pose_available(self) -> bool:
        """姿态估计是否可用。"""
        return self.pose_model is not None

    # ========================================================================
    # 手持物品检测
    # ========================================================================

    def detect_objects(self, image):
        """检测手持物品。

        Args:
            image: BGR numpy array (H, W, 3)

        Returns:
            list[dict]: 检测结果列表, 每项包含:
                {
                    'class': str,       # 类别名称
                    'class_id': int,    # 类别 ID
                    'confidence': float,# 置信度
                    'bbox': tuple(int, int, int, int)  # (x1, y1, x2, y2) 像素坐标
                }
            模型不可用时返回空列表。
        """
        detected = []
        if self.driver_state_available:
            detected.extend(self._detect_with_model(
                self.driver_state_model, image, source="driver_state"
            ))
        if self.handheld_available:
            detected.extend(self._detect_with_model(
                self.handheld_model, image, source="handheld"
            ))
        return detected

    def _detect_with_model(self, model, image, source="handheld"):
        """用指定 YOLO 模型检测并归一化为内部类别。"""
        try:
            results = model(image, verbose=False, conf=YOLO_OBJECT_CONFIDENCE)
        except Exception as e:
            logger.error("手持物品检测推理失败: %s", e)
            return []

        detected = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()       # (N, 4)
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
            confs = result.boxes.conf.cpu().numpy()       # (N,)

            for box, cls_id, conf in zip(boxes, cls_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                class_name, internal_id = self._normalize_class(model, cls_id)
                detected.append({
                    'class': class_name,
                    'class_id': int(internal_id),
                    'raw_class_id': int(cls_id),
                    'confidence': float(conf),
                    'bbox': (x1, y1, x2, y2),
                    'source': source,
                })

        return detected

    @staticmethod
    def _normalize_class(model, cls_id):
        raw_name = None
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            raw_name = names.get(int(cls_id))
        elif isinstance(names, (list, tuple)) and int(cls_id) < len(names):
            raw_name = names[int(cls_id)]

        if raw_name is None:
            return HANDHELD_CLASSES.get(int(cls_id), "unknown"), int(cls_id)

        key = str(raw_name).lower().replace("_", " ").replace("-", " ").strip()
        internal_id = DRIVER_STATE_LABELS.get(key, int(cls_id))
        return HANDHELD_CLASSES.get(internal_id, key), internal_id

    def _confirm_object_alert(self, alert_type, timestamp):
        """低置信候选需持续存在后才报警；静态图片 timestamp=0 时立即确认。"""
        if timestamp <= 0:
            return True, 0.0
        start = self.object_confirm_starts.get(alert_type)
        if start is None:
            self.object_confirm_starts[alert_type] = timestamp
            return False, 0.0
        duration = timestamp - start
        return duration >= YOLO_OBJECT_CONFIRM_DURATION, duration

    # ========================================================================
    # 姿态估计
    # ========================================================================

    def detect_pose(self, image):
        """YOLOv8 姿态估计: 检测人体关键点。

        Args:
            image: BGR numpy array (H, W, 3)

        Returns:
            Optional[list]:
                每个元素 shape (17, 3), 为 [x, y, confidence] for 17 COCO 关键点,
                按置信度降序排列（高置信度的人在前）。
                模型不可用或未检测到人时返回 None。
        """
        if not self.pose_available:
            return None

        try:
            results = self.pose_model(image, verbose=False)
        except Exception as e:
            logger.error("姿态估计推理失败: %s", e)
            return None

        all_keypoints = []
        for result in results:
            if result.keypoints is None:
                continue
            kps = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
            if kps.shape[0] == 0:
                continue
            # 按目标置信度降序排列（取每个目标所有关键点置信度的平均值）
            confs = kps[..., 2].mean(axis=1)  # (N,)
            sorted_idx = np.argsort(-confs)
            for idx in sorted_idx:
                all_keypoints.append(kps[idx])

        return all_keypoints if all_keypoints else None

    # ========================================================================
    # 手离方向盘检测
    # ========================================================================

    def check_hands_on_wheel(self, keypoints, img_width, img_height, timestamp):
        """检测手是否在方向盘上。

        通过计算左右手腕关键点(9, 10)与方向盘 ROI 的空间关系:
          手腕在 ROI 内 → 手在方向盘上
          双手均在 ROI 外且持续超过 HAND_OFF_WHEEL_DURATION → 告警

        Args:
            keypoints: numpy array (17, 3), COCO 关键点 [x, y, conf]
            img_width: 图像宽度（像素）
            img_height: 图像高度（像素）
            timestamp: 当前时间戳（秒）

        Returns:
            dict: 手部状态、ROI、告警等级和持续时间。
        """
        if img_width <= 0 or img_height <= 0:
            return {'state': 'unknown', 'alert_level': None, 'duration': 0.0}

        # 计算方向盘 ROI 像素边界
        rx, ry, rw, rh = self.wheel_roi
        roi_x1 = int(rx * img_width)
        roi_y1 = int(ry * img_height)
        roi_x2 = int((rx + rw) * img_width)
        roi_y2 = int((ry + rh) * img_height)

        # 检查左腕 (9) 和右腕 (10) 是否在 ROI 内
        left_wrist = keypoints[KP_LEFT_WRIST]
        right_wrist = keypoints[KP_RIGHT_WRIST]

        left_in_roi = (
            left_wrist[2] >= 0.3
            and self._point_in_roi(left_wrist[0], left_wrist[1],
                                    roi_x1, roi_y1, roi_x2, roi_y2)
        )
        right_in_roi = (
            right_wrist[2] >= 0.3
            and self._point_in_roi(right_wrist[0], right_wrist[1],
                                    roi_x1, roi_y1, roi_x2, roi_y2)
        )

        both_hands_off = (not left_in_roi) and (not right_in_roi)
        one_hand_off = left_in_roi != right_in_roi
        alert_level = None
        duration = 0.0
        state = 'both_on'

        if both_hands_off:
            self.single_hand_off_start = None
            if self.hands_off_start is None:
                self.hands_off_start = timestamp
            duration = timestamp - self.hands_off_start
            state = 'both_off'
            if duration >= HAND_OFF_WHEEL_DURATION:
                self.hands_off_active = True
                alert_level = 'danger'
        elif one_hand_off:
            self.hands_off_start = None
            if self.single_hand_off_start is None:
                self.single_hand_off_start = timestamp
            duration = timestamp - self.single_hand_off_start
            state = 'left_off' if not left_in_roi else 'right_off'
            if duration >= HAND_OFF_WHEEL_DURATION:
                self.hands_off_active = True
                alert_level = 'warning'
        else:
            # 双手在方向盘上 → 重置
            self.single_hand_off_start = None
            self.hands_off_start = None
            self.hands_off_active = False

        return {
            'state': state,
            'left_in_roi': bool(left_in_roi),
            'right_in_roi': bool(right_in_roi),
            'alert_level': alert_level,
            'duration': float(duration),
            'roi': (roi_x1, roi_y1, roi_x2, roi_y2),
            'wrists': {
                'left': [float(left_wrist[0]), float(left_wrist[1]), float(left_wrist[2])],
                'right': [float(right_wrist[0]), float(right_wrist[1]), float(right_wrist[2])],
            },
        }

    @staticmethod
    def _point_in_roi(px, py, rx1, ry1, rx2, ry2):
        """判断坐标点是否在矩形 ROI 内。

        Args:
            px, py: 关键点像素坐标
            rx1, ry1, rx2, ry2: ROI 像素边界

        Returns:
            bool
        """
        return (rx1 <= px <= rx2) and (ry1 <= py <= ry2)

    # ========================================================================
    # 转身检测
    # ========================================================================

    def check_body_turn(self, keypoints, timestamp):
        """检测转身取物。

        通过肩部关键点(5=左肩, 6=右肩)的水平位移分析:
          计算当前帧肩部中心点与历史均值的欧氏距离,
          当水平偏移角度超过 BODY_TURN_ANGLE 时触发转身告警。

        同时利用 SHOULDER_YAW_THRESHOLD 做第二道校验:
          肩部连线与水平线的夹角（偏航近似）超过阈值时辅助判定。

        Args:
            keypoints: numpy array (17, 3), COCO 关键点 [x, y, conf]
            timestamp: 当前时间戳（秒）

        Returns:
            bool: True 表示转身告警激活
        """
        left_shoulder  = keypoints[KP_LEFT_SHOULDER]
        right_shoulder = keypoints[KP_RIGHT_SHOULDER]

        # 置信度检查
        if left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3:
            # 关键点不可靠, 不清除状态但也不触发
            return self.body_turn_active

        # 当前肩部中心
        center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
        center_y = (left_shoulder[1] + right_shoulder[1]) / 2.0

        # 记录历史
        self.shoulder_positions.append((center_x, center_y, timestamp))

        if len(self.shoulder_positions) < 5:
            # 历史不足, 无法可靠判断
            return self.body_turn_active

        # 计算历史均值（排除最近一帧, 用前 N-1 帧作为基准）
        history = list(self.shoulder_positions)[:-1]
        mean_x = np.mean([p[0] for p in history])
        mean_y = np.mean([p[1] for p in history])

        # 当前与基准的偏移
        dx = center_x - mean_x
        dy = center_y - mean_y
        offset_px = np.sqrt(dx * dx + dy * dy)

        # 肩部宽度作为归一化基准
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        if shoulder_width < 10:
            return self.body_turn_active

        # 方法1: 水平偏移量相对肩宽的比例 → 估算偏转角度
        normalized_offset = offset_px / shoulder_width
        estimated_angle = np.degrees(np.arctan(normalized_offset))

        # 方法2: 当前肩部连线与水平线的夹角（偏航近似）
        shoulder_dx = right_shoulder[0] - left_shoulder[0]
        shoulder_dy = right_shoulder[1] - left_shoulder[1]
        shoulder_angle = abs(np.degrees(np.arctan2(shoulder_dy, shoulder_dx)))

        self.body_turn_info = {
            'center': [float(center_x), float(center_y)],
            'baseline': [float(mean_x), float(mean_y)],
            'estimated_angle': float(estimated_angle),
            'shoulder_angle': float(shoulder_angle),
            'left_shoulder': [float(left_shoulder[0]), float(left_shoulder[1]), float(left_shoulder[2])],
            'right_shoulder': [float(right_shoulder[0]), float(right_shoulder[1]), float(right_shoulder[2])],
        }

        if estimated_angle > BODY_TURN_ANGLE or shoulder_angle > SHOULDER_YAW_THRESHOLD:
            self.body_turn_active = True
        else:
            self.body_turn_active = False

        return self.body_turn_active

    # ========================================================================
    # 综合检测入口
    # ========================================================================

    def update_and_check(self, image, face_result, timestamp):
        """综合检测函数 — 同时运行手持物检测 + 姿态 + 手离把 + 转身。

        Args:
            image: BGR numpy array (H, W, 3)
            face_result: 人脸检测结果 dict（当前版本仅透传, 保留扩展性）
            timestamp: 当前时间戳（秒）

        Returns:
            dict:
                {
                    'alerts': [
                        {
                            'source': 'distraction',
                            'type': str,         # 告警类型
                            'severity': str,     # 'warning' | 'danger'
                            'timestamp': float,
                            'message': str,
                            'metadata': dict,
                        },
                        ...
                    ],
                    'objects': [              # 手持物品检测结果
                        {
                            'class': str,
                            'class_id': int,
                            'confidence': float,
                            'bbox': (x1, y1, x2, y2),
                        },
                        ...
                    ],
                    'hands_off_wheel': bool,  # 手离方向盘状态
                    'body_turn': bool,        # 转身状态
                }
        """
        alerts = []
        img_height, img_width = image.shape[:2]

        # ------------------------------------------------------------------
        # 1. 手持物品检测
        # ------------------------------------------------------------------
        objects = self.detect_objects(image)

        seen_alert_types = set()
        for obj in objects:
            cls_id = obj['class_id']
            if cls_id not in ALERT_CLASS_IDS:
                continue

            alert_type = CLASS_ALERT_TYPE.get(cls_id, "distraction_object")
            seen_alert_types.add(alert_type)
            confirmed, duration = self._confirm_object_alert(alert_type, timestamp)
            if not confirmed:
                continue
            severity = 'danger' if cls_id == 1 else 'warning'  # 手机使用 = danger

            # 构建告警消息
            class_labels = {
                1: "检测到使用手机",
                2: "检测到吸烟行为",
                3: "检测到饮水/进食",
                4: "检测到手离方向盘",
                5: "检测到饮食行为",
                6: "检测到转身取物",
                7: "检测到疲劳/睡眠状态",
            }
            message = class_labels.get(cls_id, f"检测到分心行为: {obj['class']}")

            alerts.append({
                'source': 'distraction',
                'type': alert_type,
                'severity': severity,
                'timestamp': timestamp,
                'message': message,
                'metadata': {
                    'class': obj['class'],
                    'confidence': obj['confidence'],
                    'bbox': obj['bbox'],
                    'duration': duration,
                },
            })

        for alert_type in list(self.object_confirm_starts.keys()):
            if alert_type not in seen_alert_types:
                self.object_confirm_starts.pop(alert_type, None)

        # ------------------------------------------------------------------
        # 2. 姿态估计
        # ------------------------------------------------------------------
        keypoints_list = self.detect_pose(image)

        hands_off = False
        hand_status = {}
        body_turn = False
        best_kp = None

        if keypoints_list and len(keypoints_list) > 0:
            # 取置信度最高的人
            best_kp = keypoints_list[0]  # (17, 3)

            # 2a. 手离方向盘检测
            hand_status = self.check_hands_on_wheel(
                best_kp, img_width, img_height, timestamp
            )
            hands_off = hand_status.get('alert_level') is not None

            if hands_off:
                level = hand_status.get('alert_level', 'warning')
                state = hand_status.get('state', '')
                message = (
                    f"双手离开方向盘超过 {HAND_OFF_WHEEL_DURATION:.0f} 秒"
                    if level == 'danger'
                    else f"单手离开方向盘超过 {HAND_OFF_WHEEL_DURATION:.0f} 秒"
                )
                alerts.append({
                    'source': 'distraction',
                    'type': 'hands_off_wheel' if level == 'danger' else 'single_hand_off_wheel',
                    'severity': level,
                    'timestamp': timestamp,
                    'message': message,
                    'metadata': {
                        'duration': hand_status.get('duration', HAND_OFF_WHEEL_DURATION),
                        'state': state,
                    },
                })

            # 2b. 转身检测
            body_turn = self.check_body_turn(best_kp, timestamp)

            if body_turn:
                alerts.append({
                    'source': 'distraction',
                    'type': 'body_turn',
                    'severity': 'warning',
                    'timestamp': timestamp,
                    'message': "检测到身体大幅扭转/转身",
                    'metadata': self.body_turn_info,
                })

        return {
            'alerts': alerts,
            'objects': objects,
            'hands_off_wheel': hands_off,
            'hand_status': hand_status,
            'body_turn': body_turn,
            'body_turn_info': self.body_turn_info,
            'pose_keypoints': best_kp.tolist() if best_kp is not None else [],
        }

    # ========================================================================
    # 状态重置
    # ========================================================================

    def reset(self):
        """重置所有内部状态（追踪计时器、历史队列、活跃标志）。"""
        self.hands_off_start = None
        self.single_hand_off_start = None
        self.hands_off_active = False
        self.shoulder_positions.clear()
        self.body_turn_active = False
        self.body_turn_info = {}
        self.object_confirm_starts.clear()
        logger.debug("DistractionDetector 状态已重置")
