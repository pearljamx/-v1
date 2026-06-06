"""
头部姿态估计模块
==============
基于 solvePnP 的头部 3D 姿态估计，以及低头/点头动作检测。
"""

import cv2
import numpy as np
import math
from collections import deque
import time
from typing import Optional
from config import *


class HeadPoseEstimator:
    """头部姿态估计器，使用 PnP 求解 3D 旋转并检测点头/低头异常。"""

    def __init__(self, camera_matrix=None, dist_coeffs=None, demo_mode=False):
        self.demo_mode = bool(demo_mode)
        self.head_turn_duration = DEMO_HEAD_TURN_DURATION if self.demo_mode else HEAD_TURN_DURATION

        # 使用config中的相机矩阵或默认值
        self.camera_matrix = camera_matrix or np.array(CAMERA_MATRIX, dtype=np.float64)
        self.dist_coeffs = dist_coeffs or np.array(DIST_COEFFS, dtype=np.float64)

        # 3D面部模型参考点 (毫米坐标) - 6个关键点
        self.model_points_3d = np.array([
            [0.0, 0.0, 0.0],         # 鼻尖
            [0.0, -63.6, -12.5],     # 下巴
            [-43.3, 32.7, -26.0],    # 左眼角
            [43.3, 32.7, -26.0],     # 右眼角
            [-28.9, -28.9, -24.1],   # 左嘴角
            [28.9, -28.9, -24.1],    # 右嘴角
        ], dtype=np.float64)

        # 点头(pitch)历史
        self.pitch_history = deque(maxlen=30)
        # 点头开始时间
        self.nodding_start = None
        # 转头开始时间
        self.head_turn_start = None

    # ------------------------------------------------------------------
    # 姿态估计
    # ------------------------------------------------------------------
    def estimate(self, image_points_2d: np.ndarray) -> tuple:
        """
        估计头部姿态

        参数:
          image_points_2d: shape (6, 2) numpy array
            点序: 鼻尖, 下巴, 左眼角, 右眼角, 左嘴角, 右嘴角

        返回:
          (pitch, yaw, roll) 欧拉角(度)

        如果PnP求解失败返回 (0, 0, 0)
        """
        image_points_2d = np.asarray(image_points_2d, dtype=np.float32).reshape((6, 2))

        success, rvec, _ = cv2.solvePnP(
            self.model_points_3d,
            image_points_2d,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return (0.0, 0.0, 0.0)

        pitch, yaw, roll = self.rotation_vector_to_euler(rvec)
        return (float(pitch), float(yaw), float(roll))

    # ------------------------------------------------------------------
    # 旋转向量 -> 欧拉角
    # ------------------------------------------------------------------
    def rotation_vector_to_euler(self, rvec):
        """
        旋转向量转欧拉角(度)

        使用 Rodrigues 变换 + atan2 提取欧拉角。
        注意: 本系统的3D面部模型坐标系与相机坐标系之间存在~180°旋转，
        因此正面人脸时 roll ≈ ±170° (而非 0°) 是正常现象。
        实际使用时关注 pitch (俯仰) 和 yaw (偏航) 的变化即可。

        参数:
          rvec: 旋转向量，形状 (3,1) 或 (3,) 的 numpy 数组

        返回:
          (pitch, yaw, roll) 三元组，单位为度
            - pitch: 头部俯仰角 (绕X轴，正值=仰头，负值=低头)
            - yaw:   头部偏航角 (绕Y轴，正值=右转)
            - roll:  头部翻滚角 (绕Z轴，正面≈±170°)
        """
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3,)

        # Rodrigues 变换: 旋转向量 -> 旋转矩阵
        R, _ = cv2.Rodrigues(rvec)

        # 从旋转矩阵提取欧拉角（弧度）
        # 使用 extrinsic ZYX 约定: R = Rz(roll) @ Ry(yaw) @ Rx(pitch)
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        if sy > 1e-6:
            pitch = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(-R[2, 0], sy)
            roll = math.atan2(R[1, 0], R[0, 0])
        else:
            # 万向节锁边界情况
            pitch = math.atan2(-R[1, 2], R[1, 1])
            yaw = math.atan2(-R[2, 0], sy)
            roll = 0.0

        # 转换为度
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        roll_deg = math.degrees(roll)

        return (pitch_deg, yaw_deg, roll_deg)

    # ------------------------------------------------------------------
    # 点头 / 低头检测
    # ------------------------------------------------------------------
    def check_nodding(self, pitch: float, timestamp: float) -> Optional[dict]:
        """
        检测点头动作

        当 |pitch| 超过 PITCH_THRESHOLD 且持续时间超过 PITCH_DURATION
        时，生成告警。

        参数:
          pitch: 当前帧的俯仰角（度）
          timestamp: 当前帧的时间戳（秒）

        返回:
          满足告警条件时返回告警字典，否则返回 None
        """
        # 记录俯仰角历史
        self.pitch_history.append(float(pitch))

        is_extreme = abs(pitch) > PITCH_THRESHOLD

        if is_extreme:
            if self.nodding_start is None:
                # 首次进入异常区域，记录起始时间
                self.nodding_start = timestamp

            # 检查是否已持续足够长时间
            elapsed = timestamp - self.nodding_start
            if elapsed >= PITCH_DURATION:
                direction = "低头" if pitch < 0 else "仰头"
                return {
                    'source': 'head_pose',
                    'type': 'nodding',
                    'severity': 'warning',
                    'timestamp': timestamp,
                    'message': f"驾驶员持续{direction}，俯仰角 {pitch:.1f}°，已持续 {elapsed:.1f}s",
                    'metadata': {'pitch': float(pitch), 'duration': elapsed},
                }
        else:
            # 俯仰角回到正常范围，重置状态
            if self.nodding_start is not None:
                self.nodding_start = None

        return None

    # ------------------------------------------------------------------
    # 转头检测
    # ------------------------------------------------------------------
    def check_head_turn(self, yaw: float, timestamp: float) -> Optional[dict]:
        """
        检测持续转头分心。

        正常驾驶允许短暂扫视后视镜或侧方，因此只在 yaw 明显偏离并持续
        HEAD_TURN_DURATION 后报警。
        """
        yaw_abs = abs(float(yaw))
        is_turning = yaw_abs > HEAD_TURN_YAW_THRESHOLD

        if is_turning:
            if self.head_turn_start is None:
                self.head_turn_start = timestamp

            elapsed = timestamp - self.head_turn_start
            if elapsed >= self.head_turn_duration:
                direction = "右转头" if yaw > 0 else "左转头"
                severity = "danger" if yaw_abs >= HEAD_TURN_DANGER_YAW else "warning"
                return {
                    'source': 'head_pose',
                    'type': 'head_turn',
                    'severity': severity,
                    'timestamp': timestamp,
                    'message': (
                        f"驾驶员持续{direction}，偏航角 {yaw:.1f}°，"
                        f"已持续 {elapsed:.1f}s"
                    ),
                    'metadata': {
                        'yaw': float(yaw),
                        'duration': float(elapsed),
                        'threshold_yaw': float(HEAD_TURN_YAW_THRESHOLD),
                        'threshold_seconds': float(self.head_turn_duration),
                        'demo_mode': bool(self.demo_mode),
                        'direction': direction,
                    },
                }
        else:
            self.head_turn_start = None

        return None

    # ------------------------------------------------------------------
    # 重置
    # ------------------------------------------------------------------
    def reset(self):
        """重置所有内部状态（历史记录、检测标志）。"""
        self.pitch_history.clear()
        self.nodding_start = None
        self.head_turn_start = None
