"""
视线方向检测模块
================
基于面部关键点估计驾驶员视线方向，检测视线是否离开路面。

原理:
  1. 从面部关键点计算双眼中心和鼻梁点
  2. 计算图像空间的2D视线向量 (eye_center -> nose_bridge)
  3. 用头部姿态 (pitch, yaw) 补偿，得到3D世界空间视线方向
  4. 计算视线与正前方 [0, 0, 1] 的夹角
  5. 当夹角超出 GAZE_ANGLE_THRESHOLD 且持续超过 GAZE_DURATION 时触发告警

依赖:
  - FaceDetector.detect() 的面部检测结果 (face_result)
  - HeadPoseEstimator.estimate() 的头部姿态 (pitch, yaw, roll)
"""

import numpy as np
import math
from collections import deque
from typing import Optional
from config import *

# MediaPipe 关键点索引 (与 utils/landmark_map.py 保持一致)
NOSE_BRIDGE_IDX = 168
LEFT_EYE_EAR_IDX = [33, 133, 159, 145, 158, 153]
RIGHT_EYE_EAR_IDX = [362, 263, 386, 374, 385, 380]


class GazeEstimator:
    """视线方向估计器

    结合面部关键点和头部姿态，估计驾驶员的3D视线方向，
    并检测视线是否持续偏离道路前方。
    """

    def __init__(self):
        # 视线偏离历史 (角度值，用于平滑/统计)
        self.gaze_angle_history = deque(maxlen=30)

        # 视线偏离开始时间 (秒)，None 表示当前未处于偏离状态
        self.gaze_deviation_start = None

        # 前向向量 — 假设摄像机朝向道路前方，[0, 0, 1] 表示“注视路面”
        self.forward_vector = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # ==================================================================
    # 视线方向估计
    # ==================================================================

    def estimate(self, face_result: dict, head_pose: tuple) -> dict:
        """
        估计视线方向。

        参数:
            face_result: FaceDetector.detect() 的返回结果字典。
                包含 landmarks_468 或 landmarks_5，以及 image_shape 等字段。
            head_pose: (pitch, yaw, roll) 头部姿态欧拉角，单位：度。
                来自 HeadPoseEstimator.estimate()。

        返回:
            dict:
            {
                'gaze_vector': np.ndarray (3,),   # 3D 视线方向单位向量
                'gaze_angle': float,               # 视线偏离正前方的角度(度)
                'eye_center': np.ndarray (2,),     # 双眼中心像素坐标
                'nose_bridge': np.ndarray (2,),    # 鼻梁像素坐标
            }
            当 face_result 为 None 时返回默认零值。
        """
        # ---- 处理空输入 ----
        if face_result is None:
            return {
                'gaze_vector': np.array([0.0, 0.0, 1.0], dtype=np.float64),
                'gaze_angle': 0.0,
                'eye_center': np.zeros(2, dtype=np.float64),
                'nose_bridge': np.zeros(2, dtype=np.float64),
            }

        landmarks_468 = face_result.get('landmarks_468')
        landmarks_5 = face_result.get('landmarks_5')

        # ---- 提取双眼中心和鼻梁点 ----
        if landmarks_468 is not None and len(landmarks_468) >= 468:
            # MediaPipe 后端 — 468 个关键点
            left_eye_pts = landmarks_468[LEFT_EYE_EAR_IDX]
            right_eye_pts = landmarks_468[RIGHT_EYE_EAR_IDX]

            # 取前两列 (x, y)，忽略可能的第三列 z
            left_eye_pts = left_eye_pts[..., :2].astype(np.float64)
            right_eye_pts = right_eye_pts[..., :2].astype(np.float64)

            left_center = np.mean(left_eye_pts, axis=0)    # (2,)
            right_center = np.mean(right_eye_pts, axis=0)  # (2,)

            nb = landmarks_468[NOSE_BRIDGE_IDX]
            nose_bridge = np.array([float(nb[0]), float(nb[1])], dtype=np.float64)

        elif landmarks_5 is not None and len(landmarks_5) >= 5:
            # Yunet 后端 — 5 个关键点: [右眼, 左眼, 鼻尖, 右嘴角, 左嘴角]
            left_center = np.array([float(landmarks_5[1, 0]),
                                    float(landmarks_5[1, 1])], dtype=np.float64)
            right_center = np.array([float(landmarks_5[0, 0]),
                                     float(landmarks_5[0, 1])], dtype=np.float64)

            # 鼻梁 ≈ 鼻尖上方 15 px (Yunet 只有鼻尖，无鼻梁关键点)
            nose_bridge = np.array([float(landmarks_5[2, 0]),
                                    float(landmarks_5[2, 1]) - 15.0],
                                   dtype=np.float64)
        else:
            # 无可用的关键点数据
            return {
                'gaze_vector': np.array([0.0, 0.0, 1.0], dtype=np.float64),
                'gaze_angle': 0.0,
                'eye_center': np.zeros(2, dtype=np.float64),
                'nose_bridge': np.zeros(2, dtype=np.float64),
            }

        eye_center = (left_center + right_center) / 2.0

        # ---- 计算 2D 视线向量 (图像空间): gaze = nose_bridge - eye_center ----
        gaze_2d = nose_bridge - eye_center  # (dx, dy)

        # ---- 用头部姿态补偿，得到 3D 视线方向 ----
        gaze_3d = self._compensate_with_head_pose(gaze_2d, head_pose)

        # ---- 计算与正前方的夹角 ----
        dot = np.dot(gaze_3d, self.forward_vector)
        dot = np.clip(dot, -1.0, 1.0)
        gaze_angle = float(np.degrees(np.arccos(dot)))

        return {
            'gaze_vector': gaze_3d,
            'gaze_angle': gaze_angle,
            'eye_center': eye_center,
            'nose_bridge': nose_bridge,
        }

    # ==================================================================
    # 头部姿态补偿
    # ==================================================================

    def _compensate_with_head_pose(self, gaze_vec: np.ndarray,
                                   head_pose: tuple) -> np.ndarray:
        """
        用头部姿态补偿视线方向。

        思路:
          2D 图像空间的视线向量反映了眼球相对于头部朝向的偏移。
          头部姿态 (pitch, yaw) 给出了头部在世界/摄像机空间中的朝向。
          世界视线方向 = 头部旋转矩阵 @ 眼球视线方向。

        参数:
            gaze_vec: 2D 视线向量 (dx, dy)，从图像直接计算。
                dx > 0  → 眼球在图像中向右偏 (驾驶员自身向左看)
                dy < 0  → 鼻梁在眼睛上方 (正常前视姿态)
            head_pose: (pitch, yaw, roll) 头部姿态欧拉角，单位：度。

        返回:
            补偿后的 3D 单位视线向量，shape (3,)。
        """
        gx, gy = float(gaze_vec[0]), float(gaze_vec[1])

        # ---- 2D 视线向量归一化 ----
        norm_2d = math.sqrt(gx * gx + gy * gy)
        if norm_2d < 1e-6:
            # 2D 视线向量过短 (鼻梁与眼睛几乎重合)，无法可靠估计眼球方向，
            # 退化为默认前视方向。
            eye_gaze_3d = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            gx /= norm_2d
            gy /= norm_2d

            # 将 2D 图像空间向量映射到 3D 摄像机空间:
            #   gx → 水平分量 (x 轴)，保持不变
            #   gy → 垂直分量: gy < 0 表示鼻梁在上方 (正常前视)，
            #         对应 z 轴正方向，因此取 -gy 作为 z 分量
            #   y 分量默认为 0 (2D 图像无法直接区分俯仰)
            eye_gaze_3d = np.array([gx, 0.0, max(-gy, 0.1)], dtype=np.float64)
            eye_gaze_3d /= np.linalg.norm(eye_gaze_3d)

        # ---- 构建头部姿态旋转矩阵 ----
        pitch, yaw, roll = float(head_pose[0]), float(head_pose[1]), float(head_pose[2])

        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)

        # R_x(pitch): 绕 X 轴旋转 (点头/仰头)
        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
        Rx = np.array([
            [1.0, 0.0,  0.0],
            [0.0,  cp,  -sp],
            [0.0,  sp,   cp],
        ], dtype=np.float64)

        # R_y(yaw): 绕 Y 轴旋转 (左右转头)
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
        Ry = np.array([
            [ cy, 0.0,  sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0,  cy],
        ], dtype=np.float64)

        # 组合旋转: 先绕 X (pitch)，再绕 Y (yaw)
        R = Ry @ Rx

        # 应用头部旋转: world_gaze = R @ eye_gaze
        compensated = R @ eye_gaze_3d
        compensated /= np.linalg.norm(compensated)

        return compensated

    # ==================================================================
    # 视线偏离检测
    # ==================================================================

    def check_gaze_deviation(self, gaze_angle: float,
                             timestamp: float) -> Optional[dict]:
        """
        检测视线是否持续偏离道路。

        当 gaze_angle 超过 GAZE_ANGLE_THRESHOLD 且持续时间
        超过 GAZE_DURATION 时生成告警。

        参数:
            gaze_angle: 当前帧的视线偏离角度 (度)，来自 estimate()。
            timestamp:  当前帧的时间戳 (秒)。

        返回:
            满足告警条件 → dict:
            {
                'source':    'gaze',
                'type':      'gaze_deviation',
                'severity':  'warning',
                'timestamp': float,
                'message':   str,
                'metadata':  {'gaze_angle': float, 'duration': float},
            }
            未触发告警 → None。
        """
        # 记录历史 (用于后续可能的平滑/趋势分析)
        self.gaze_angle_history.append(float(gaze_angle))

        is_deviated = gaze_angle > GAZE_ANGLE_THRESHOLD

        if is_deviated:
            if self.gaze_deviation_start is None:
                # 首次检测到偏离，记录起始时间
                self.gaze_deviation_start = timestamp

            elapsed = timestamp - self.gaze_deviation_start
            if elapsed >= GAZE_DURATION:
                return {
                    'source': 'gaze',
                    'type': 'gaze_deviation',
                    'severity': 'warning',
                    'timestamp': timestamp,
                    'message': (
                        f"视线偏离路面 {gaze_angle:.1f}°，"
                        f"已持续 {elapsed:.1f}s"
                    ),
                    'metadata': {
                        'gaze_angle': float(gaze_angle),
                        'duration': elapsed,
                    },
                }
        else:
            # 视线回到正常范围，重置计时
            self.gaze_deviation_start = None

        return None

    # ==================================================================
    # 重置
    # ==================================================================

    def reset(self):
        """重置所有内部状态 (历史记录、偏离计时)。"""
        self.gaze_angle_history.clear()
        self.gaze_deviation_start = None
