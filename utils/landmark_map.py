"""
MediaPipe Face Mesh 468关键点 -> dlib 68点映射模块

本模块定义了MediaPipe Face Mesh 468个关键点中与dlib 68点模型相对应的
关键点索引集，并提供辅助函数提取常用面部区域的关键点坐标，
用于眼动检测(EAR)、嘴部检测(MAR)、头部姿态估计、rPPG信号提取等任务。

参考: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md
"""

import numpy as np


# ============================================================================
# 1. 左右眼 EAR (Eye Aspect Ratio) 计算的6个关键点索引 (MediaPipe)
# ============================================================================
# EAR 公式:  EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
# p1(左)-p2(左上)-p3(右上)-p4(右)-p5(右下)-p6(左下)

LEFT_EYE_EAR_IDX = [33, 133, 159, 145, 158, 153]
"""左眼EAR关键点 (MediaPipe索引):
   p1=33 (左/内眼角), p2=133 (左上眼睑),
   p3=159 (右上眼睑), p4=145 (右/外眼角),
   p5=158 (右下眼睑), p6=153 (左下眼睑)"""

RIGHT_EYE_EAR_IDX = [362, 263, 386, 374, 385, 380]
"""右眼EAR关键点 (MediaPipe索引):
   p1=362 (左/外眼角), p2=263 (左上眼睑),
   p3=386 (右上眼睑), p4=374 (右/内眼角),
   p5=385 (右下眼睑), p6=380 (左下眼睑)"""


# ============================================================================
# 2. 嘴部 MAR (Mouth Aspect Ratio) 计算的8个关键点索引
# ============================================================================

MOUTH_MAR_IDX = [61, 291, 13, 14, 84, 17, 314, 0]
"""嘴部MAR关键点 (MediaPipe索引):
   61=左嘴角, 291=右嘴角,
   13=上唇外顶, 14=下唇外底,
   84=上唇内上, 17=上唇内下,
   314=下唇内上, 0=下唇内下"""


# ============================================================================
# 3. 头部姿态估计用6个点索引 (字典形式)
# ============================================================================

HEAD_POSE_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}
"""头部姿态关键点字典:
   鼻尖=1, 下巴=152,
   左眼角=33, 右眼角=263,
   左嘴角=61, 右嘴角=291"""


# ============================================================================
# 4. 前额 ROI 索引 (用于rPPG远程光电容积描记)
# ============================================================================

FOREHEAD_ROI_IDX = [10, 67, 69, 104, 108, 151, 337, 338, 299, 297]
"""前额ROI关键点集 (MediaPipe索引):
   覆盖额头中央区域，用于提取rPPG心率信号"""


# ============================================================================
# 5. 鼻尖和鼻梁索引
# ============================================================================

NOSE_TIP_IDX = 1
"""鼻尖关键点索引 (MediaPipe)"""

NOSE_BRIDGE_IDX = 168
"""鼻梁关键点索引 (MediaPipe)"""


# ============================================================================
# 6. 辅助函数
# ============================================================================

def get_left_eye_ear_points(landmarks_468: np.ndarray) -> np.ndarray:
    """
    从 (N,2) 或 (N,3) 的468点landmarks数组中提取左眼EAR所需的6个点。

    Args:
        landmarks_468: shape为 (468, 2) 或 (468, 3) 的numpy数组，
                       每行是一个关键点的 (x, y) 或 (x, y, z) 坐标。

    Returns:
        shape为 (6, 2) 的numpy数组，按p1~p6顺序排列。
    """
    points = landmarks_468[LEFT_EYE_EAR_IDX]
    return points[..., :2].astype(np.float64)


def get_right_eye_ear_points(landmarks_468: np.ndarray) -> np.ndarray:
    """
    从 (N,2) 或 (N,3) 的468点landmarks数组中提取右眼EAR所需的6个点。

    Args:
        landmarks_468: shape为 (468, 2) 或 (468, 3) 的numpy数组。

    Returns:
        shape为 (6, 2) 的numpy数组，按p1~p6顺序排列。
    """
    points = landmarks_468[RIGHT_EYE_EAR_IDX]
    return points[..., :2].astype(np.float64)


def get_mouth_mar_points(landmarks_468: np.ndarray) -> np.ndarray:
    """
    从 (N,2) 或 (N,3) 的468点landmarks数组中提取嘴部MAR所需的8个点。

    Args:
        landmarks_468: shape为 (468, 2) 或 (468, 3) 的numpy数组。

    Returns:
        shape为 (8, 2) 的numpy数组，按映射表顺序排列。
    """
    points = landmarks_468[MOUTH_MAR_IDX]
    return points[..., :2].astype(np.float64)


def get_head_pose_points(landmarks_468: np.ndarray) -> np.ndarray:
    """
    从 (N,2) 或 (N,3) 的468点landmarks数组中提取头部姿态估计所需的6个点。

    按照 [鼻尖, 下巴, 左眼角, 右眼角, 左嘴角, 右嘴角] 的顺序排列，
    与通用3D人脸模型关键点对齐。

    Args:
        landmarks_468: shape为 (468, 2) 或 (468, 3) 的numpy数组。

    Returns:
        shape为 (6, 2) 的numpy数组，按上述顺序排列。
    """
    order = [
        HEAD_POSE_IDX["nose_tip"],
        HEAD_POSE_IDX["chin"],
        HEAD_POSE_IDX["left_eye_corner"],
        HEAD_POSE_IDX["right_eye_corner"],
        HEAD_POSE_IDX["left_mouth_corner"],
        HEAD_POSE_IDX["right_mouth_corner"],
    ]
    points = landmarks_468[order]
    return points[..., :2].astype(np.float64)


def get_forehead_roi_points(landmarks_468: np.ndarray) -> np.ndarray:
    """
    从 (N,2) 或 (N,3) 的468点landmarks数组中提取前额ROI关键点集。

    Args:
        landmarks_468: shape为 (468, 2) 或 (468, 3) 的numpy数组。

    Returns:
        shape为 (10, 2) 的numpy数组，对应前额区域10个关键点。
    """
    points = landmarks_468[FOREHEAD_ROI_IDX]
    return points[..., :2].astype(np.float64)


def get_eye_centers(landmarks_468: np.ndarray) -> tuple:
    """
    计算左右眼的中心坐标。

    左眼中心 = 左眼EAR 6点的均值
    右眼中心 = 右眼EAR 6点的均值

    Args:
        landmarks_468: shape为 (468, 2) 或 (468, 3) 的numpy数组。

    Returns:
        (left_center, right_center): 两个shape为 (2,) 的numpy数组，
        分别表示左眼和右眼的中心坐标 (x, y)。
    """
    left_points = get_left_eye_ear_points(landmarks_468)
    right_points = get_right_eye_ear_points(landmarks_468)

    left_center = np.mean(left_points, axis=0)
    right_center = np.mean(right_points, axis=0)

    return left_center, right_center
