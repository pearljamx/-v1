"""
几何计算工具模块

提供欧氏距离、眼睛纵横比(EAR)、嘴巴纵横比(MAR)、
旋转向量转欧拉角、向量夹角、向量归一化、视线方向计算等工具函数。
"""

import numpy as np
import math
import cv2
from typing import Tuple, List, Union


def euclidean_distance(
    p1: Union[Tuple[float, float], np.ndarray],
    p2: Union[Tuple[float, float], np.ndarray]
) -> float:
    """
    计算两点之间的欧氏距离。

    Args:
        p1: 第一个点，可以是 (x, y) 元组或 numpy 数组
        p2: 第二个点，可以是 (x, y) 元组或 numpy 数组

    Returns:
        两点之间的欧氏距离
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    return float(np.linalg.norm(p1 - p2))


def compute_ear(eye_points: List[Tuple[float, float]]) -> float:
    """
    计算单只眼睛的 Eye Aspect Ratio (EAR)。

    眼部6个关键点按轮廓顺序排列：
        p1(左) - p2(左上) - p3(右上) - p4(右) - p5(右下) - p6(左下)

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Args:
        eye_points: 6个点的列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6)]

    Returns:
        Eye Aspect Ratio 值，眼睛闭合时趋近于0
    """
    if len(eye_points) != 6:
        raise ValueError(f"eye_points 必须包含6个点，实际提供了{len(eye_points)}个")

    p1 = eye_points[0]
    p2 = eye_points[1]
    p3 = eye_points[2]
    p4 = eye_points[3]
    p5 = eye_points[4]
    p6 = eye_points[5]

    # 两组垂直距离
    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)

    # 水平距离
    horizontal = euclidean_distance(p1, p4)

    if horizontal < 1e-8:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def compute_ear_both(
    left_eye_pts: List[Tuple[float, float]],
    right_eye_pts: List[Tuple[float, float]]
) -> float:
    """
    计算双眼的平均 Eye Aspect Ratio (EAR)。

    Args:
        left_eye_pts: 左眼6个关键点列表
        right_eye_pts: 右眼6个关键点列表

    Returns:
        双眼平均 EAR 值
    """
    left_ear = compute_ear(left_eye_pts)
    right_ear = compute_ear(right_eye_pts)
    return (left_ear + right_ear) / 2.0


def compute_mar(mouth_points: List[Tuple[float, float]]) -> float:
    """
    计算 Mouth Aspect Ratio (MAR)。

    嘴部8个关键点，顺序为：
        [左嘴角, 右嘴角, 上唇顶, 下唇底, 上唇内上, 上唇内下, 下唇内上, 下唇内下]

    MAR = (|p2-p6| + |p3-p7| + |p4-p8|) / (3 * |p0-p1|)

    其中 p0-p7 对应列表索引 0-7。

    Args:
        mouth_points: 8个点的列表

    Returns:
        Mouth Aspect Ratio 值
    """
    if len(mouth_points) != 8:
        raise ValueError(f"mouth_points 必须包含8个点，实际提供了{len(mouth_points)}个")

    # p0: 左嘴角, p1: 右嘴角
    # p2: 上唇顶, p3: 下唇底
    # p4: 上唇内上, p5: 上唇内下, p6: 下唇内上, p7: 下唇内下
    p0 = mouth_points[0]
    p1 = mouth_points[1]
    p2 = mouth_points[2]
    p3 = mouth_points[3]
    p4 = mouth_points[4]
    p5 = mouth_points[5]
    p6 = mouth_points[6]
    p7 = mouth_points[7]

    # 公式: (|p2-p6| + |p3-p7| + |p4-p8|) / (3 * |p0-p1|)
    # 1-indexed 的 p6, p7, p8 对应 0-indexed 的索引 5, 6, 7
    dist_1 = euclidean_distance(p2, p5)   # |p2-p6|
    dist_2 = euclidean_distance(p3, p6)   # |p3-p7|
    dist_3 = euclidean_distance(p4, p7)   # |p4-p8|

    horizontal = euclidean_distance(p0, p1)  # |p0-p1|

    if horizontal < 1e-8:
        return 0.0

    mar = (dist_1 + dist_2 + dist_3) / (3.0 * horizontal)
    return mar


def rotation_vector_to_euler(
    rvec: np.ndarray
) -> Tuple[float, float, float]:
    """
    将 OpenCV solvePnP 返回的旋转向量转换为欧拉角（度）。

    使用 Rodrigues 变换将旋转向量转换为旋转矩阵，
    然后提取 pitch（绕X轴）、yaw（绕Y轴）、roll（绕Z轴）。

    Args:
        rvec: 旋转向量，形状为 (3,1) 或 (3,) 的 numpy 数组

    Returns:
        (pitch, yaw, roll) 三元组，单位为度
    """
    # 确保 rvec 是正确的形状
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3,)

    # Rodrigues 变换：旋转向量 -> 旋转矩阵
    R, _ = cv2.Rodrigues(rvec)

    # 提取欧拉角（弧度）
    # pitch: 绕 X 轴
    # yaw:   绕 Y 轴
    # roll:  绕 Z 轴
    pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    yaw = math.atan2(R[1, 0], R[0, 0])
    roll = math.atan2(R[2, 1], R[2, 2])

    # 转换为度
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    roll_deg = math.degrees(roll)

    return (pitch_deg, yaw_deg, roll_deg)


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算两个向量之间的夹角（度），返回 0-180 度的角度。

    Args:
        v1: 第一个向量（numpy 数组）
        v2: 第二个向量（numpy 数组）

    Returns:
        两个向量之间的夹角，单位为度，范围 [0, 180]
    """
    v1 = np.asarray(v1, dtype=np.float64).flatten()
    v2 = np.asarray(v2, dtype=np.float64).flatten()

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 < 1e-12 or norm_v2 < 1e-12:
        return 0.0

    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)

    # 防止浮点误差导致 cos_angle 超出 [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    向量归一化。

    将向量除以其模长，使其成为单位向量。
    若为零向量，返回原向量以避免除零错误。

    Args:
        v: 输入向量（numpy 数组）

    Returns:
        归一化后的单位向量，或原零向量
    """
    v = np.asarray(v, dtype=np.float64)

    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v.copy()

    return v / norm


def compute_gaze_vector(
    left_eye_center: np.ndarray,
    right_eye_center: np.ndarray,
    nose_bridge: np.ndarray
) -> np.ndarray:
    """
    计算视线方向向量。

    基于双眼中心和鼻梁位置估算视线方向：
        eye_center = (left_eye_center + right_eye_center) / 2
        gaze = normalize(nose_bridge - eye_center)

    Args:
        left_eye_center: 左眼中心点，3D 坐标 numpy 数组
        right_eye_center: 右眼中心点，3D 坐标 numpy 数组
        nose_bridge: 鼻梁点，3D 坐标 numpy 数组

    Returns:
        归一化后的3D视线方向向量
    """
    left_eye_center = np.asarray(left_eye_center, dtype=np.float64)
    right_eye_center = np.asarray(right_eye_center, dtype=np.float64)
    nose_bridge = np.asarray(nose_bridge, dtype=np.float64)

    eye_center = (left_eye_center + right_eye_center) / 2.0
    gaze_direction = nose_bridge - eye_center

    return normalize_vector(gaze_direction)
