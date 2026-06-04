"""
OpenCV可视化绘制函数模块

提供面部检测、特征点、姿态、视线、目标检测等可视化功能。
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 面部相关绘制
# ---------------------------------------------------------------------------

def draw_face_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    """
    绘制面部边界框

    Parameters
    ----------
    image : np.ndarray
        输入BGR图像（原地修改）
    bbox : tuple
        (x, y, w, h) 边界框
    color : tuple
        BGR颜色，默认绿色
    thickness : int
        线宽
    """
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)


def draw_landmarks(image, landmarks, indices=None, color=(0, 255, 0), radius=2):
    """
    绘制面部特征点

    Parameters
    ----------
    image : np.ndarray
        输入BGR图像（原地修改）
    landmarks : np.ndarray
        形状为(N, 2)的特征点数组，每行为(x, y)
    indices : list or None
        要绘制的点子集索引列表，为None则绘制全部点
    color : tuple
        BGR颜色，默认绿色
    radius : int
        点半径（像素）
    """
    if landmarks is None or len(landmarks) == 0:
        return

    if indices is not None:
        pts = landmarks[indices]
    else:
        pts = landmarks

    pts_i32 = np.round(pts).astype(np.int32)
    for pt in pts_i32:
        cv2.circle(image, tuple(pt), radius, color, -1)


def draw_eye_contours(image, left_eye_pts, right_eye_pts, ear_value=None,
                      color=(255, 255, 0)):
    """
    绘制左右眼轮廓，可选叠加EAR值文字

    Parameters
    ----------
    image : np.ndarray
        输入BGR图像（原地修改）
    left_eye_pts : np.ndarray
        左眼特征点 (N, 2)
    right_eye_pts : np.ndarray
        右眼特征点 (N, 2)
    ear_value : tuple or None
        (left_ear, right_ear) 或单个浮点数
    color : tuple
        轮廓BGR颜色，默认青色
    """
    # 左眼 —— 使用凸包或直接多边形
    if left_eye_pts is not None and len(left_eye_pts) >= 3:
        pts = np.round(left_eye_pts).astype(np.int32)
        hull = cv2.convexHull(pts)
        cv2.polylines(image, [hull], isClosed=True, color=color, thickness=1)

    # 右眼
    if right_eye_pts is not None and len(right_eye_pts) >= 3:
        pts = np.round(right_eye_pts).astype(np.int32)
        hull = cv2.convexHull(pts)
        cv2.polylines(image, [hull], isClosed=True, color=color, thickness=1)

    # 检测画面禁止绘制文字，EAR数值仅通过页面面板/JSON展示。


def draw_mouth_contour(image, mouth_pts, mar_value=None, color=(0, 255, 255)):
    """
    绘制嘴部轮廓，可选叠加MAR值文字

    Parameters
    ----------
    image : np.ndarray
        输入BGR图像（原地修改）
    mouth_pts : np.ndarray
        嘴部特征点 (N, 2)
    mar_value : float or None
        Mouth Aspect Ratio 值
    color : tuple
        轮廓BGR颜色，默认黄色
    """
    if mouth_pts is not None and len(mouth_pts) >= 3:
        pts = np.round(mouth_pts).astype(np.int32)
        # 外唇用凸包包围
        hull = cv2.convexHull(pts)
        cv2.polylines(image, [hull], isClosed=True, color=color, thickness=1)

    # 检测画面禁止绘制文字，MAR数值仅通过页面面板/JSON展示。


# ---------------------------------------------------------------------------
# 头部姿态与视线
# ---------------------------------------------------------------------------

def draw_head_pose_axes(image, pose, origin, scale=80):
    """
    在图像上绘制头部姿态坐标轴

    Parameters
    ----------
    image : np.ndarray
        输入BGR图像（原地修改）
    pose : tuple
        (pitch, yaw, roll) 欧拉角，单位：度
    origin : tuple
        (x, y) 轴原点像素坐标（如鼻尖位置）
    scale : int
        轴长度
    """
    pitch, yaw, roll = pose

    # 角度转弧度
    pitch_rad = np.deg2rad(pitch)
    yaw_rad   = np.deg2rad(yaw)
    roll_rad  = np.deg2rad(roll)

    # 方向向量（相机空间，归一化）
    # X轴（右）：roll影响最大
    rx = (np.cos(roll_rad) * np.cos(yaw_rad) -
          np.sin(roll_rad) * np.sin(pitch_rad) * np.sin(yaw_rad))
    ry = np.sin(roll_rad) * np.cos(pitch_rad)
    rz = (np.cos(roll_rad) * np.sin(yaw_rad) +
          np.sin(roll_rad) * np.sin(pitch_rad) * np.cos(yaw_rad))

    # Y轴（上）
    ux = (-np.sin(roll_rad) * np.cos(yaw_rad) -
          np.cos(roll_rad) * np.sin(pitch_rad) * np.sin(yaw_rad))
    uy = np.cos(roll_rad) * np.cos(pitch_rad)
    uz = (-np.sin(roll_rad) * np.sin(yaw_rad) +
          np.cos(roll_rad) * np.sin(pitch_rad) * np.cos(yaw_rad))

    # Z轴（前方，朝外）
    fx = np.cos(pitch_rad) * np.sin(yaw_rad)
    fy = -np.sin(pitch_rad)
    fz = np.cos(pitch_rad) * np.cos(yaw_rad)

    ox, oy = int(origin[0]), int(origin[1])

    # 投影到2D：X/Y在图像平面内，Z分量影响较小，直接用XY分量
    def _endpoint(ux, uy, uz, s):
        return (ox + int(ux * s), oy - int(uy * s))

    # X轴 红色（右）
    cv2.line(image, (ox, oy), _endpoint(rx, ry, rz, scale), (0, 0, 255), 2)
    # Y轴 绿色（上）
    cv2.line(image, (ox, oy), _endpoint(ux, uy, uz, scale), (0, 255, 0), 2)
    # Z轴 蓝色（前方）
    cv2.line(image, (ox, oy), _endpoint(-fx, -fy, -fz, scale), (255, 0, 0), 2)

    # 原点小圆
    cv2.circle(image, (ox, oy), 3, (255, 255, 255), -1)


def draw_gaze_vector(image, origin, direction, length=100,
                     color=(0, 165, 255)):
    """
    绘制视线方向箭头

    Parameters
    ----------
    image : np.ndarray
        输入BGR图像（原地修改）
    origin : tuple
        (x, y) 箭头起点像素坐标
    direction : tuple or np.ndarray
        3D方向向量 (dx, dy, dz)
    length : int
        箭头像素长度
    color : tuple
        BGR颜色，默认橙色
    """
    dx, dy, dz = direction
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    if norm < 1e-8:
        return

    # 归一化并投影到图像平面（dx分量水平，-dy分量垂直）
    ndx = dx / norm
    ndy = dy / norm

    ox, oy = int(origin[0]), int(origin[1])
    end_x = int(ox + ndx * length)
    end_y = int(oy - ndy * length)

    cv2.arrowedLine(image, (ox, oy), (end_x, end_y), color, 2,
                    tipLength=0.2)


# ---------------------------------------------------------------------------
# 目标检测
# ---------------------------------------------------------------------------

def draw_detection_boxes(image, detections, class_names, color_map=None):
    """
    绘制YOLO/目标检测边界框

    Parameters
    ----------
    image : np.ndarray
        输入BGR图像（原地修改）
    detections : list of dict
        每个元素包含:
          - class_id : int      类别ID
          - confidence : float  置信度
          - bbox : tuple        (x1, y1, x2, y2)
    class_names : dict
        {class_id: name}
    color_map : dict or None
        {class_id: (B,G,R)}，为None则自动生成
    """
    if color_map is None:
        # 自动生成颜色（HSV色相均匀分布）
        n = max(len(class_names), 1)
        hues = np.linspace(0, 179, n, endpoint=False, dtype=np.uint8)
        hsv = np.zeros((n, 1, 3), dtype=np.uint8)
        hsv[:, 0, 0] = hues
        hsv[:, 0, 1] = 255
        hsv[:, 0, 2] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        color_map = {cid: tuple(int(c) for c in bgr[i][0]) for i, cid in
                     enumerate(sorted(class_names.keys()))}

    for det in detections:
        class_id = det["class_id"]
        conf = det.get("confidence", 0.0)
        x1, y1, x2, y2 = det["bbox"]
        color = color_map.get(class_id, (0, 255, 0))

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                      color, 2)

        # 检测画面禁止绘制类别名/置信度，使用角标增强框选风格。
        corner = max(8, min(24, int((x2 - x1) * 0.18)))
        x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
        cv2.line(image, (x1_i, y1_i), (x1_i + corner, y1_i), color, 3)
        cv2.line(image, (x1_i, y1_i), (x1_i, y1_i + corner), color, 3)
        cv2.line(image, (x2_i, y1_i), (x2_i - corner, y1_i), color, 3)
        cv2.line(image, (x2_i, y1_i), (x2_i, y1_i + corner), color, 3)
        cv2.line(image, (x1_i, y2_i), (x1_i + corner, y2_i), color, 3)
        cv2.line(image, (x1_i, y2_i), (x1_i, y2_i - corner), color, 3)
        cv2.line(image, (x2_i, y2_i), (x2_i - corner, y2_i), color, 3)
        cv2.line(image, (x2_i, y2_i), (x2_i, y2_i - corner), color, 3)


# ---------------------------------------------------------------------------
# 告警叠加
# ---------------------------------------------------------------------------

def _alert_to_text(alert):
    """把告警对象归一化成 OpenCV 可绘制的短文本。"""
    if isinstance(alert, dict):
        text = alert.get('message') or alert.get('type') or alert.get('alert_type')
    else:
        text = alert
    return str(text) if text is not None else ''


def draw_alert_overlay(image, alerts, position='top'):
    """
    在图像上叠加告警信息，最多显示3条

    Parameters
    ----------
    image : np.ndarray
        输入BGR图像（原地修改）
    alerts : list of str or dict
        告警文本列表，或包含 message/type 字段的告警字典列表
    position : str
        'top' 或 'bottom'
    """
    if not alerts:
        return

    h, w = image.shape[:2]
    max_display = min(len(alerts), 3)
    # 半透明背景矩形
    overlay = image.copy()
    line_h = 28
    pad = 10
    box_h = max_display * line_h + pad * 2
    box_w = min(180, w - 20)

    if position == 'top':
        y0 = 10
    else:
        y0 = h - box_h - 10

    cv2.rectangle(overlay, (10, y0), (10 + box_w, y0 + box_h),
                  (0, 0, 0), -1)
    cv2.rectangle(image, (10, y0), (10 + box_w, y0 + box_h),
                  (0, 0, 255), 2)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    # 检测画面禁止绘制告警文字，仅保留红色半透明告警区域。


# ---------------------------------------------------------------------------
# ROI区域
# ---------------------------------------------------------------------------

def draw_wheel_roi(image, roi_rect, color=(0, 0, 255)):
    """
    绘制方向盘ROI区域

    Parameters
    ----------
    image : np.ndarray
        输入BGR图像（原地修改）
    roi_rect : tuple
        (x, y, w, h) 矩形ROI
    color : tuple
        BGR颜色，默认红色
    """
    x, y, w, h = roi_rect
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # 虚线效果 —— 分段绘制以表示ROI
    dash_len = 10
    for i in range(0, w, dash_len * 2):
        cv2.line(image, (x + i, y), (x + min(i + dash_len, w), y),
                 color, 2)
        cv2.line(image, (x + i, y + h), (x + min(i + dash_len, w), y + h),
                 color, 2)
    for i in range(0, h, dash_len * 2):
        cv2.line(image, (x, y + i), (x, y + min(i + dash_len, h)),
                 color, 2)
        cv2.line(image, (x + w, y + i), (x + w, y + min(i + dash_len, h)),
                 color, 2)


# ---------------------------------------------------------------------------
# PPG波形
# ---------------------------------------------------------------------------

def draw_ppg_waveform(image, ppg_signal, roi):
    """
    在图像指定区域绘制PPG波形迷你图

    Parameters
    ----------
    image : np.ndarray
        输入BGR图像（原地修改）
    ppg_signal : list or np.ndarray
        一维PPG信号序列
    roi : tuple
        (x, y, w, h) 波形显示区域
    """
    if ppg_signal is None or len(ppg_signal) < 2:
        return

    x, y, w, h = roi
    signal = np.asarray(ppg_signal, dtype=np.float32)

    # 归一化到ROI高度范围
    sig_min = np.min(signal)
    sig_max = np.max(signal)
    if sig_max - sig_min < 1e-6:
        normalized = np.full_like(signal, h / 2.0)
    else:
        normalized = (signal - sig_min) / (sig_max - sig_min) * h

    # 生成像素坐标
    n_pts = len(signal)
    xs = np.linspace(x, x + w, n_pts, dtype=np.float32)
    ys = y + h - normalized  # 翻转y轴（图像坐标系y朝下）

    pts = np.column_stack((xs, ys)).astype(np.int32)

    # 绘制半透明背景
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

    # 绘制波形折线
    for i in range(len(pts) - 1):
        cv2.line(image, tuple(pts[i]), tuple(pts[i + 1]),
                 (0, 255, 0), 1, cv2.LINE_AA)

    # 绘制ROI边框
    cv2.rectangle(image, (x, y), (x + w, y + h), (80, 80, 80), 1)
