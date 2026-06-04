"""
面部检测与特征点提取核心模块
==========================
采用策略模式，支持MediaPipe Face Mesh、dlib 68点和OpenCV Yunet三种后端。
MediaPipe为主方案（468个3D关键点），dlib为备选方案（68个关键点），Yunet为回退方案（5个关键点）。

后端功能对比:
  - mediapipe: 468点Face Mesh，支持EAR/MAR/头部姿态/rPPG前额提取
  - dlib:      68点关键点检测，支持EAR/MAR/头部姿态/rPPG前额提取
  - yunet:     5点快速检测，支持基本头部姿态，EAR/MAR使用近似估计

用法:
    detector = FaceDetector(backend='auto')          # 自动选择
    detector = FaceDetector(backend='mediapipe')      # 强制 MediaPipe
    detector = FaceDetector(backend='dlib')           # 强制 dlib
    detector = FaceDetector(backend='yunet')          # 强制 Yunet
    result = detector.detect(frame)
    if result:
        eyes = detector.get_eye_regions(result)
"""

from __future__ import annotations

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

# ---------------------------------------------------------------------------
# 可选依赖检测 (MediaPipe 0.10.x tasks API)
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import base_options as mp_base_options
    _MP_AVAILABLE = True
    _MP_TASKS_API = True
except ImportError:
    try:
        import mediapipe as mp
        _MP_AVAILABLE = True
        _MP_TASKS_API = False
    except ImportError:
        _MP_AVAILABLE = False
        _MP_TASKS_API = False
        mp = None

# ---------------------------------------------------------------------------
# 可选依赖检测 (dlib)
# ---------------------------------------------------------------------------
try:
    import dlib
    _DLIB_AVAILABLE = True
except ImportError:
    _DLIB_AVAILABLE = False
    dlib = None

from config import BASE_DIR

# MediaPipe Face Landmarker 模型文件路径
_MODEL_FILENAME = "face_landmarker.task"
_MP_MODEL_PATH = str(BASE_DIR / "models_data" / _MODEL_FILENAME)

# Yunet ONNX 模型路径
YUNET_MODEL_PATH = str(BASE_DIR / "models_data" / "face_detection_yunet_2023mar.onnx")

# dlib 68点 landmark 模型路径 (多位置搜索)
_DLIB_68_LANDMARK_PATHS = [
    str(BASE_DIR / "models_data" / "shape_predictor_68_face_landmarks.dat"),
    os.path.join(
        os.path.dirname(os.path.abspath(dlib.__file__)) if dlib is not None else "",
        "shape_predictor_68_face_landmarks.dat",
    ),
]
# face_recognition_models 包安装路径 (Windows pip 安装)
try:
    import face_recognition_models
    _FR_MODELS_DIR = os.path.join(
        os.path.dirname(face_recognition_models.__file__), "models"
    )
    _DLIB_68_LANDMARK_PATHS.insert(
        0, os.path.join(_FR_MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
    )
    _DLIB_FACE_RECOG_PATH = os.path.join(
        _FR_MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat"
    )
    import site
    for sp in site.getsitepackages():
        candidate = os.path.join(
            sp, "face_recognition_models", "models",
            "shape_predictor_68_face_landmarks.dat"
        )
        if candidate not in _DLIB_68_LANDMARK_PATHS:
            _DLIB_68_LANDMARK_PATHS.append(candidate)
except ImportError:
    _DLIB_FACE_RECOG_PATH = None

# 类型别名
FaceResult = Dict[str, Any]


def _ensure_ascii_model_path(original_path):
    """
    确保模型路径为纯ASCII（MediaPipe C++后端不支持中文路径）。

    如果原始路径含非ASCII字符，自动复制到系统临时目录。
    返回可用路径或None。
    """
    if os.path.exists(original_path):
        try:
            original_path.encode('ascii')
            return original_path  # 已是纯ASCII
        except UnicodeEncodeError:
            pass
        # 复制到临时目录
        import shutil, tempfile
        temp_dir = os.path.join(tempfile.gettempdir(), 'driver_monitor_models')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, _MODEL_FILENAME)
        if not os.path.exists(temp_path) or (
            os.path.getmtime(original_path) > os.path.getmtime(temp_path)
        ):
            shutil.copy2(original_path, temp_path)
            print(f"[FaceDetector] 模型已复制到ASCII路径: {temp_path}")
        return temp_path
    return None


# ==============================================================================
# dlib 68-point 后端
# ==============================================================================

# dlib 68点标准索引定义
_DLIB_LEFT_EYE_EAR_IDX = [36, 37, 38, 39, 40, 41]   # 左眼 EAR 6点
_DLIB_RIGHT_EYE_EAR_IDX = [42, 43, 44, 45, 46, 47]  # 右眼 EAR 6点
_DLIB_MOUTH_MAR_IDX = [48, 54, 51, 57, 62, 66, 64, 60]  # 嘴部 MAR 8点
_DLIB_NOSE_TIP_IDX = 30      # 鼻尖
_DLIB_NOSE_BRIDGE_IDX = 27   # 鼻梁
_DLIB_CHIN_IDX = 8           # 下巴
_DLIB_LEFT_EYE_CORNER = 36   # 左眼角
_DLIB_RIGHT_EYE_CORNER = 45  # 右眼角
_DLIB_LEFT_MOUTH_CORNER = 48 # 左嘴角
_DLIB_RIGHT_MOUTH_CORNER = 54 # 右嘴角
_DLIB_FOREHEAD_IDX = list(range(17, 27))  # 前额 10 点 (17-26)


class DlibBackend:
    """
    基于 dlib 的面部检测与 68 点关键点提取后端。

    使用 dlib.get_frontal_face_detector() 进行人脸检测，
    使用 dlib.shape_predictor() 提取 68 个面部关键点。

    提供与 MediaPipe 后端一致的接口方法，可在 MediaPipe 不可用时作为备选方案。
    """

    def __init__(self, min_detection_confidence=0.5,
                 landmark_model_path=None):
        """
        初始化 dlib 后端。

        参数:
            min_detection_confidence: float
                最小检测置信度（dlib 检测器通常返回固定置信度，此值保留用于接口统一）
            landmark_model_path: str | None
                68点关键点模型文件路径，为 None 则自动搜索

        异常:
            ImportError: dlib 未安装
            FileNotFoundError: 找不到 68 点 landmark 模型文件
        """
        if not _DLIB_AVAILABLE:
            raise ImportError(
                "dlib 未安装。请安装预编译wheel: "
                "pip install dlib-bin 或 pip install dlib"
            )

        self.min_detection_confidence = float(min_detection_confidence)

        # ---- 人脸检测器 ----
        self.face_detector = dlib.get_frontal_face_detector()

        # ---- 68点关键点预测器 ----
        if landmark_model_path is not None:
            resolved_path = landmark_model_path
        else:
            resolved_path = None
            for candidate in _DLIB_68_LANDMARK_PATHS:
                if os.path.exists(candidate):
                    resolved_path = candidate
                    break

        if resolved_path is None or not os.path.exists(resolved_path):
            searched = "\n  ".join(_DLIB_68_LANDMARK_PATHS)
            raise FileNotFoundError(
                f"找不到 dlib 68点 landmark 模型文件。已搜索:\n  {searched}\n"
                f"请将 shape_predictor_68_face_landmarks.dat 放入 models_data/ 目录"
            )

        self.landmark_model_path = resolved_path
        self.shape_predictor = dlib.shape_predictor(str(resolved_path))

    # ==================================================================
    # 面部检测
    # ==================================================================

    def detect(self, image: np.ndarray) -> Optional[FaceResult]:
        """
        检测图像中的面部并提取 68 个关键点。

        参数:
            image: BGR 格式的 numpy 图像数组 (H, W, 3)

        返回:
            dict: {
                'bbox':          (x, y, w, h) int,
                'landmarks_68':  np.ndarray (68, 2) float64,
                'landmarks_468': None,
                'landmarks_5':   None,
                'score':         float,
                'image_shape':   (h, w),
                'backend':       'dlib',
            }
            None -- 未检测到面部
        """
        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]

        # dlib 检测器需要灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 检测人脸
        faces = self.face_detector(gray, 0)
        if len(faces) == 0:
            return None

        # 取面积最大的人脸
        face_rect = max(faces, key=lambda r: r.width() * r.height())

        # 68 点关键点
        shape = self.shape_predictor(gray, face_rect)
        num_landmarks = shape.num_parts  # 应该为 68
        landmarks_68 = np.zeros((num_landmarks, 2), dtype=np.float64)
        for i in range(num_landmarks):
            landmarks_68[i, 0] = shape.part(i).x
            landmarks_68[i, 1] = shape.part(i).y

        # 包围盒
        bx = max(0, face_rect.left())
        by = max(0, face_rect.top())
        bw = min(w - bx, max(1, face_rect.width()))
        bh = min(h - by, max(1, face_rect.height()))

        return {
            'bbox':          (bx, by, bw, bh),
            'landmarks_68':  landmarks_68,
            'landmarks_468': None,
            'landmarks_5':   None,
            'score':         1.0,  # dlib 检测器不返回置信度
            'image_shape':   (h, w),
            'backend':       'dlib',
        }

    # ==================================================================
    # 面部特征区域提取
    # ==================================================================

    def get_eye_regions(
        self, face_result: FaceResult
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        返回左右眼 EAR 计算所需的 6 个关键点。

        点序: [内眼角(左), 上睑左, 上睑右, 外眼角(右), 下睑右, 下睑左]
        左眼: dlib 索引 36-41
        右眼: dlib 索引 42-47

        返回:
            (left_eye_pts_6, right_eye_pts_6)
        """
        if face_result is None:
            return ([(0.0, 0.0)] * 6, [(0.0, 0.0)] * 6)

        landmarks_68 = face_result.get('landmarks_68')
        if landmarks_68 is None:
            return ([(0.0, 0.0)] * 6, [(0.0, 0.0)] * 6)

        def _pts_6(indices):
            return [(float(landmarks_68[i, 0]), float(landmarks_68[i, 1]))
                    for i in indices]

        return (
            _pts_6(_DLIB_LEFT_EYE_EAR_IDX),
            _pts_6(_DLIB_RIGHT_EYE_EAR_IDX),
        )

    def get_mouth_region(self, face_result: FaceResult) -> List[Tuple[float, float]]:
        """
        返回 MAR 计算的 8 个嘴部关键点。

        点序: [左嘴角(48), 右嘴角(54), 上唇外顶(51), 下唇外底(57),
               上唇内上(62), 下唇内下(66), 下唇内上(64), 上唇内下(60)]

        返回:
            8 个 (x, y) 元组的列表
        """
        if face_result is None:
            return [(0.0, 0.0)] * 8

        landmarks_68 = face_result.get('landmarks_68')
        if landmarks_68 is None:
            return [(0.0, 0.0)] * 8

        return [(float(landmarks_68[i, 0]), float(landmarks_68[i, 1]))
                for i in _DLIB_MOUTH_MAR_IDX]

    def get_head_pose_points(self, face_result: FaceResult) -> np.ndarray:
        """
        返回头部姿态估计所需的 6 个 2D 点。

        点序: [鼻尖(30), 下巴(8), 左眼角(36), 右眼角(45),
               左嘴角(48), 右嘴角(54)]

        返回:
            shape (6, 2) 的 numpy float64 数组
        """
        if face_result is None:
            return np.zeros((6, 2), dtype=np.float64)

        landmarks_68 = face_result.get('landmarks_68')
        if landmarks_68 is None:
            return np.zeros((6, 2), dtype=np.float64)

        hp_idx = [
            _DLIB_NOSE_TIP_IDX,    # 30: 鼻尖
            _DLIB_CHIN_IDX,        # 8:  下巴
            _DLIB_LEFT_EYE_CORNER, # 36: 左眼角
            _DLIB_RIGHT_EYE_CORNER,# 45: 右眼角
            _DLIB_LEFT_MOUTH_CORNER,# 48: 左嘴角
            _DLIB_RIGHT_MOUTH_CORNER,# 54: 右嘴角
        ]
        pts = np.zeros((6, 2), dtype=np.float64)
        for i, idx in enumerate(hp_idx):
            pts[i, 0] = landmarks_68[idx, 0]
            pts[i, 1] = landmarks_68[idx, 1]
        return pts

    def get_nose_tip(self, face_result: FaceResult) -> Tuple[float, float]:
        """
        返回鼻尖坐标 (dlib 索引 30)。

        返回:
            (x, y) 元组
        """
        if face_result is None:
            return (0.0, 0.0)

        landmarks_68 = face_result.get('landmarks_68')
        if landmarks_68 is None:
            return (0.0, 0.0)

        return (float(landmarks_68[_DLIB_NOSE_TIP_IDX, 0]),
                float(landmarks_68[_DLIB_NOSE_TIP_IDX, 1]))

    def get_nose_bridge(self, face_result: FaceResult) -> Tuple[float, float]:
        """
        返回鼻梁坐标 (dlib 索引 27)。

        返回:
            (x, y) 元组
        """
        if face_result is None:
            return (0.0, 0.0)

        landmarks_68 = face_result.get('landmarks_68')
        if landmarks_68 is None:
            return (0.0, 0.0)

        return (float(landmarks_68[_DLIB_NOSE_BRIDGE_IDX, 0]),
                float(landmarks_68[_DLIB_NOSE_BRIDGE_IDX, 1]))

    def get_forehead_roi(self, face_result: FaceResult,
                          image: np.ndarray) -> np.ndarray:
        """
        提取前额 ROI 区域图像，基于 dlib 前额 10 个关键点 (索引 17-26)。

        参数:
            face_result: detect() 返回的面部检测结果字典
            image:       原始 BGR 格式图像

        返回:
            前额 ROI 区域的 BGR numpy 数组
        """
        if face_result is None or image is None or image.size == 0:
            return np.array([])

        h, w = image.shape[:2]
        landmarks_68 = face_result.get('landmarks_68')

        if landmarks_68 is None:
            return np.array([])

        fh_pts = landmarks_68[_DLIB_FOREHEAD_IDX]  # (10, 2)
        x_min = int(np.min(fh_pts[:, 0]))
        y_min = int(np.min(fh_pts[:, 1]))
        x_max = int(np.max(fh_pts[:, 0]))
        y_max = int(np.max(fh_pts[:, 1]))

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        if x_max <= x_min or y_max <= y_min:
            return np.array([])

        return image[y_min:y_max, x_min:x_max].copy()

    # ==================================================================
    # 可视化
    # ==================================================================

    def draw_landmarks(self, image: np.ndarray, face_result: FaceResult,
                       draw_eyes: bool = True,
                       draw_mouth: bool = True) -> np.ndarray:
        """
        在图像上绘制 dlib 68 点面部关键点和包围盒。

        参数:
            image:       BGR 格式 numpy 图像数组 (原地修改)
            face_result: detect() 返回的面部检测结果字典
            draw_eyes:   是否绘制眼部关键点
            draw_mouth:  是否绘制嘴部关键点

        返回:
            标注后的图像
        """
        if image is None or face_result is None:
            return image

        h, w = image.shape[:2]
        bbox = face_result.get('bbox')
        landmarks_68 = face_result.get('landmarks_68')

        # ---- 包围盒 ----
        if bbox is not None:
            x, y, bw_box, bh_box = bbox
            cv2.rectangle(image, (x, y), (x + bw_box, y + bh_box),
                          (0, 255, 0), 2)

        # ---- 68 点关键点 ----
        if landmarks_68 is not None:
            for i in range(len(landmarks_68)):
                px, py = int(landmarks_68[i, 0]), int(landmarks_68[i, 1])
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(image, (px, py), 1, (0, 255, 0), -1)

            # 眼部高亮
            if draw_eyes:
                for idx in _DLIB_LEFT_EYE_EAR_IDX + _DLIB_RIGHT_EYE_EAR_IDX:
                    if 0 <= idx < len(landmarks_68):
                        px, py = (int(landmarks_68[idx, 0]),
                                  int(landmarks_68[idx, 1]))
                        if 0 <= px < w and 0 <= py < h:
                            cv2.circle(image, (px, py), 2, (255, 100, 0), -1)

                # 绘制眼部轮廓线
                for eye_idx in [_DLIB_LEFT_EYE_EAR_IDX, _DLIB_RIGHT_EYE_EAR_IDX]:
                    pts = [(int(landmarks_68[i, 0]), int(landmarks_68[i, 1]))
                           for i in eye_idx
                           if 0 <= int(landmarks_68[i, 0]) < w
                           and 0 <= int(landmarks_68[i, 1]) < h]
                    if len(pts) >= 2:
                        for j in range(len(pts) - 1):
                            cv2.line(image, pts[j], pts[j + 1],
                                     (255, 100, 0), 1, cv2.LINE_AA)
                        cv2.line(image, pts[-1], pts[0],
                                 (255, 100, 0), 1, cv2.LINE_AA)

            # 嘴部高亮
            if draw_mouth:
                for idx in _DLIB_MOUTH_MAR_IDX:
                    if 0 <= idx < len(landmarks_68):
                        px, py = (int(landmarks_68[idx, 0]),
                                  int(landmarks_68[idx, 1]))
                        if 0 <= px < w and 0 <= py < h:
                            cv2.circle(image, (px, py), 2, (0, 100, 255), -1)

                # 外唇轮廓 (索引 48-59)
                outer_lip = list(range(48, 60))
                pts = [(int(landmarks_68[i, 0]), int(landmarks_68[i, 1]))
                       for i in outer_lip
                       if 0 <= int(landmarks_68[i, 0]) < w
                       and 0 <= int(landmarks_68[i, 1]) < h]
                if len(pts) >= 2:
                    for j in range(len(pts) - 1):
                        cv2.line(image, pts[j], pts[j + 1],
                                 (0, 100, 255), 1, cv2.LINE_AA)
                    cv2.line(image, pts[-1], pts[0],
                             (0, 100, 255), 1, cv2.LINE_AA)

        return image

    # ==================================================================
    # 资源管理
    # ==================================================================

    def close(self):
        """释放后端资源。"""
        self.face_detector = None
        self.shape_predictor = None

    def release(self):
        """释放资源 (close 的别名)。"""
        self.close()

    def __del__(self):
        """析构时自动释放资源。"""
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return (f"DlibBackend(landmark_model='{self.landmark_model_path}', "
                f"confidence={self.min_detection_confidence})")


class FaceDetector:
    """
    面部检测器，支持多后端:
    - mediapipe: MediaPipe Face Mesh (468点) - 主方案
    - dlib:      dlib 68点关键点检测 - 备选方案
    - yunet:     OpenCV FaceDetectorYN (5点) - 回退方案
    """

    def __init__(self, backend='auto', min_detection_confidence=0.5):
        """
        初始化面部检测器，尝试顺序: dlib/face_recognition_models -> mediapipe -> yunet。

        参数:
            backend: 'auto' | 'mediapipe' | 'dlib' | 'yunet'
                'auto' 自动选择可用后端 (优先 dlib 68点)
            min_detection_confidence: float
                最小检测置信度 (0.0 - 1.0)

        异常:
            RuntimeError: 所有后端均不可用
            ImportError: 指定 mediapipe 但未安装
            ValueError:   backend 值不合法
        """
        backend = backend.lower().strip()
        if backend not in ('auto', 'mediapipe', 'dlib', 'yunet'):
            raise ValueError(
                f"不支持的后端: '{backend}'。可选: 'auto', 'mediapipe', 'dlib', 'yunet'"
            )

        self.backend: Optional[str] = None
        self.min_detection_confidence = float(min_detection_confidence)

        # 后端实例
        self.face_landmarker = None     # MediaPipe FaceLandmarker (tasks API)
        self.yunet_detector = None      # cv2.FaceDetectorYN 实例
        self._dlib_backend = None       # DlibBackend 实例
        self._frame_timestamp = 0       # VIDEO模式需要递增时间戳

        # ---- 按优先级初始化 ----
        if backend in ('auto', 'dlib'):
            if _DLIB_AVAILABLE:
                try:
                    self._init_dlib()
                    if self._dlib_backend is not None:
                        self.backend = 'dlib'
                except Exception as e:
                    print(f"[FaceDetector] dlib 初始化失败: {e}", file=sys.stderr)

        if self.backend is None and backend in ('auto', 'mediapipe'):
            if _MP_AVAILABLE and _MP_TASKS_API:
                try:
                    self._init_mediapipe()
                    if self.face_landmarker is not None:
                        self.backend = 'mediapipe'
                except Exception as e:
                    print(f"[FaceDetector] MediaPipe 初始化失败: {e}", file=sys.stderr)

        if self.backend is None and backend in ('auto', 'yunet'):
            self._init_yunet()
            if self.yunet_detector is not None:
                self.backend = 'yunet'

        if self.backend is None:
            if backend == 'mediapipe':
                raise ImportError("MediaPipe 未安装。请运行: pip install mediapipe")
            if backend == 'dlib':
                raise ImportError(
                    "dlib 未安装或68点模型缺失。"
                    "请安装 dlib 并将 shape_predictor_68_face_landmarks.dat 放入 models_data/"
                )
            raise RuntimeError(
                "无法初始化任何面部检测后端。"
                "请安装 mediapipe (pip install mediapipe) 或 dlib，或"
                "将 face_detection_yunet_2023mar.onnx 放入 models_data/ 目录。"
            )

    # ==================================================================
    # 后端初始化
    # ==================================================================

    def _init_mediapipe(self):
        """
        初始化 MediaPipe Face Landmarker (tasks API, 0.10.x+).
        使用 face_landmarker.task 模型文件。
        """
        if not _MP_AVAILABLE:
            raise ImportError("MediaPipe 未安装。请运行: pip install mediapipe")

        if not _MP_TASKS_API:
            raise RuntimeError(
                "当前 MediaPipe 版本不支持 tasks API，请升级到 0.10.0+"
            )

        # 确保模型文件存在 (自动处理中文路径)
        model_path = _ensure_ascii_model_path(_MP_MODEL_PATH)
        if model_path is None:
            raise FileNotFoundError(
                f"MediaPipe 模型文件不存在: {_MP_MODEL_PATH}\n"
                f"请将 face_landmarker.task 放入 models_data/ 目录"
            )

        try:
            base_opts = mp_python.BaseOptions(model_asset_path=model_path)
            opts = vision.FaceLandmarkerOptions(
                base_options=base_opts,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(opts)
        except Exception as e:
            raise RuntimeError(f"MediaPipe FaceLandmarker 初始化失败: {e}") from e

    def _init_yunet(self):
        """
        初始化 OpenCV FaceDetectorYN (基于 ONNX 模型)。

        模型文件: models_data/face_detection_yunet_2023mar.onnx
        """
        if not os.path.exists(YUNET_MODEL_PATH):
            print(
                f"[FaceDetector] Yunet 模型文件不存在: {YUNET_MODEL_PATH}",
                file=sys.stderr,
            )
            self.yunet_detector = None
            return

        try:
            self.yunet_detector = cv2.FaceDetectorYN_create(
                model=YUNET_MODEL_PATH,
                config="",
                input_size=(320, 320),
                score_threshold=self.min_detection_confidence,
                nms_threshold=0.3,
                top_k=5000,
            )
        except Exception as e:
            print(f"[FaceDetector] Yunet 初始化失败: {e}", file=sys.stderr)
            self.yunet_detector = None

    def _init_dlib(self):
        """
        初始化 dlib 后端 (68点关键点检测)。

        使用 DlibBackend 类提供与 MediaPipe 一致的接口。
        """
        try:
            self._dlib_backend = DlibBackend(
                min_detection_confidence=self.min_detection_confidence
            )
        except Exception as e:
            raise RuntimeError(f"dlib 后端初始化失败: {e}") from e

    # ==================================================================
    # 面部检测 (核心接口)
    # ==================================================================

    def detect(self, image: np.ndarray) -> Optional[FaceResult]:
        """
        检测图像中的面部。

        参数:
            image: BGR 格式的 numpy 图像数组 (H, W, 3)

        返回:
            dict: {
                'bbox':          (x, y, w, h) int,
                'landmarks_468': np.ndarray (468, 2) float64  或 None,
                'landmarks_68':  np.ndarray (68, 2)  float64  或 None,
                'landmarks_5':   np.ndarray (5, 2)   float64  或 None,
                'score':         float,
                'image_shape':   (h, w) 原始图像尺寸,
                'backend':       'mediapipe' | 'dlib' | 'yunet',
            }
            None -- 未检测到面部
        """
        if image is None or image.size == 0:
            return None

        if self.backend == 'mediapipe':
            return self._detect_mediapipe(image)
        elif self.backend == 'dlib':
            return self._dlib_backend.detect(image)
        elif self.backend == 'yunet':
            return self._detect_yunet(image)
        return None

    # ------------------------------------------------------------------
    # MediaPipe 检测实现
    # ------------------------------------------------------------------

    def _detect_mediapipe(self, image: np.ndarray) -> Optional[FaceResult]:
        """使用 MediaPipe Face Landmarker (tasks API) 进行面部检测。"""
        h, w = image.shape[:2]

        # BGR -> RGB -> MediaPipe Image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO模式需要递增时间戳
        self._frame_timestamp += 1
        results = self.face_landmarker.detect_for_video(mp_image, self._frame_timestamp)

        if (results.face_landmarks is None
                or len(results.face_landmarks) == 0):
            return None

        face_lm = results.face_landmarks[0]

        # 提取所有关键点: 归一化坐标 -> 像素坐标
        num_landmarks = len(face_lm)
        landmarks_468 = np.zeros((num_landmarks, 2), dtype=np.float64)
        for i, lm in enumerate(face_lm):
            landmarks_468[i, 0] = lm.x * w
            landmarks_468[i, 1] = lm.y * h

        # 由 landmark 极值计算包围盒
        x_min = float(np.min(landmarks_468[:, 0]))
        y_min = float(np.min(landmarks_468[:, 1]))
        x_max = float(np.max(landmarks_468[:, 0]))
        y_max = float(np.max(landmarks_468[:, 1]))

        margin = 0.05
        margin_w = (x_max - x_min) * margin
        margin_h = (y_max - y_min) * margin
        bx = max(0, int(x_min - margin_w))
        by = max(0, int(y_min - margin_h))
        bw = min(w - bx, max(1, int(x_max - x_min + 2 * margin_w)))
        bh = min(h - by, max(1, int(y_max - y_min + 2 * margin_h)))

        return {
            'bbox':          (bx, by, bw, bh),
            'landmarks_468': landmarks_468,
            'landmarks_5':   None,
            'score':         1.0,
            'image_shape':   (h, w),
            'backend':       'mediapipe',
        }

    # ------------------------------------------------------------------
    # Yunet 检测实现
    # ------------------------------------------------------------------

    def _detect_yunet(self, image: np.ndarray) -> Optional[FaceResult]:
        """使用 OpenCV FaceDetectorYN 进行面部检测。"""
        h, w = image.shape[:2]

        self.yunet_detector.setInputSize((w, h))
        _, faces = self.yunet_detector.detect(image)

        if faces is None or len(faces) == 0:
            return None

        face = faces[0].flatten()
        n_elems = len(face)

        # --- 解析不同格式的 Yunet 输出 ---
        # 格式 A (14元素): [x, y, w, h, re_x, re_y, le_x, le_y,
        #                     nt_x, nt_y, rm_x, rm_y, lm_x, lm_y]
        # 格式 B (15元素): [x, y, w, h, conf, re_x, re_y, le_x, le_y,
        #                     nt_x, nt_y, rm_x, rm_y, lm_x, lm_y]
        if n_elems == 15:
            bx, by, bw_box, bh_box = (int(face[0]), int(face[1]),
                                       int(face[2]), int(face[3]))
            score = float(face[4])
            lm_data = face[5:15]
        elif n_elems == 14:
            bx, by, bw_box, bh_box = (int(face[0]), int(face[1]),
                                       int(face[2]), int(face[3]))
            score = 0.9
            lm_data = face[4:14]
        else:
            bx = int(face[0]) if n_elems > 0 else 0
            by = int(face[1]) if n_elems > 1 else 0
            bw_box = int(face[2]) if n_elems > 2 else 0
            bh_box = int(face[3]) if n_elems > 3 else 0
            score = 0.9
            offset = 4 if n_elems > 4 else 0
            lm_data = face[offset:min(n_elems, offset + 10)]
            if len(lm_data) < 10:
                lm_data = np.pad(lm_data, (0, 10 - len(lm_data)),
                                 mode='constant')

        # 裁剪到图像边界内
        bx = max(0, bx)
        by = max(0, by)
        bw_box = min(w - bx, max(1, bw_box))
        bh_box = min(h - by, max(1, bh_box))

        # 5 个关键点: [右眼, 左眼, 鼻尖, 右嘴角, 左嘴角]
        landmarks_5 = np.zeros((5, 2), dtype=np.float64)
        for i in range(5):
            landmarks_5[i, 0] = lm_data[i * 2]
            landmarks_5[i, 1] = lm_data[i * 2 + 1]

        return {
            'bbox':          (bx, by, bw_box, bh_box),
            'landmarks_468': None,
            'landmarks_5':   landmarks_5,
            'score':         score,
            'image_shape':   (h, w),
            'backend':       'yunet',
        }

    # ==================================================================
    # 面部特征区域提取
    # ==================================================================

    def get_eye_regions(
        self, face_result: FaceResult
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        返回左右眼EAR计算所需的6个关键点。

        每个是6个 (x, y) 元组组成的列表:
          点序: p1(内眼角), p2(上睑左), p3(上睑右),
                 p4(外眼角), p5(下睑右), p6(下睑左)

        参数:
            face_result: detect() 返回的面部检测结果字典

        返回:
            (left_eye_pts_6, right_eye_pts_6)
        """
        if face_result is None:
            return ([(0.0, 0.0)] * 6, [(0.0, 0.0)] * 6)

        # dlib 后端: 委托给 DlibBackend
        if self.backend == 'dlib' and self._dlib_backend is not None:
            return self._dlib_backend.get_eye_regions(face_result)

        landmarks_468 = face_result.get('landmarks_468')

        if landmarks_468 is not None:
            # ---- MediaPipe: 精确 6 点提取 ----
            from utils.landmark_map import (
                get_left_eye_ear_points,
                get_right_eye_ear_points,
            )
            left_arr = get_left_eye_ear_points(landmarks_468)
            right_arr = get_right_eye_ear_points(landmarks_468)
            return (
                [(float(p[0]), float(p[1])) for p in left_arr],
                [(float(p[0]), float(p[1])) for p in right_arr],
            )

        # ---- Yunet 回退: 基于 5 点估计眼区 ----
        landmarks_5 = face_result.get('landmarks_5')
        bbox = face_result.get('bbox', (0, 0, 0, 0))

        if landmarks_5 is not None and len(landmarks_5) >= 2:
            # landmarks_5: [右眼, 左眼, ...]
            re_cx, re_cy = float(landmarks_5[0, 0]), float(landmarks_5[0, 1])
            le_cx, le_cy = float(landmarks_5[1, 0]), float(landmarks_5[1, 1])

            # 根据 bbox 宽度估算眼部尺寸
            face_w = bbox[2] if bbox[2] > 0 else 100
            ew = face_w * 0.12
            eh = face_w * 0.06

            def _synth_eye(cx, cy):
                return [
                    (cx - ew/2, cy),
                    (cx - ew/3, cy - eh/2),
                    (cx + ew/3, cy - eh/2),
                    (cx + ew/2, cy),
                    (cx + ew/3, cy + eh/2),
                    (cx - ew/3, cy + eh/2),
                ]

            return (_synth_eye(le_cx, le_cy), _synth_eye(re_cx, re_cy))

        return ([(0.0, 0.0)] * 6, [(0.0, 0.0)] * 6)

    def get_eye_centers(
        self, face_result: FaceResult
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算左右眼的中心坐标。

        参数:
            face_result: detect() 返回的面部检测结果字典

        返回:
            (left_center, right_center): 两个 shape (2,) 的 float64 numpy 数组
        """
        if face_result is None:
            return (np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64))

        # dlib 后端: 从68点计算眼中心
        if self.backend == 'dlib':
            landmarks_68 = face_result.get('landmarks_68')
            if landmarks_68 is not None:
                left_center = np.mean(
                    landmarks_68[_DLIB_LEFT_EYE_EAR_IDX], axis=0
                )
                right_center = np.mean(
                    landmarks_68[_DLIB_RIGHT_EYE_EAR_IDX], axis=0
                )
                return (left_center, right_center)

        landmarks_468 = face_result.get('landmarks_468')

        if landmarks_468 is not None:
            from utils.landmark_map import get_eye_centers as _get_centers
            return _get_centers(landmarks_468)

        # Yunet 回退
        landmarks_5 = face_result.get('landmarks_5')
        if landmarks_5 is not None and len(landmarks_5) >= 2:
            le = np.array([float(landmarks_5[1, 0]), float(landmarks_5[1, 1])],
                          dtype=np.float64)
            re = np.array([float(landmarks_5[0, 0]), float(landmarks_5[0, 1])],
                          dtype=np.float64)
            return (le, re)

        return (np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64))

    def get_mouth_region(self, face_result: FaceResult) -> List[Tuple[float, float]]:
        """
        返回MAR计算的8个嘴部关键点。

        返回 8 个 (x, y) 元组组成的列表:
          [左嘴角, 右嘴角, 上唇顶, 下唇底, 上唇内上, 上唇内下, 下唇内上, 下唇内下]

        参数:
            face_result: detect() 返回的面部检测结果字典

        返回:
            8 个 (x, y) 元组的列表
        """
        if face_result is None:
            return [(0.0, 0.0)] * 8

        # dlib 后端: 委托给 DlibBackend
        if self.backend == 'dlib' and self._dlib_backend is not None:
            return self._dlib_backend.get_mouth_region(face_result)

        landmarks_468 = face_result.get('landmarks_468')

        if landmarks_468 is not None:
            from utils.landmark_map import get_mouth_mar_points
            pts = get_mouth_mar_points(landmarks_468)
            return [(float(p[0]), float(p[1])) for p in pts]

        # Yunet 回退: 基于嘴角估计嘴唇轮廓
        landmarks_5 = face_result.get('landmarks_5')
        if landmarks_5 is not None and len(landmarks_5) >= 5:
            # 右嘴角 idx=3, 左嘴角 idx=4
            rmx, rmy = float(landmarks_5[3, 0]), float(landmarks_5[3, 1])
            lmx, lmy = float(landmarks_5[4, 0]), float(landmarks_5[4, 1])

            mw = abs(lmx - rmx) or 1.0
            mh = mw * 0.35
            mid_x = (rmx + lmx) / 2.0
            mid_y = (rmy + lmy) / 2.0

            return [
                (lmx, lmy),                          # 0: 左嘴角
                (rmx, rmy),                          # 1: 右嘴角
                (mid_x, mid_y - mh),                 # 2: 上唇外顶
                (mid_x, mid_y + mh),                 # 3: 下唇外底
                (mid_x, mid_y - mh * 0.5),           # 4: 上唇内上
                (mid_x, mid_y - mh * 0.25),          # 5: 上唇内下
                (mid_x, mid_y + mh * 0.5),           # 6: 下唇内上
                (mid_x, mid_y + mh * 0.25),          # 7: 下唇内下
            ]

        return [(0.0, 0.0)] * 8

    def get_head_pose_points(self, face_result: FaceResult) -> np.ndarray:
        """
        返回头部姿态估计所需的6个2D点。

        点序: [鼻尖, 下巴, 左眼角, 右眼角, 左嘴角, 右嘴角]

        参数:
            face_result: detect() 返回的面部检测结果字典

        返回:
            shape (6, 2) 的 numpy float64 数组
        """
        if face_result is None:
            return np.zeros((6, 2), dtype=np.float64)

        # dlib 后端: 委托给 DlibBackend
        if self.backend == 'dlib' and self._dlib_backend is not None:
            return self._dlib_backend.get_head_pose_points(face_result)

        landmarks_468 = face_result.get('landmarks_468')

        if landmarks_468 is not None:
            from utils.landmark_map import get_head_pose_points as _get_hp
            return _get_hp(landmarks_468)

        # Yunet 回退: 5 点 + 合成下巴
        landmarks_5 = face_result.get('landmarks_5')
        bbox = face_result.get('bbox', (0, 0, 0, 0))

        if landmarks_5 is not None and len(landmarks_5) >= 5:
            pts = np.array([
                [float(landmarks_5[2, 0]), float(landmarks_5[2, 1])],   # 鼻尖
                [bbox[0] + bbox[2]/2.0, bbox[1] + bbox[3]],             # 下巴(估计)
                [float(landmarks_5[1, 0]), float(landmarks_5[1, 1])],   # 左眼角
                [float(landmarks_5[0, 0]), float(landmarks_5[0, 1])],   # 右眼角
                [float(landmarks_5[4, 0]), float(landmarks_5[4, 1])],   # 左嘴角
                [float(landmarks_5[3, 0]), float(landmarks_5[3, 1])],   # 右嘴角
            ], dtype=np.float64)
            return pts

        return np.zeros((6, 2), dtype=np.float64)

    def get_nose_tip(self, face_result: FaceResult) -> Tuple[float, float]:
        """
        返回鼻尖坐标。

        参数:
            face_result: detect() 返回的面部检测结果字典

        返回:
            (x, y) 元组
        """
        if face_result is None:
            return (0.0, 0.0)

        # dlib 后端: 委托给 DlibBackend
        if self.backend == 'dlib' and self._dlib_backend is not None:
            return self._dlib_backend.get_nose_tip(face_result)

        landmarks_468 = face_result.get('landmarks_468')

        if landmarks_468 is not None:
            from utils.landmark_map import NOSE_TIP_IDX
            pt = landmarks_468[NOSE_TIP_IDX]
            return (float(pt[0]), float(pt[1]))

        landmarks_5 = face_result.get('landmarks_5')
        if landmarks_5 is not None and len(landmarks_5) >= 3:
            return (float(landmarks_5[2, 0]), float(landmarks_5[2, 1]))

        return (0.0, 0.0)

    def get_nose_bridge(self, face_result: FaceResult) -> Tuple[float, float]:
        """
        返回鼻梁坐标（用于视线方向估算）。

        参数:
            face_result: detect() 返回的面部检测结果字典

        返回:
            (x, y) 元组
        """
        if face_result is None:
            return (0.0, 0.0)

        # dlib 后端: 委托给 DlibBackend
        if self.backend == 'dlib' and self._dlib_backend is not None:
            return self._dlib_backend.get_nose_bridge(face_result)

        landmarks_468 = face_result.get('landmarks_468')

        if landmarks_468 is not None:
            from utils.landmark_map import NOSE_BRIDGE_IDX
            pt = landmarks_468[NOSE_BRIDGE_IDX]
            return (float(pt[0]), float(pt[1]))

        # Yunet 回退: 鼻梁 ≈ 鼻尖上方约 15px
        landmarks_5 = face_result.get('landmarks_5')
        if landmarks_5 is not None and len(landmarks_5) >= 3:
            return (float(landmarks_5[2, 0]), float(landmarks_5[2, 1]) - 15.0)

        return (0.0, 0.0)

    def get_forehead_roi(self, face_result: FaceResult,
                          image: np.ndarray) -> np.ndarray:
        """
        提取前额ROI区域图像，用于rPPG远程光电容积描记。

        基于前额10个关键点的包围矩形裁剪前额区域。

        参数:
            face_result: detect() 返回的面部检测结果字典
            image:       原始 BGR 格式图像

        返回:
            前额 ROI 区域的 BGR numpy 数组。
            无法提取时返回形状为 (0,) 的空数组。
        """
        if face_result is None or image is None or image.size == 0:
            return np.array([])

        h, w = image.shape[:2]

        # dlib 后端: 委托给 DlibBackend
        if self.backend == 'dlib' and self._dlib_backend is not None:
            return self._dlib_backend.get_forehead_roi(face_result, image)

        landmarks_468 = face_result.get('landmarks_468')

        if landmarks_468 is not None:
            from utils.landmark_map import get_forehead_roi_points

            fh_pts = get_forehead_roi_points(landmarks_468)
            x_min = int(np.min(fh_pts[:, 0]))
            y_min = int(np.min(fh_pts[:, 1]))
            x_max = int(np.max(fh_pts[:, 0]))
            y_max = int(np.max(fh_pts[:, 1]))

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            if x_max <= x_min or y_max <= y_min:
                return np.array([])

            return image[y_min:y_max, x_min:x_max].copy()

        # Yunet 回退: bbox 顶部中央区域作为前额估计
        bbox = face_result.get('bbox')
        if bbox is not None:
            bx, by, bw_box, bh_box = bbox
            fh = max(1, int(bh_box * 0.18))
            fw = max(1, int(bw_box * 0.45))
            fx = bx + int(bw_box * 0.275)
            fy = by + max(0, int(bh_box * 0.03))

            fx = max(0, fx)
            fy = max(0, fy)
            fw = min(w - fx, fw)
            fh = min(h - fy, fh)

            if fw <= 0 or fh <= 0:
                return np.array([])

            return image[fy:fy + fh, fx:fx + fw].copy()

        return np.array([])

    # ==================================================================
    # 可视化
    # ==================================================================

    def draw_landmarks(self, image: np.ndarray, face_result: FaceResult,
                       draw_eyes: bool = True,
                       draw_mouth: bool = True) -> np.ndarray:
        """
        在图像上绘制面部特征点和包围盒。

        MediaPipe 后端: 绘制 468 点网格 + 眼部/嘴部高亮
        Yunet 后端:     绘制 5 个关键点 + 标签

        参数:
            image:       BGR 格式 numpy 图像数组 (原地修改)
            face_result: detect() 返回的面部检测结果字典
            draw_eyes:   是否绘制眼部关键点
            draw_mouth:  是否绘制嘴部关键点

        返回:
            标注后的图像 (与输入 image 是同一对象)
        """
        if image is None or face_result is None:
            return image

        # dlib 后端: 委托给 DlibBackend
        if self.backend == 'dlib' and self._dlib_backend is not None:
            return self._dlib_backend.draw_landmarks(
                image, face_result, draw_eyes, draw_mouth
            )

        bbox = face_result.get('bbox')
        landmarks_468 = face_result.get('landmarks_468')

        # ---- 包围盒 ----
        if bbox is not None:
            x, y, bw_box, bh_box = bbox
            cv2.rectangle(image, (x, y), (x + bw_box, y + bh_box),
                          (0, 255, 0), 2)

        # ---- 关键点 ----
        if landmarks_468 is not None:
            self._draw_468_landmarks(image, landmarks_468, draw_eyes, draw_mouth)
        else:
            landmarks_5 = face_result.get('landmarks_5')
            if landmarks_5 is not None:
                self._draw_yunet_landmarks(image, landmarks_5)

        return image

    def _draw_468_landmarks(self, image, landmarks_468, draw_eyes, draw_mouth):
        """绘制 MediaPipe 468 点网格。"""
        h, w = image.shape[:2]

        # 全部 468 点
        for i in range(len(landmarks_468)):
            px, py = int(landmarks_468[i, 0]), int(landmarks_468[i, 1])
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(image, (px, py), 1, (0, 255, 0), -1)

        if draw_eyes:
            from utils.landmark_map import (
                LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX,
                get_left_eye_ear_points, get_right_eye_ear_points,
            )
            for idx in LEFT_EYE_EAR_IDX + RIGHT_EYE_EAR_IDX:
                if 0 <= idx < len(landmarks_468):
                    px, py = int(landmarks_468[idx, 0]), int(landmarks_468[idx, 1])
                    if 0 <= px < w and 0 <= py < h:
                        cv2.circle(image, (px, py), 2, (255, 100, 0), -1)

            left_eye = get_left_eye_ear_points(landmarks_468)
            right_eye = get_right_eye_ear_points(landmarks_468)
            self._draw_polyline(image, left_eye, (255, 100, 0), closed=True)
            self._draw_polyline(image, right_eye, (255, 100, 0), closed=True)

        if draw_mouth:
            from utils.landmark_map import MOUTH_MAR_IDX, get_mouth_mar_points
            for idx in MOUTH_MAR_IDX:
                if 0 <= idx < len(landmarks_468):
                    px, py = int(landmarks_468[idx, 0]), int(landmarks_468[idx, 1])
                    if 0 <= px < w and 0 <= py < h:
                        cv2.circle(image, (px, py), 2, (0, 100, 255), -1)

            mouth_pts = get_mouth_mar_points(landmarks_468)
            self._draw_polyline(image, mouth_pts[:4], (0, 100, 255), closed=True)
            self._draw_polyline(image, mouth_pts[4:], (0, 100, 255), closed=True)

    def _draw_yunet_landmarks(self, image, landmarks_5):
        """绘制 Yunet 5 个关键点。"""
        h, w = image.shape[:2]
        colors = [
            (255, 100, 0),   # 右眼
            (255, 100, 0),   # 左眼
            (0, 255, 255),   # 鼻尖
            (0, 100, 255),   # 右嘴角
            (0, 100, 255),   # 左嘴角
        ]
        for i in range(len(landmarks_5)):
            px, py = int(landmarks_5[i, 0]), int(landmarks_5[i, 1])
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(image, (px, py), 3, colors[i], -1)

    @staticmethod
    def _draw_polyline(image, points, color, closed=False):
        """绘制一条折线（用于眼部/嘴部轮廓）。"""
        h, w = image.shape[:2]
        pts = [(int(p[0]), int(p[1])) for p in points
               if 0 <= int(p[0]) < w and 0 <= int(p[1]) < h]

        if len(pts) < 2:
            return

        for j in range(len(pts) - 1):
            cv2.line(image, pts[j], pts[j + 1], color, 1, cv2.LINE_AA)

        if closed and len(pts) >= 3:
            cv2.line(image, pts[-1], pts[0], color, 1, cv2.LINE_AA)

    # ==================================================================
    # 资源管理
    # ==================================================================

    def close(self):
        """释放后端资源。多次调用安全。"""
        if self.face_landmarker is not None:
            try:
                self.face_landmarker.close()
            except Exception:
                pass
            self.face_landmarker = None

        if self._dlib_backend is not None:
            try:
                self._dlib_backend.close()
            except Exception:
                pass
            self._dlib_backend = None

        self.yunet_detector = None
        self.mp_face_mesh = None

    def release(self):
        """释放资源 (close 的别名，保持向后兼容)。"""
        self.close()

    def __del__(self):
        """析构时自动释放资源。"""
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return (f"FaceDetector(backend='{self.backend}', "
                f"confidence={self.min_detection_confidence})")
