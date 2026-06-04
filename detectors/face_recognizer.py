"""
面部识别模块
============
使用 dlib 的 ResNet 模型生成 128D 面部嵌入向量，实现驾驶员身份注册与识别。

功能:
  - 注册驾驶员: 从面部图像提取 128D 嵌入，绑定姓名后持久化
  - 识别驾驶员: 计算嵌入向量的余弦相似度，返回最佳匹配
  - 驾驶员管理: 列出 / 删除已注册驾驶员

依赖:
  - dlib (dlib_face_recognition_resnet_model_v1.dat)
  - 68点关键点模型 (shape_predictor_68_face_landmarks.dat)

用法:
    recognizer = FaceRecognizer()
    recognizer.register_driver("张三", face_image_bgr)
    name, confidence = recognizer.identify_driver(face_image_bgr)
"""

from __future__ import annotations

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from config import BASE_DIR

# ============================================================================
# 可选依赖检测
# ============================================================================
try:
    import dlib
    _DLIB_AVAILABLE = True
except ImportError:
    _DLIB_AVAILABLE = False
    dlib = None


# ============================================================================
# 模型路径搜索
# ============================================================================

def _find_recognition_model() -> Optional[str]:
    """搜索 dlib 面部识别 ResNet 模型文件。"""
    candidates = []

    # 1. face_recognition_models 包 (pip安装)
    try:
        import face_recognition_models
        fr_dir = os.path.join(
            os.path.dirname(face_recognition_models.__file__), "models"
        )
        candidates.append(
            os.path.join(fr_dir, "dlib_face_recognition_resnet_model_v1.dat")
        )
    except ImportError:
        pass

    # 2. 项目 models_data 目录
    candidates.append(
        str(BASE_DIR / "models_data" / "dlib_face_recognition_resnet_model_v1.dat")
    )

    # 3. site-packages 下的 face_recognition_models
    try:
        import site
        for sp in site.getsitepackages():
            candidates.append(
                os.path.join(
                    sp, "face_recognition_models", "models",
                    "dlib_face_recognition_resnet_model_v1.dat"
                )
            )
    except Exception:
        pass

    # 4. dlib 安装目录
    if dlib is not None:
        try:
            dlib_dir = os.path.dirname(os.path.abspath(dlib.__file__))
            candidates.append(
                os.path.join(dlib_dir, "dlib_face_recognition_resnet_model_v1.dat")
            )
        except Exception:
            pass

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _find_68_landmark_model() -> Optional[str]:
    """搜索 dlib 68点关键点模型文件。"""
    candidates = []

    # face_recognition_models 包
    try:
        import face_recognition_models
        fr_dir = os.path.join(
            os.path.dirname(face_recognition_models.__file__), "models"
        )
        candidates.append(
            os.path.join(fr_dir, "shape_predictor_68_face_landmarks.dat")
        )
    except ImportError:
        pass

    # 项目 models_data 目录
    candidates.append(
        str(BASE_DIR / "models_data" / "shape_predictor_68_face_landmarks.dat")
    )

    # site-packages
    try:
        import site
        for sp in site.getsitepackages():
            candidates.append(
                os.path.join(
                    sp, "face_recognition_models", "models",
                    "shape_predictor_68_face_landmarks.dat"
                )
            )
    except Exception:
        pass

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# ============================================================================
# 驾驶员数据库文件路径
# ============================================================================
_DRIVERS_DB_DIR = str(BASE_DIR / "models_data")
_DRIVERS_DB_FILE = os.path.join(_DRIVERS_DB_DIR, "registered_drivers.json")

# 识别阈值: 余弦相似度低于此值视为未知人员
_DEFAULT_RECOGNITION_THRESHOLD = 0.6


class FaceRecognizer:
    """
    基于 dlib ResNet 的面部识别器。

    使用 dlib_face_recognition_resnet_model_v1.dat 模型将人脸图像
    映射为 128D 嵌入向量，通过余弦相似度进行身份匹配。

    已注册驾驶员信息持久化到 models_data/registered_drivers.json。
    """

    def __init__(self, recognition_threshold: float = _DEFAULT_RECOGNITION_THRESHOLD):
        """
        初始化面部识别器。

        参数:
            recognition_threshold: float
                余弦相似度阈值 (0.0 - 1.0)，低于此值判定为未知人员。
                推荐值 0.6，要求严格可调高。

        异常:
            ImportError: dlib 未安装
            FileNotFoundError: 找不到必需的模型文件
        """
        if not _DLIB_AVAILABLE:
            raise ImportError(
                "dlib 未安装。请安装: pip install dlib-bin 或 pip install dlib"
            )

        self.recognition_threshold = float(recognition_threshold)

        # ---- 加载 68 点 landmark 模型 ----
        landmark_path = _find_68_landmark_model()
        if landmark_path is None:
            raise FileNotFoundError(
                "找不到 dlib 68点关键点模型 (shape_predictor_68_face_landmarks.dat)。"
                "请将其放入 models_data/ 目录或安装 face_recognition_models 包。"
            )
        self.shape_predictor = dlib.shape_predictor(str(landmark_path))

        # ---- 加载面部识别 ResNet 模型 ----
        recog_path = _find_recognition_model()
        if recog_path is None:
            raise FileNotFoundError(
                "找不到 dlib 面部识别模型 "
                "(dlib_face_recognition_resnet_model_v1.dat)。"
                "请将其放入 models_data/ 目录或安装 face_recognition_models 包。"
            )
        self.face_rec_model = dlib.face_recognition_model_v1(str(recog_path))

        # ---- 人脸检测器 ----
        self.face_detector = dlib.get_frontal_face_detector()

        # ---- 加载已注册驾驶员 ----
        self._drivers: Dict[str, List[float]] = {}  # {name: embedding_list}
        self._load_drivers()

        # 模型路径记录
        self._landmark_model_path = landmark_path
        self._recog_model_path = recog_path

    # ==================================================================
    # 嵌入计算
    # ==================================================================

    def compute_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        从人脸图像计算 128D 嵌入向量。

        参数:
            face_image: BGR 格式的 numpy 图像数组 (H, W, 3)，
                        应包含一张清晰的人脸。

        返回:
            shape (128,) 的 numpy float64 数组，或 None（未检测到人脸）。
        """
        if face_image is None or face_image.size == 0:
            return None

        # 转灰度用于检测和关键点
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image

        # 检测人脸
        faces = self.face_detector(gray, 0)
        if len(faces) == 0:
            return None

        # 取最大人脸
        face_rect = max(faces, key=lambda r: r.width() * r.height())

        # 提取 68 点关键点
        shape = self.shape_predictor(gray, face_rect)

        # 计算 128D 嵌入 (需要 RGB 图像)
        if len(face_image.shape) == 3:
            rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        embedding = self.face_rec_model.compute_face_descriptor(rgb, shape)

        return np.array(embedding, dtype=np.float64)

    # ==================================================================
    # 驾驶员注册与管理
    # ==================================================================

    def register_driver(self, name: str, face_image: np.ndarray) -> bool:
        """
        注册一名驾驶员。

        从给定人脸图像提取 128D 嵌入并绑定到指定姓名。

        参数:
            name:       驾驶员姓名（唯一标识）
            face_image: BGR 格式的人脸图像

        返回:
            True 表示注册成功，False 表示未检测到人脸。
        """
        name = str(name).strip()
        if not name:
            raise ValueError("驾驶员姓名不能为空")

        embedding = self.compute_embedding(face_image)
        if embedding is None:
            return False

        # 存储嵌入向量 (转为 list 以便 JSON 序列化)
        self._drivers[name] = embedding.tolist()

        # 持久化
        self._save_drivers()

        print(f"[FaceRecognizer] 已注册驾驶员: {name}")
        return True

    def identify_driver(
        self, face_image: np.ndarray
    ) -> Optional[Tuple[str, float]]:
        """
        识别驾驶员身份。

        计算输入人脸的 128D 嵌入，与所有已注册驾驶员进行余弦相似度比对。

        参数:
            face_image: BGR 格式的人脸图像

        返回:
            (name, confidence) 元组，或 None（未检测到人脸或无匹配）。
            confidence 为余弦相似度，范围 [-1, 1]。
        """
        if not self._drivers:
            return None

        embedding = self.compute_embedding(face_image)
        if embedding is None:
            return None

        best_name = None
        best_similarity = -1.0

        for name, stored_embedding_list in self._drivers.items():
            stored_embedding = np.array(stored_embedding_list, dtype=np.float64)

            # 余弦相似度
            dot = np.dot(embedding, stored_embedding)
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(stored_embedding)
            if norm_a < 1e-10 or norm_b < 1e-10:
                similarity = 0.0
            else:
                similarity = float(dot / (norm_a * norm_b))

            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name

        if best_name is not None and best_similarity >= self.recognition_threshold:
            return (best_name, best_similarity)

        return None

    def list_drivers(self) -> List[str]:
        """
        列出所有已注册驾驶员姓名。

        返回:
            姓名字符串列表，按字母序排列。
        """
        return sorted(self._drivers.keys())

    def delete_driver(self, name: str) -> bool:
        """
        删除一名已注册驾驶员。

        参数:
            name: 驾驶员姓名

        返回:
            True 表示删除成功，False 表示该驾驶员不存在。
        """
        name = str(name).strip()
        if name in self._drivers:
            del self._drivers[name]
            self._save_drivers()
            print(f"[FaceRecognizer] 已删除驾驶员: {name}")
            return True
        return False

    def get_driver_count(self) -> int:
        """返回已注册驾驶员数量。"""
        return len(self._drivers)

    # ==================================================================
    # 持久化
    # ==================================================================

    def _save_drivers(self):
        """将已注册驾驶员数据保存到 JSON 文件。"""
        os.makedirs(_DRIVERS_DB_DIR, exist_ok=True)
        try:
            with open(_DRIVERS_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._drivers, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[FaceRecognizer] 保存驾驶员数据失败: {e}", file=sys.stderr)

    def _load_drivers(self):
        """从 JSON 文件加载已注册驾驶员数据。"""
        if not os.path.exists(_DRIVERS_DB_FILE):
            self._drivers = {}
            return

        try:
            with open(_DRIVERS_DB_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._drivers = data
            else:
                self._drivers = {}
        except Exception as e:
            print(f"[FaceRecognizer] 加载驾驶员数据失败: {e}", file=sys.stderr)
            self._drivers = {}

    # ==================================================================
    # 资源管理
    # ==================================================================

    def close(self):
        """释放模型资源。"""
        self.face_detector = None
        self.shape_predictor = None
        self.face_rec_model = None

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
        return (f"FaceRecognizer(drivers={len(self._drivers)}, "
                f"threshold={self.recognition_threshold})")
