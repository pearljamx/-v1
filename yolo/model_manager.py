"""
YOLO模型管理器
==============
统一管理YOLO模型的加载、缓存和推理接口。
支持手持物品检测模型和人体姿态估计模型。
"""

import logging
import os
import numpy as np

logger = logging.getLogger(__name__)
_GLOBAL_MANAGER = None


class YOLOModelManager:
    """
    YOLO模型管理器
    提供模型加载、推理和资源释放的统一接口
    """

    def __init__(self):
        self._handheld_model = None
        self._pose_model = None

    @property
    def handheld_model(self):
        """延迟加载手持物品检测模型"""
        if self._handheld_model is None:
            self._load_handheld_model()
        return self._handheld_model

    @property
    def pose_model(self):
        """延迟加载人体姿态估计模型"""
        if self._pose_model is None:
            self._load_pose_model()
        return self._pose_model

    def _load_handheld_model(self):
        """加载手持物品检测YOLO模型"""
        from config import YOLO_HANDHELD_MODEL

        try:
            from ultralytics import YOLO
            if os.path.exists(YOLO_HANDHELD_MODEL):
                self._handheld_model = YOLO(YOLO_HANDHELD_MODEL)
                logger.info("[YOLO] 手持物品检测模型已加载: %s", YOLO_HANDHELD_MODEL)
            else:
                logger.warning("[YOLO] 手持物品检测模型不存在 (%s)，跳过加载", YOLO_HANDHELD_MODEL)
                self._handheld_model = False  # 标记为不可用
        except Exception as e:
            logger.error("[YOLO] 加载手持物品模型失败: %s", e)
            self._handheld_model = False

    def _load_pose_model(self):
        """加载人体姿态估计YOLO模型"""
        from config import YOLO_POSE_MODEL

        try:
            from ultralytics import YOLO
            if os.path.exists(YOLO_POSE_MODEL):
                self._pose_model = YOLO(YOLO_POSE_MODEL)
                logger.info("[YOLO] 人体姿态估计模型已加载: %s", YOLO_POSE_MODEL)
            else:
                logger.warning("[YOLO] 姿态估计模型不存在 (%s)，跳过加载", YOLO_POSE_MODEL)
                self._pose_model = False
        except Exception as e:
            logger.error("[YOLO] 加载姿态估计模型失败: %s", e)
            self._pose_model = False

    def detect_objects(self, image, conf_threshold=0.25):
        """
        使用手持物品检测模型进行推理

        参数:
            image: BGR numpy数组
            conf_threshold: 置信度阈值

        返回:
            [{'class': str, 'class_id': int, 'confidence': float, 'bbox': (x1,y1,x2,y2)}, ...]
        """
        if self.handheld_model is None or self.handheld_model is False:
            return []

        try:
            results = self.handheld_model(image, conf=conf_threshold, verbose=False)

            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        class_names = {0: "normal", 1: "phone", 2: "smoking",
                                      3: "drinking", 4: "hands_off"}
                        class_name = class_names.get(class_id, f"class_{class_id}")

                        detections.append({
                            'class': class_name,
                            'class_id': class_id,
                            'confidence': confidence,
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        })

            return detections

        except Exception as e:
            logger.error("[YOLO] 物体检测推理失败: %s", e)
            return []

    def detect_pose(self, image, conf_threshold=0.5):
        """
        使用姿态估计模型检测人体关键点

        参数:
            image: BGR numpy数组
            conf_threshold: 置信度阈值

        返回:
            {
                'keypoints': numpy array (17, 3) - (x, y, confidence),
                'bbox': (x1, y1, x2, y2) 或 None
            }
            如果未检测到人，返回 None
        """
        if self.pose_model is None or self.pose_model is False:
            return None

        try:
            results = self.pose_model(image, conf=conf_threshold, verbose=False)

            if results and len(results) > 0:
                result = results[0]
                if result.keypoints is not None and len(result.keypoints) > 0:
                    kpts = result.keypoints[0]
                    keypoints_data = kpts.data.cpu().numpy() if hasattr(kpts.data, 'cpu') else np.array(kpts.data)

                    # 获取边界框
                    bbox = None
                    if result.boxes is not None and len(result.boxes) > 0:
                        x1, y1, x2, y2 = result.boxes[0].xyxy[0].tolist()
                        bbox = (int(x1), int(y1), int(x2), int(y2))

                    return {
                        'keypoints': keypoints_data[0] if keypoints_data.ndim == 3 else keypoints_data,
                        'bbox': bbox,
                    }

            return None

        except Exception as e:
            logger.error("[YOLO] 姿态检测推理失败: %s", e)
            return None

    def is_handheld_available(self):
        """检查手持物品检测模型是否可用"""
        return self._handheld_model is not None and self._handheld_model is not False

    def is_pose_available(self):
        """检查姿态估计模型是否可用"""
        return self._pose_model is not None and self._pose_model is not False

    def release(self):
        """释放模型资源"""
        self._handheld_model = None
        self._pose_model = None


def get_yolo_model_manager():
    """获取进程级 YOLOModelManager，供 Web 任务和实时 session 复用。"""
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = YOLOModelManager()
    return _GLOBAL_MANAGER
