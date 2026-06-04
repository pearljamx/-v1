"""
实时摄像头帧处理器
==================
FramePipeline的轻量包装，专为摄像头实时检测优化：
- 复用单个FramePipeline实例（避免重复加载模型）
- 单帧处理 + Base64 JPEG编码
- 无磁盘I/O、无进度回调开销
- 返回简洁JSON就绪字典
"""

import time
import cv2
import numpy as np
from processors.frame_pipeline import FramePipeline


class RealtimeProcessor:
    """
    实时摄像头帧处理器

    封装FramePipeline，为摄像头实时帧提供高效处理接口。
    复用所有检测器实例，避免重复初始化开销。

    用法:
        processor = RealtimeProcessor(enable_modules={'fatigue': True, ...})
        result = processor.process_frame(bgr_frame)
        # result包含: overlay, face_detected, fatigue, head_pose,
        #             gaze, distraction, alerts, summary
    """

    def __init__(self, enable_modules=None):
        """
        初始化实时处理器

        参数:
            enable_modules: dict, 控制启用的检测模块
                默认: {'fatigue': True, 'pose': True, 'gaze': True,
                       'distraction': False, 'physio': False}
                注意: distraction和physio默认关闭以提升实时性能
        """
        if enable_modules is None:
            enable_modules = {
                'fatigue': True,
                'pose': True,
                'gaze': True,
                'distraction': False,  # YOLO推理较慢, 默认关闭
                'physio': False,       # rPPG需要长视频, 默认关闭
            }

        self.enable_modules = enable_modules
        self.pipeline = FramePipeline(enable_modules=enable_modules)
        self.session_start_time = time.time()
        self.frame_count = 0

        # 自动重置参数：每处理 AUTO_RESET_FRAMES 帧后自动重建pipeline
        # 防止TFLite XNNPACK长时间运行崩溃和内存累积
        self.AUTO_RESET_FRAMES = 600  # ~2分钟 @ 5FPS
        self._total_frames = 0

    def process_frame(self, frame_bgr):
        """
        处理单帧摄像头图像

        参数:
            frame_bgr: BGR格式numpy数组 (H, W, 3)

        返回:
            dict: {
                'overlay': dict,               # 前端 canvas 纯图形叠加数据
                'face_detected': bool,
                'fatigue': {...},
                'head_pose': {...},
                'gaze': {...},
                'alerts': [...],              # 告警列表(dict格式)
                'summary': {...},
            }
        """
        self.frame_count += 1
        self._total_frames += 1

        # 定期自动重建pipeline，防止TFLite XNNPACK崩溃和内存累积
        if self._total_frames > 0 and self._total_frames % self.AUTO_RESET_FRAMES == 0:
            old_pipeline = self.pipeline
            self.pipeline = FramePipeline(enable_modules=self.enable_modules)
            # 保留告警管理器的关键状态
            self.pipeline.alert_manager = old_pipeline.alert_manager
            del old_pipeline

        # 调整帧大小以加速处理
        h, w = frame_bgr.shape[:2]
        if w > 640:
            scale = 640.0 / w
            frame_bgr = cv2.resize(frame_bgr, (640, int(h * scale)))

        # 处理帧
        timestamp = time.time() - self.session_start_time
        result = self.pipeline.process_frame(frame_bgr, timestamp)

        # 生成前端 overlay：复用流水线本帧已完成的人脸检测结果，避免重复推理。
        face_result = self.pipeline.last_face_result
        overlay = self.pipeline.build_overlay(
            face_result if isinstance(face_result, dict) else None,
            result,
            frame_bgr.shape,
        )

        # 构建返回数据 (清理非可序列化对象)
        output = {
            'face_detected': result.get('face_detected', False),
            'fatigue': {
                'ear': result.get('fatigue', {}).get('ear', 0),
                'mar': result.get('fatigue', {}).get('mar', 0),
                'blink_rate': result.get('fatigue', {}).get('blink_rate', 0),
            },
            'head_pose': result.get('head_pose', {}),
            'gaze': {
                'gaze_angle': result.get('gaze', {}).get('gaze_angle', 0),
                'is_deviated': result.get('gaze', {}).get('is_deviated', False),
            },
            'distraction': result.get('distraction', {}),
            'physio': result.get('physio', {}),
            'alerts': result.get('alerts', []),
            'summary': result.get('summary', {}),
            'overlay': overlay,
        }

        return output

    def get_timeseries_data(self):
        """获取时序数据(用于前端迷你图表)"""
        return self.pipeline.get_timeseries_data()

    def reset(self):
        """重置处理器状态"""
        self.pipeline.reset()
        self.session_start_time = time.time()
        self.frame_count = 0
