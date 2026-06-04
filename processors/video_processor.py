"""
视频处理器
==========
负责视频文件的逐帧读取、检测流水线调用、进度上报和结果汇总。
使用后台线程异步处理，通过回调函数上报进度。
"""

import os
import time
import cv2
import numpy as np
from config import FPS_TARGET, BASE_DIR
from utils.helpers import ensure_dir, safe_json_dump


class VideoProcessor:
    """
    视频文件处理器
    在后台线程中逐帧处理视频，通过回调函数实时上报进度
    """

    def __init__(self, pipeline):
        """
        参数:
            pipeline: FramePipeline 实例
        """
        self.pipeline = pipeline
        self._stop_flag = False

    def process_video(self, video_path, task_id, progress_callback=None, output_video=False):
        """
        处理视频文件

        参数:
            video_path: 视频文件路径
            task_id: 任务ID
            progress_callback: 进度回调函数 callback(progress_dict)
            output_video: 是否输出标注视频

        返回:
            结果字典
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': '无法打开视频文件'}

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / original_fps if original_fps > 0 else 0

        # 计算跳帧步长（实现目标FPS）
        if original_fps > FPS_TARGET:
            frame_step = int(original_fps / FPS_TARGET)
        else:
            frame_step = 1

        self.pipeline.reset()
        self._stop_flag = False

        processed_frames = 0
        total_to_process = max(1, (total_frames + frame_step - 1) // frame_step)
        annotated_frames = []  # 存储部分标注帧用于预览
        results = []

        start_time = time.time()
        frame_idx = 0
        video_writer = None
        output_path = None
        output_size = None

        while frame_idx < total_frames and not self._stop_flag:
            ret, frame = cap.read()

            if not ret:
                break

            should_process = frame_idx % frame_step == 0
            frame_idx += 1
            if not should_process:
                continue

            # 调整帧大小（限制宽度640px以加速处理）
            if video_width > 640:
                scale = 640.0 / video_width
                new_w = 640
                new_h = int(video_height * scale)
                frame = cv2.resize(frame, (new_w, new_h))

            if output_video and video_writer is None:
                output_dir = os.path.join(BASE_DIR, 'outputs')
                ensure_dir(output_dir)
                output_path = os.path.join(output_dir, f'{task_id}_annotated.mp4')
                output_size = (frame.shape[1], frame.shape[0])
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    output_path, fourcc, FPS_TARGET, output_size
                )

            # 处理帧
            timestamp = processed_frames / FPS_TARGET
            result = self.pipeline.process_frame(frame, timestamp)
            results.append(result)

            # 生成标注帧
            face_result = self.pipeline.last_face_result if result.get('face_detected') else None
            annotated = self.pipeline.get_annotated_frame(
                frame,
                face_result if isinstance(face_result, dict) else None,
                result
            )

            # 写入输出视频
            if video_writer:
                video_writer.write(annotated)

            # 每处理30帧保存一个标注帧用于预览
            if processed_frames % 30 == 0 and len(annotated_frames) < 20:
                import base64
                _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                annotated_frames.append(base64.b64encode(buffer).decode('utf-8'))

            processed_frames += 1

            # 进度回调
            if progress_callback and processed_frames % 5 == 0:
                progress = min(95, int(processed_frames / total_to_process * 100))
                progress_callback({
                    'task_id': task_id,
                    'status': 'processing',
                    'progress': progress,
                    'current_frame': processed_frames,
                    'total_frames': total_to_process,
                    'fps_processing': processed_frames / (time.time() - start_time) if start_time else 0,
                    'current_alerts': self.pipeline.alert_manager.get_active_alerts(),
                    'preview_frame': annotated_frames[-1] if annotated_frames else None,
                })

        cap.release()
        if video_writer:
            video_writer.release()

        # 收集最终结果
        final_results = self.pipeline.get_final_results()
        final_results['video_info'] = {
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'original_fps': original_fps,
            'duration': duration,
            'width': output_size[0] if output_size else video_width,
            'height': output_size[1] if output_size else video_height,
        }
        final_results['annotated_frames'] = annotated_frames

        if output_path:
            final_results['annotated_video_path'] = output_path

        # 保存结果到文件
        result_dir = os.path.join(BASE_DIR, 'outputs')
        ensure_dir(result_dir)
        result_path = os.path.join(result_dir, f'{task_id}.json')
        safe_json_dump(final_results, result_path)

        # 完成回调
        if progress_callback:
            progress_callback({
                'task_id': task_id,
                'status': 'done',
                'progress': 100,
                'result_path': result_path,
                'summary': self.pipeline.alert_manager.get_summary(),
            })

        return final_results

    def stop(self):
        """停止处理"""
        self._stop_flag = True


class ImageProcessor:
    """
    单张图像处理器
    对单张图像进行检测和标注
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def process_image(self, image_path, task_id):
        """
        处理单张图像

        返回:
            {
                'task_id': str,
                'annotated_image_base64': str,
                'face_detected': bool,
                'fatigue': {...},
                'head_pose': {...},
                'gaze': {...},
                'distraction': {...},
                'alerts': [...],
                'summary': {...},
            }
        """
        image = cv2.imread(image_path)
        if image is None:
            return {'error': '无法读取图像文件'}

        self.pipeline.reset()

        # 调整图像大小
        h, w = image.shape[:2]
        if w > 1280:
            scale = 1280.0 / w
            image = cv2.resize(image, (1280, int(h * scale)))
            h, w = image.shape[:2]

        # 处理
        timestamp = 0.0
        result = self.pipeline.process_frame(image, timestamp)

        # 生成标注图
        face_result = self.pipeline.last_face_result
        annotated = self.pipeline.get_annotated_frame(
            image,
            face_result,
            result
        )

        # 编码为Base64
        import base64
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        # 构建返回结果
        output = {
            'task_id': task_id,
            'annotated_image_base64': annotated_b64,
            'face_detected': result.get('face_detected', False),
            'fatigue': result.get('fatigue', {}),
            'head_pose': result.get('head_pose', {}),
            'gaze': result.get('gaze', {}),
            'distraction': result.get('distraction', {}),
            'alerts': result.get('alerts', []),
            'summary': result.get('summary', {}),
        }

        # 保存结果
        result_dir = os.path.join(BASE_DIR, 'outputs')
        ensure_dir(result_dir)
        result_path = os.path.join(result_dir, f'{task_id}.json')
        safe_json_dump(output, result_path)

        return output
