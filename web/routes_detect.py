"""
检测任务路由模块
================
处理检测任务的启动和管理。视频处理在后台线程中执行。
"""

import os
import logging
import threading
from datetime import datetime
from flask import request, jsonify, current_app
from . import bp
from utils.helpers import safe_json_load, safe_json_dump, ensure_dir


# 全局任务状态字典（内存中跟踪正在运行的任务）
_running_tasks = {}
_task_lock = threading.Lock()
logger = logging.getLogger(__name__)


def _cleanup_running_task(task_id):
    """后台任务结束后清理内存中的线程引用。"""
    with _task_lock:
        _running_tasks.pop(task_id, None)


def _save_to_database(task_id, result):
    """将检测结果保存到 SQLite 历史数据库。"""
    try:
        from models.database import DetectionHistoryDB

        meta = _get_task_meta(task_id)

        db = DetectionHistoryDB()
        db.save_result(task_id, {
            'timestamp': result.get('timestamp', datetime.now().isoformat()),
            'file_name': meta.get('filename', meta.get('original_filename', '')) if meta else '',
            'file_type': meta.get('file_type', 'image') if meta else 'image',
            'overall_risk': (
                result.get('summary', {}).get('overall_risk', 'low')
                if isinstance(result.get('summary'), dict) else 'low'
            ),
            'fatigue_score': (
                result.get('summary', {}).get('fatigue_score', 100.0)
                if isinstance(result.get('summary'), dict) else 100.0
            ),
            'distraction_score': (
                result.get('summary', {}).get('distraction_score', 100.0)
                if isinstance(result.get('summary'), dict) else 100.0
            ),
            'heart_rate': (
                result.get('physiological', {}).get('heart_rate')
                if isinstance(result.get('physiological'), dict) else None
            ),
            'bp_systolic': (
                result.get('physiological', {}).get('bp_systolic')
                if isinstance(result.get('physiological'), dict) else None
            ),
            'bp_diastolic': (
                result.get('physiological', {}).get('bp_diastolic')
                if isinstance(result.get('physiological'), dict) else None
            ),
            'alerts': result.get('alerts', []),
            'summary': result.get('summary', {}),
        })
    except Exception:
        logger.exception("保存检测历史失败: task_id=%s", task_id)


def _get_task_meta(task_id):
    """读取任务元数据"""
    for ftype in ['images', 'videos']:
        meta_path = os.path.join(
            current_app.config['UPLOAD_FOLDER'], ftype, f'{task_id}_meta.json'
        )
        if os.path.exists(meta_path):
            return safe_json_load(meta_path)
    return None


def _save_task_status(task_id, status_data):
    """保存任务状态到文件"""
    output_dir = os.path.join(current_app.config.get('OUTPUT_FOLDER', 'outputs'))
    ensure_dir(output_dir)
    status_path = os.path.join(output_dir, f'{task_id}_status.json')
    safe_json_dump(status_data, status_path)


def _get_task_status(task_id):
    """获取任务状态"""
    output_dir = os.path.join(current_app.config.get('OUTPUT_FOLDER', 'outputs'))
    status_path = os.path.join(output_dir, f'{task_id}_status.json')
    return safe_json_load(status_path, {'status': 'unknown'})


@bp.route('/detect/image', methods=['POST'])
def detect_image():
    """
    启动图像检测任务

    请求: JSON
        {'task_id': str}

    响应:
        {'task_id': str, 'status': 'processing'}
    """
    data = request.get_json() or {}
    task_id = data.get('task_id')

    if not task_id:
        return jsonify({'error': '缺少task_id参数'}), 400

    meta = _get_task_meta(task_id)
    if meta is None:
        return jsonify({'error': '任务不存在'}), 404

    if meta['file_type'] != 'image':
        return jsonify({'error': '请使用图像检测接口处理图像文件'}), 400

    # 标记为处理中
    _save_task_status(task_id, {'status': 'processing', 'progress': 0})
    app_obj = current_app._get_current_object()

    # 在后台线程中处理
    def _process():
        with app_obj.app_context():
            try:
                from processors.frame_pipeline import FramePipeline
                from processors.video_processor import ImageProcessor

                pipeline = FramePipeline(enable_modules=meta.get('enable_modules'))
                image_processor = ImageProcessor(pipeline)

                result = image_processor.process_image(meta['file_path'], task_id)
                result['status'] = 'done'
                _save_task_status(task_id, result)
                _save_to_database(task_id, result)

            except Exception as e:
                logger.exception("图像检测任务失败: task_id=%s", task_id)
                _save_task_status(task_id, {
                    'status': 'error',
                    'error': str(e),
                })
            finally:
                _cleanup_running_task(task_id)

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()

    with _task_lock:
        _running_tasks[task_id] = thread

    return jsonify({'task_id': task_id, 'status': 'processing'})


@bp.route('/detect/video', methods=['POST'])
def detect_video():
    """
    启动视频检测任务

    请求: JSON
        {'task_id': str}

    响应:
        {'task_id': str, 'status': 'processing'}
    """
    data = request.get_json() or {}
    task_id = data.get('task_id')

    if not task_id:
        return jsonify({'error': '缺少task_id参数'}), 400

    meta = _get_task_meta(task_id)
    if meta is None:
        return jsonify({'error': '任务不存在'}), 404

    if meta['file_type'] != 'video':
        return jsonify({'error': '请使用视频检测接口处理视频文件'}), 400

    # 标记为处理中
    _save_task_status(task_id, {
        'status': 'processing',
        'progress': 0,
        'current_frame': 0,
        'current_alerts': [],
    })
    app_obj = current_app._get_current_object()

    # 在后台线程中处理
    def _process():
        with app_obj.app_context():
            try:
                from processors.frame_pipeline import FramePipeline
                from processors.video_processor import VideoProcessor

                pipeline = FramePipeline(enable_modules=meta.get('enable_modules'))
                video_processor = VideoProcessor(pipeline)

                def progress_callback(progress_data):
                    """进度更新回调"""
                    _save_task_status(task_id, {
                        'status': progress_data.get('status', 'processing'),
                        'progress': progress_data.get('progress', 0),
                        'current_frame': progress_data.get('current_frame', 0),
                        'total_frames': progress_data.get('total_frames', 0),
                        'current_alerts': progress_data.get('current_alerts', []),
                        'preview_frame': progress_data.get('preview_frame'),
                        'summary': progress_data.get('summary'),
                    })

                result = video_processor.process_video(
                    meta['file_path'],
                    task_id,
                    progress_callback=progress_callback,
                    output_video=False  # 可配置
                )

                # 最终状态更新
                status_data = {
                    'status': 'done',
                    'progress': 100,
                    'summary': result.get('summary', {}),
                    'fatigue': result.get('fatigue', {}),
                    'head_pose': result.get('head_pose', {}),
                    'distraction': result.get('distraction', {}),
                    'physiological': result.get('physiological', {}),
                    'alerts': result.get('alerts', []),
                    'annotated_frames': result.get('annotated_frames', []),
                    'video_info': result.get('video_info', {}),
                }
                _save_task_status(task_id, status_data)
                _save_to_database(task_id, status_data)

            except Exception as e:
                logger.exception("视频检测任务失败: task_id=%s", task_id)
                _save_task_status(task_id, {
                    'status': 'error',
                    'error': str(e),
                })
            finally:
                _cleanup_running_task(task_id)

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()

    with _task_lock:
        _running_tasks[task_id] = thread

    return jsonify({'task_id': task_id, 'status': 'processing'})


@bp.route('/detect/auto', methods=['POST'])
def detect_auto():
    """
    自动检测（根据文件类型路由到图像或视频处理）

    请求: JSON
        {'task_id': str}
    """
    data = request.get_json() or {}
    task_id = data.get('task_id')

    if not task_id:
        return jsonify({'error': '缺少task_id参数'}), 400

    meta = _get_task_meta(task_id)
    if meta is None:
        return jsonify({'error': '任务不存在'}), 404

    if meta.get('detect_type') == 'image' or meta.get('file_type') == 'image':
        return detect_image()
    else:
        return detect_video()
