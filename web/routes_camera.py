"""
摄像头实时检测路由
==================
提供摄像头检测页面和实时帧处理API。
使用HTTP轮询模式（非WebSocket），兼容Windows。
"""

import time
import threading
import logging
import numpy as np
import cv2
from flask import request, jsonify, render_template, current_app
from . import bp
from utils.lighting import compute_brightness, get_adaptive_thresholds


# ============================================================================
# 全局Session管理
# ============================================================================

# {session_id: {'processor': RealtimeProcessor, 'last_active': timestamp}}
_realtime_sessions = {}
_sessions_lock = threading.Lock()
SESSION_TIMEOUT = 60  # 秒，闲置超过此时间自动清理
logger = logging.getLogger(__name__)


def _get_or_create_processor(session_id, form_data=None):
    """
    获取或创建实时处理器实例

    参数:
        session_id: 客户端生成的会话ID
        form_data: 请求表单数据，用于解析模块开关

    返回:
        RealtimeProcessor 实例
    """
    with _sessions_lock:
        # 清理闲置session
        _cleanup_idle_sessions()

        if session_id not in _realtime_sessions:
            # 解析模块开关
            # 注意：实时摄像头默认关闭 distraction（YOLO推理较慢）和 physio（需要长视频），
            # 即使模型文件存在也不自动开启，以保障实时帧率。
            # 用户可通过前端开关手动启用。
            enable_modules = {
                'fatigue': True,
                'pose': True,
                'gaze': True,
                'distraction': False,  # YOLO推理在CPU上较慢，默认关闭保障实时性
                'physio': False,       # rPPG需要较长视频，默认关闭
            }
            if form_data:
                enable_modules['fatigue'] = form_data.get('enable_fatigue', 'true').lower() == 'true'
                enable_modules['pose'] = form_data.get('enable_pose', 'true').lower() == 'true'
                enable_modules['gaze'] = form_data.get('enable_gaze', 'true').lower() == 'true'
                enable_modules['distraction'] = form_data.get('enable_distraction', 'false').lower() == 'true'
                enable_modules['physio'] = form_data.get('enable_physio', 'false').lower() == 'true'

            from processors.realtime_processor import RealtimeProcessor
            _realtime_sessions[session_id] = {
                'processor': RealtimeProcessor(enable_modules=enable_modules),
                'last_active': time.time(),
            }

        # 更新活跃时间
        _realtime_sessions[session_id]['last_active'] = time.time()
        return _realtime_sessions[session_id]['processor']


def _cleanup_idle_sessions():
    """清理超过SESSION_TIMEOUT秒未活跃的session"""
    now = time.time()
    idle_sessions = [
        sid for sid, data in _realtime_sessions.items()
        if now - data['last_active'] > SESSION_TIMEOUT
    ]
    for sid in idle_sessions:
        try:
            _realtime_sessions[sid]['processor'].reset()
        except Exception:
            pass
        del _realtime_sessions[sid]


# ============================================================================
# 路由
# ============================================================================


@bp.route('/camera')
def camera_page():
    """摄像头实时检测页面"""
    return render_template('camera.html')


@bp.route('/camera/frame', methods=['POST'])
def process_camera_frame():
    """
    处理单帧摄像头图像

    请求: multipart/form-data
        - frame: JPEG图像文件 (Blob)
        - session_id: 客户端会话ID (UUID)
        - enable_fatigue/pose/gaze/distraction/physio: 模块开关

    响应: JSON
        {
            'face_detected': bool,
            'fatigue': {ear, mar, blink_rate},
            'head_pose': {pitch, yaw, roll},
            'gaze': {gaze_angle, is_deviated},
            'distraction': {...},
            'physio': {...},
            'alerts': [{type, severity, message, timestamp}, ...],
            'summary': {overall_risk, fatigue_score, distraction_score, total_alerts},
            'overlay': {纯图形叠加数据}
        }
    """
    if 'frame' not in request.files:
        return jsonify({'error': '未收到帧数据'}), 400

    session_id = request.form.get('session_id', '').strip()
    if not session_id:
        return jsonify({'error': '缺少session_id参数'}), 400
    file = request.files['frame']

    try:
        # 读取JPEG字节流并解码为OpenCV图像
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': '无效的图像数据'}), 400

        # 获取或创建处理器
        processor = _get_or_create_processor(session_id, request.form)

        # 处理帧
        result = processor.process_frame(frame)

        # 计算当前帧亮度，用于前端显示和自适应阈值调整
        brightness = compute_brightness(frame)
        lighting_info = get_adaptive_thresholds(brightness)
        # 前端可通过 lighting_info 获取:
        #   - lighting_level: 当前光照等级 (dark/dim/normal/bright)
        #   - ear_threshold / mar_threshold: 自适应阈值
        #   - brightness: 平均亮度值

        # 构建响应
        response = {
            'face_detected': result['face_detected'],
            'fatigue': result['fatigue'],
            'head_pose': result['head_pose'],
            'gaze': result['gaze'],
            'distraction': result.get('distraction', {}),
            'physio': result.get('physio', {}),
            'alerts': result['alerts'],
            'summary': result['summary'],
            'overlay': result.get('overlay', {}),
            'lighting': lighting_info,
        }

        return jsonify(response)

    except Exception as e:
        logger.exception("摄像头帧处理失败: session_id=%s", session_id)
        return jsonify({'error': str(e)}), 500


@bp.route('/camera/session/stop', methods=['POST'])
def stop_camera_session():
    """
    停止摄像头session并释放资源

    请求: JSON {'session_id': str}
    """
    data = request.get_json() or {}
    session_id = data.get('session_id', '')

    with _sessions_lock:
        if session_id in _realtime_sessions:
            try:
                _realtime_sessions[session_id]['processor'].reset()
            except Exception:
                pass
            del _realtime_sessions[session_id]
            return jsonify({'status': 'ok', 'message': 'Session已释放'})

    return jsonify({'status': 'ok', 'message': 'Session不存在或已释放'})


@bp.route('/camera/sessions')
def list_sessions():
    """列出当前活跃的摄像头session（调试用）"""
    with _sessions_lock:
        sessions_info = {
            sid: {
                'last_active': data['last_active'],
                'idle_seconds': time.time() - data['last_active'],
                'frame_count': data['processor'].frame_count,
            }
            for sid, data in _realtime_sessions.items()
        }
    return jsonify({
        'active_sessions': len(_realtime_sessions),
        'sessions': sessions_info,
    })
