"""
主页路由模块
============
处理首页和结果页面渲染，以及面部识别注册接口。
"""

import os
import io
import numpy as np
from PIL import Image
from flask import render_template, jsonify, current_app, request
from . import bp


@bp.route('/')
def index():
    """系统首页 — 上传与检测面板"""
    return render_template('index.html')


@bp.route('/result/<task_id>')
def result_page(task_id):
    """检测结果详情页"""
    return render_template('result.html', task_id=task_id)


# 全局缓存的FaceDetector实例（避免反复创建/销毁导致TFLite崩溃）
_cached_detector = None


@bp.route('/api/heartbeat')
def heartbeat():
    """系统健康检查 & 模型状态（使用缓存FaceDetector避免重复初始化）"""
    global _cached_detector
    try:
        if _cached_detector is None:
            from detectors.face_detector import FaceDetector
            _cached_detector = FaceDetector()
        status = {
            'status': 'ok',
            'models_loaded': True,
            'backend': _cached_detector.backend if hasattr(_cached_detector, 'backend') else 'unknown',
        }
    except Exception as e:
        status = {
            'status': 'degraded',
            'models_loaded': False,
            'error': str(e),
        }

    return jsonify(status)


@bp.route('/api/models')
def list_models():
    """列出已加载的模型信息"""
    from config import (
        YOLO_HANDHELD_MODEL,
        YOLO_DRIVER_STATE_MODEL,
        YOLO_STEERING_HAND_MODEL,
        YOLO_POSE_MODEL,
        BP_LSTM_MODEL,
    )

    drowsiness_cls_model = os.path.join(
        os.path.dirname(YOLO_HANDHELD_MODEL), 'yolo_drowsiness_cls.pt'
    )
    models = {
        'face_detector': {
            'description': 'face_recognition/dlib 68点 + MediaPipe/Yunet fallback',
            'available': True,
        },
        'yolo_driver_state': {
            'description': 'YOLOv8 驾驶状态检测模型',
            'path': YOLO_DRIVER_STATE_MODEL,
            'available': os.path.exists(YOLO_DRIVER_STATE_MODEL),
        },
        'yolo_handheld': {
            'description': 'YOLOv8 手持物/手机/抽烟/饮水检测模型',
            'path': YOLO_HANDHELD_MODEL,
            'available': os.path.exists(YOLO_HANDHELD_MODEL),
        },
        'yolo_steering_hand': {
            'description': 'YOLOv8 方向盘/手部真实数据集检测模型（可选）',
            'path': YOLO_STEERING_HAND_MODEL,
            'available': os.path.exists(YOLO_STEERING_HAND_MODEL),
        },
        'yolo_pose': {
            'description': 'YOLOv8n-pose 人体姿态估计模型',
            'path': YOLO_POSE_MODEL,
            'available': os.path.exists(YOLO_POSE_MODEL),
        },
        'yolo_drowsiness_cls': {
            'description': 'YOLOv8 真实数据集驾驶疲劳二分类模型',
            'path': drowsiness_cls_model,
            'available': os.path.exists(drowsiness_cls_model),
        },
        'bp_lstm': {
            'description': 'PPG血压趋势预测 LSTM 模型',
            'path': BP_LSTM_MODEL,
            'available': os.path.exists(BP_LSTM_MODEL),
        },
    }
    return jsonify(models)


# 全局缓存的FaceRecognizer实例
_cached_recognizer = None


def _get_recognizer():
    """获取或创建缓存的 FaceRecognizer 实例。"""
    global _cached_recognizer
    if _cached_recognizer is None:
        from detectors.face_recognizer import FaceRecognizer
        _cached_recognizer = FaceRecognizer()
    return _cached_recognizer


@bp.route('/api/face/register', methods=['POST'])
def register_face():
    """
    注册驾驶员面部

    请求: multipart/form-data
        - file: 人脸图像文件 (JPG/PNG/BMP)
        - name: 驾驶员姓名

    响应:
        {
            'success': bool,
            'name': str,
            'message': str,
        }
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '未找到上传文件'}), 400

    file = request.files['file']
    name = request.form.get('name', '').strip()

    if file.filename == '':
        return jsonify({'success': False, 'error': '文件名为空'}), 400

    if not name:
        return jsonify({'success': False, 'error': '驾驶员姓名不能为空'}), 400

    # 读取图像
    try:
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        # PIL RGB -> OpenCV BGR
        face_image = np.array(pil_img)[:, :, ::-1].copy()
    except Exception as e:
        return jsonify({'success': False, 'error': f'图像读取失败: {str(e)}'}), 400

    # 注册
    try:
        recognizer = _get_recognizer()
        ok = recognizer.register_driver(name, face_image)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'面部识别模型初始化失败: {str(e)}'
        }), 500

    if ok:
        return jsonify({
            'success': True,
            'name': name,
            'message': f'驾驶员 "{name}" 注册成功',
        })
    else:
        return jsonify({
            'success': False,
            'name': name,
            'error': '未检测到人脸，请使用清晰的面部图像',
        }), 400


@bp.route('/api/face/identify', methods=['POST'])
def identify_face():
    """
    识别驾驶员面部

    请求: multipart/form-data
        - file: 人脸图像文件 (JPG/PNG/BMP)

    响应:
        {
            'success': bool,
            'identified': bool,
            'name': str | null,
            'confidence': float | null,
            'message': str,
        }
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '未找到上传文件'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'error': '文件名为空'}), 400

    # 读取图像
    try:
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        face_image = np.array(pil_img)[:, :, ::-1].copy()
    except Exception as e:
        return jsonify({'success': False, 'error': f'图像读取失败: {str(e)}'}), 400

    # 识别
    try:
        recognizer = _get_recognizer()
        result = recognizer.identify_driver(face_image)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'面部识别模型初始化失败: {str(e)}'
        }), 500

    if result is None:
        return jsonify({
            'success': True,
            'identified': False,
            'name': None,
            'confidence': None,
            'message': '未识别到已注册驾驶员，或未检测到人脸',
        })
    else:
        name, confidence = result
        return jsonify({
            'success': True,
            'identified': True,
            'name': name,
            'confidence': round(float(confidence), 4),
            'message': f'识别为 "{name}" (相似度: {confidence:.2%})',
        })


@bp.route('/api/face/drivers', methods=['GET'])
def list_drivers():
    """
    列出所有已注册驾驶员

    响应:
        {
            'success': bool,
            'drivers': [str, ...],
            'count': int,
        }
    """
    try:
        recognizer = _get_recognizer()
        drivers = recognizer.list_drivers()
        return jsonify({
            'success': True,
            'drivers': drivers,
            'count': len(drivers),
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取驾驶员列表失败: {str(e)}'
        }), 500


@bp.route('/api/face/drivers/<name>', methods=['DELETE'])
def delete_driver(name):
    """
    删除已注册驾驶员

    路径参数:
        name: 驾驶员姓名

    响应:
        {
            'success': bool,
            'message': str,
        }
    """
    try:
        recognizer = _get_recognizer()
        ok = recognizer.delete_driver(name)
        if ok:
            return jsonify({
                'success': True,
                'message': f'驾驶员 "{name}" 已删除',
            })
        else:
            return jsonify({
                'success': False,
                'error': f'驾驶员 "{name}" 不存在',
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'删除失败: {str(e)}'
        }), 500
