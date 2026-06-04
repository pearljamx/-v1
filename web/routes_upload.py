"""
文件上传路由模块
================
处理图像和视频文件的上传、验证和存储。
"""

import os
from flask import request, jsonify, current_app
from werkzeug.utils import secure_filename
from . import bp
from utils.helpers import generate_task_id, allowed_file, ensure_dir


@bp.route('/upload', methods=['POST'])
def upload_file():
    """
    上传图像或视频文件

    请求: multipart/form-data
        - file: 文件
        - mode: 'auto' | 'image' | 'video'
        - enable_fatigue: 'true' | 'false'
        - enable_pose: 'true' | 'false'
        - enable_gaze: 'true' | 'false'
        - enable_distraction: 'true' | 'false'
        - enable_physio: 'true' | 'false'

    响应:
        {
            'task_id': str,
            'filename': str,
            'file_type': 'image' | 'video',
            'detect_type': 'image' | 'video',
            'file_size': int,
            'status': 'queued'
        }
    """
    if 'file' not in request.files:
        return jsonify({'error': '未找到上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    # 验证文件类型
    if not allowed_file(file.filename):
        return jsonify({
            'error': '不支持的文件格式，请上传图像(JPG/PNG/BMP)或视频(MP4/AVI/MOV)文件'
        }), 400

    # 生成任务ID和安全文件名
    task_id = generate_task_id()
    original_filename = secure_filename(file.filename)
    ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''

    # 确定文件类型
    image_exts = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
    video_exts = {'mp4', 'avi', 'mov', 'webm', 'mkv'}

    if ext in image_exts:
        file_type = 'image'
    elif ext in video_exts:
        file_type = 'video'
    else:
        return jsonify({'error': '无法识别的文件类型'}), 400

    # 确定检测模式
    mode = request.form.get('mode', 'auto')
    if mode == 'auto':
        detect_type = file_type
    else:
        detect_type = mode

    # 保存文件
    upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], file_type + 's')
    ensure_dir(upload_dir)

    saved_filename = f"{task_id}.{ext}"
    file_path = os.path.join(upload_dir, saved_filename)
    file.save(file_path)

    # 获取文件大小
    file_size = os.path.getsize(file_path)

    # 解析模块开关
    enable_modules = {
        'fatigue': request.form.get('enable_fatigue', 'true').lower() == 'true',
        'pose': request.form.get('enable_pose', 'true').lower() == 'true',
        'gaze': request.form.get('enable_gaze', 'true').lower() == 'true',
        'distraction': request.form.get('enable_distraction', 'true').lower() == 'true',
        'physio': request.form.get('enable_physio', 'false').lower() == 'true',
    }

    # 保存任务元数据
    import json
    task_meta = {
        'task_id': task_id,
        'filename': original_filename,
        'saved_filename': saved_filename,
        'file_path': file_path,
        'file_type': file_type,
        'detect_type': detect_type,
        'file_size': file_size,
        'enable_modules': enable_modules,
        'status': 'queued',
    }
    meta_path = os.path.join(upload_dir, f'{task_id}_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(task_meta, f, ensure_ascii=False, indent=2)

    return jsonify({
        'task_id': task_id,
        'filename': original_filename,
        'file_type': file_type,
        'detect_type': detect_type,
        'file_size': file_size,
        'status': 'queued',
    })
