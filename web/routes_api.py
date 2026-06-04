"""
结果查询API路由模块
===================
提供任务状态查询、结果获取、标注帧访问等REST API。
"""

import os
from flask import request, jsonify, send_file, current_app
from . import bp
from utils.helpers import safe_json_load, ensure_dir


def _get_output_dir():
    return current_app.config.get('OUTPUT_FOLDER', 'outputs')


@bp.route('/api/status/<task_id>')
def get_task_status(task_id):
    """
    查询任务处理状态

    响应:
        {
            'task_id': str,
            'status': 'queued' | 'processing' | 'done' | 'error',
            'progress': 0-100,
            'current_frame': int,
            'total_frames': int,
            'current_alerts': [...],
            'preview_frame': str (base64, 仅处理中),
            'summary': {...} (仅完成时),
            'error': str (仅出错时)
        }
    """
    status_path = os.path.join(_get_output_dir(), f'{task_id}_status.json')
    status_data = safe_json_load(status_path, {'status': 'unknown'})

    # 查找上传元数据确定任务是否存在
    if status_data.get('status') == 'unknown':
        # 检查任务元数据是否存在
        upload_dir = current_app.config.get('UPLOAD_FOLDER', 'uploads')
        meta_found = False
        for ftype in ['images', 'videos']:
            meta_path = os.path.join(upload_dir, ftype, f'{task_id}_meta.json')
            if os.path.exists(meta_path):
                meta_found = True
                break
        if not meta_found:
            return jsonify({'error': '任务不存在'}), 404

        status_data = {'status': 'queued', 'progress': 0}

    return jsonify({
        'task_id': task_id,
        **status_data
    })


@bp.route('/api/results/<task_id>')
def get_task_results(task_id):
    """
    获取完整检测结果

    响应:
        完整的检测结果JSON，包含summary, fatigue, head_pose,
        distraction, physiological, alerts, annotated_frames等
    """
    # 先从status文件获取
    status_path = os.path.join(_get_output_dir(), f'{task_id}_status.json')
    status_data = safe_json_load(status_path, None)

    if status_data:
        # 移除大型base64数据以减少响应大小（前端可通过/api/frame获取）
        if 'annotated_frames' in status_data:
            del status_data['annotated_frames']
        return jsonify(status_data)

    # 尝试从完整结果文件获取
    result_path = os.path.join(_get_output_dir(), f'{task_id}.json')
    result_data = safe_json_load(result_path, None)

    if result_data:
        if 'annotated_frames' in result_data:
            del result_data['annotated_frames']
        if 'annotated_image_base64' in result_data:
            del result_data['annotated_image_base64']
        return jsonify(result_data)

    return jsonify({'error': '结果不存在'}), 404


@bp.route('/api/frame/<task_id>/<int:frame_idx>')
def get_annotated_frame(task_id, frame_idx):
    """
    获取指定索引的标注帧图像

    返回: JPEG图像
    """
    import base64
    from io import BytesIO
    from flask import Response

    status_path = os.path.join(_get_output_dir(), f'{task_id}_status.json')
    status_data = safe_json_load(status_path, {})

    frames = status_data.get('annotated_frames', [])
    if frame_idx < 0 or frame_idx >= len(frames):
        # 尝试从完整结果文件获取
        result_path = os.path.join(_get_output_dir(), f'{task_id}.json')
        result_data = safe_json_load(result_path, {})
        frames = result_data.get('annotated_frames', [])

    if frame_idx < 0 or frame_idx >= len(frames):
        return jsonify({'error': '帧索引超出范围'}), 404

    frame_b64 = frames[frame_idx]
    img_data = base64.b64decode(frame_b64)

    return Response(img_data, mimetype='image/jpeg')


@bp.route('/api/summary/<task_id>')
def get_task_summary(task_id):
    """
    获取任务摘要统计

    响应:
        {
            'total_alerts': int,
            'fatigue_score': int,
            'distraction_score': int,
            'overall_risk': 'low' | 'medium' | 'high',
            'heart_rate': float | None,
            'alerts_by_type': {...}
        }
    """
    status_path = os.path.join(_get_output_dir(), f'{task_id}_status.json')
    status_data = safe_json_load(status_path, {})

    summary = status_data.get('summary', {})

    # 添加心率信息
    physiological = status_data.get('physiological', {})
    if physiological:
        summary['heart_rate'] = physiological.get('heart_rate')
        summary['signal_quality'] = physiological.get('signal_quality')

    return jsonify(summary)


@bp.route('/api/annotated-image/<task_id>')
def get_annotated_image(task_id):
    """
    获取图像检测的标注结果图像

    返回: JPEG图像
    """
    import base64
    from flask import Response

    # 从状态文件获取
    status_path = os.path.join(_get_output_dir(), f'{task_id}_status.json')
    status_data = safe_json_load(status_path, {})

    img_b64 = status_data.get('annotated_image_base64')

    if not img_b64:
        # 从完整结果文件获取
        result_path = os.path.join(_get_output_dir(), f'{task_id}.json')
        result_data = safe_json_load(result_path, {})
        img_b64 = result_data.get('annotated_image_base64')

    if not img_b64:
        return jsonify({'error': '标注图像不存在'}), 404

    img_data = base64.b64decode(img_b64)
    return Response(img_data, mimetype='image/jpeg')


@bp.route('/api/download/<task_id>')
def download_result(task_id):
    """
    下载完整检测结果JSON文件
    """
    result_path = os.path.join(_get_output_dir(), f'{task_id}.json')
    if not os.path.exists(result_path):
        # 尝试状态文件
        status_path = os.path.join(_get_output_dir(), f'{task_id}_status.json')
        if os.path.exists(status_path):
            return send_file(status_path, mimetype='application/json',
                             as_attachment=True, download_name=f'{task_id}_result.json')
        return jsonify({'error': '结果文件不存在'}), 404

    return send_file(result_path, mimetype='application/json',
                     as_attachment=True, download_name=f'{task_id}_result.json')


# ============================================================================
# 检测历史数据库 API
# ============================================================================

@bp.route('/api/history')
def get_history():
    """
    获取分页检测历史记录

    Query 参数:
        limit : int, 默认 50
        offset: int, 默认 0

    响应:
        {
            'records': [...],
            'total': int,
            'limit': int,
            'offset': int
        }
    """
    try:
        from models.database import DetectionHistoryDB
    except ImportError:
        return jsonify({'error': '数据库模块不可用'}), 500

    db = DetectionHistoryDB()
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)

    # 限制每页最大数量
    limit = min(max(1, limit), 200)

    records, total = db.get_history(limit=limit, offset=offset)
    return jsonify({
        'records': records,
        'total': total,
        'limit': limit,
        'offset': offset,
    })


@bp.route('/api/history/stats')
def get_history_stats():
    """
    获取检测历史聚合统计

    Query 参数:
        start_date: str, YYYY-MM-DD 格式，可选
        end_date  : str, YYYY-MM-DD 格式，可选

    响应:
        {
            'total_tasks': int,
            'avg_fatigue_score': float,
            'avg_distraction_score': float,
            'risk_distribution': {'low': int, 'medium': int, 'high': int},
            'avg_heart_rate': float | None,
            'avg_alert_count': float,
            ...
        }
    """
    try:
        from models.database import DetectionHistoryDB
    except ImportError:
        return jsonify({'error': '数据库模块不可用'}), 500

    db = DetectionHistoryDB()
    start_date = request.args.get('start_date', None, type=str)
    end_date = request.args.get('end_date', None, type=str)

    stats = db.get_stats(start_date=start_date, end_date=end_date)
    return jsonify(stats)


@bp.route('/api/history/<task_id>')
def get_history_item(task_id):
    """
    从数据库获取指定任务的检测记录

    响应:
        完整的检测记录对象，包含 alerts 和 summary
    """
    try:
        from models.database import DetectionHistoryDB
    except ImportError:
        return jsonify({'error': '数据库模块不可用'}), 500

    db = DetectionHistoryDB()
    record = db.get_result(task_id)

    if record is None:
        return jsonify({'error': '数据库中没有该任务的记录'}), 404

    return jsonify(record)


@bp.route('/api/history/<task_id>', methods=['DELETE'])
def delete_history_item(task_id):
    """
    删除数据库中指定任务的检测记录

    响应:
        {'message': str, 'deleted': bool}
    """
    try:
        from models.database import DetectionHistoryDB
    except ImportError:
        return jsonify({'error': '数据库模块不可用'}), 500

    db = DetectionHistoryDB()
    deleted = db.delete_result(task_id)

    if not deleted:
        return jsonify({'error': '数据库中没有该任务的记录'}), 404

    return jsonify({'message': '记录已删除', 'deleted': True})


# ============================================================================
# 报告生成 API
# ============================================================================

@bp.route('/api/report/<task_id>')
def generate_report_api(task_id):
    """
    生成并返回检测报告

    响应:
        HTML 文件或 PDF 文件（如果 weasyprint 可用）
    """
    try:
        from utils.report_generator import generate_report
    except ImportError:
        return jsonify({'error': '报告生成模块不可用'}), 500

    # 先尝试从状态文件获取结果数据
    status_path = os.path.join(_get_output_dir(), f'{task_id}_status.json')
    result_data = safe_json_load(status_path, None)

    if result_data is None:
        result_path = os.path.join(_get_output_dir(), f'{task_id}.json')
        result_data = safe_json_load(result_path, None)

    if result_data is None:
        return jsonify({'error': '检测结果不存在，无法生成报告'}), 404

    # 确保包含 task_id
    result_data['task_id'] = task_id

    # 生成输出路径
    output_dir = os.path.join(current_app.config.get('OUTPUT_FOLDER', 'outputs'))
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, f'{task_id}_report.html')

    try:
        report_path = generate_report(result_data, output_path)

        # 根据文件扩展名设置 MIME 类型
        if report_path.endswith('.pdf'):
            return send_file(
                report_path,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'{task_id}_report.pdf',
            )
        else:
            return send_file(
                report_path,
                mimetype='text/html',
                as_attachment=False,
            )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'报告生成失败: {str(e)}'}), 500
