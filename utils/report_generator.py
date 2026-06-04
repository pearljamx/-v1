"""
检测报告生成模块
===============
生成结构化的 HTML 检测报告，可在浏览器中查看或打印为 PDF。

优先使用 weasyprint 生成 PDF（如果已安装），否则保存为 HTML 文件。
"""

import os
import json
from datetime import datetime
from pathlib import Path
from utils.helpers import ensure_dir


# ---------------------------------------------------------------------------
# 尝试导入 weasyprint
# ---------------------------------------------------------------------------
try:
    from weasyprint import HTML as WeasyHTML
    _HAS_WEASYPRINT = True
except ImportError:
    _HAS_WEASYPRINT = False


# ---------------------------------------------------------------------------
# 样式字符串（内嵌 CSS）
# ---------------------------------------------------------------------------

_REPORT_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans",
                 "Microsoft YaHei", "PingFang SC", sans-serif;
    color: #1a1a2e;
    background: #fff;
    line-height: 1.6;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
}
.header {
    text-align: center;
    border-bottom: 3px solid #1a73e8;
    padding-bottom: 20px;
    margin-bottom: 24px;
}
.header h1 {
    font-size: 26px;
    color: #1a73e8;
    margin-bottom: 6px;
}
.header .subtitle {
    font-size: 14px;
    color: #666;
}
.header .meta {
    font-size: 12px;
    color: #999;
    margin-top: 4px;
}
.section {
    margin-bottom: 24px;
    page-break-inside: avoid;
}
.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #1a1a2e;
    border-left: 5px solid #1a73e8;
    padding-left: 12px;
    margin-bottom: 12px;
}
.risk-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 20px;
    font-size: 16px;
    font-weight: 700;
    color: #fff;
}
.risk-high { background: #d32f2f; }
.risk-medium { background: #f57c00; }
.risk-low { background: #388e3c; }
.score-grid {
    display: flex;
    gap: 16px;
    margin-bottom: 12px;
}
.score-card {
    flex: 1;
    background: #f5f7fa;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.score-card .label {
    font-size: 12px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.score-card .value {
    font-size: 32px;
    font-weight: 800;
    margin: 6px 0;
}
.score-card .unit {
    font-size: 12px;
    color: #888;
}
.value-good { color: #388e3c; }
.value-warning { color: #f57c00; }
.value-danger { color: #d32f2f; }
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
thead th {
    background: #f5f7fa;
    text-align: left;
    padding: 8px 12px;
    font-weight: 600;
    border-bottom: 2px solid #ddd;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}
tbody td {
    padding: 8px 12px;
    border-bottom: 1px solid #eee;
}
tbody tr:hover { background: #fafbfc; }
.severity-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
}
.severity-dot.low { background: #388e3c; }
.severity-dot.medium { background: #f57c00; }
.severity-dot.high { background: #d32f2f; }
.data-row {
    display: flex;
    gap: 16px;
    margin-bottom: 12px;
}
.data-item {
    flex: 1;
    background: #f5f7fa;
    border-radius: 6px;
    padding: 12px;
    text-align: center;
}
.data-item .label {
    font-size: 12px;
    color: #888;
}
.data-item .value {
    font-size: 20px;
    font-weight: 700;
}
.footer {
    text-align: center;
    border-top: 1px solid #eee;
    padding-top: 14px;
    margin-top: 30px;
    font-size: 12px;
    color: #aaa;
}
.physio-grid {
    display: flex;
    gap: 16px;
}
.physio-card {
    flex: 1;
    background: #fef3e0;
    border: 1px solid #f5c842;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
}
.physio-card .label {
    font-size: 12px;
    color: #b87a14;
}
.physio-card .value {
    font-size: 28px;
    font-weight: 700;
    color: #e65100;
}
.physio-card .unit {
    font-size: 11px;
    color: #b87a14;
}
.no-data {
    color: #aaa;
    font-style: italic;
    padding: 8px 0;
}
.tag-list {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.tag-item {
    display: inline-block;
    background: #e8f0fe;
    color: #1a73e8;
    border-radius: 16px;
    padding: 4px 14px;
    font-size: 12px;
}
"""


# ---------------------------------------------------------------------------
# 风险等级映射
# ---------------------------------------------------------------------------

_RISK_LABELS = {
    'high': '高风险',
    'medium': '中等风险',
    'low': '低风险',
}

_RISK_BADGE_CLASS = {
    'high': 'risk-high',
    'medium': 'risk-medium',
    'low': 'risk-low',
}

_SOURCE_LABELS = {
    'fatigue': '疲劳检测',
    'distraction': '分心检测',
    'head_pose': '头部姿态',
    'gaze': '视线检测',
    'physiological': '生理信号',
}

_ALERT_TYPE_LABELS = {
    'eye_closure': '闭眼',
    'yawn': '打哈欠',
    'blink_rate_low': '眨眼频率过低',
    'head_down': '低头',
    'gaze_deviation': '视线偏离',
    'hand_off_wheel': '手离方向盘',
    'phone_detected': '手机检测',
    'smoking_detected': '吸烟检测',
    'heart_rate_abnormal': '心率异常',
    'nodding': '点头',
}

_SEVERITY_LABELS = {
    'danger': '危险',
    'warning': '警告',
    'info': '信息',
}


# ---------------------------------------------------------------------------
# HTML 报告生成
# ---------------------------------------------------------------------------

def _build_html(result_data, task_id):
    """
    根据检测结果数据构建完整的 HTML 报告字符串。

    Parameters
    ----------
    result_data : dict
        检测结果 JSON 数据。
    task_id : str
        任务 ID。

    Returns
    -------
    str
        完整的 HTML 文档字符串。
    """
    summary = result_data.get('summary', {}) or {}
    fatigue = result_data.get('fatigue', {}) or {}
    head_pose = result_data.get('head_pose', {}) or {}
    gaze = result_data.get('gaze', {}) or {}
    distraction = result_data.get('distraction', {}) or {}
    physiological = result_data.get('physiological', {}) or {}
    alerts = result_data.get('alerts', []) or []

    overall_risk = summary.get('overall_risk', 'low')
    fatigue_score = summary.get('fatigue_score', 100)
    distraction_score = summary.get('distraction_score', 100)

    # 评分颜色
    def _score_class(val):
        if val >= 70:
            return 'value-good'
        if val >= 40:
            return 'value-warning'
        return 'value-danger'

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ----- 构建 HTML 片段 -----
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>DriveGuard 检测报告 - {task_id}</title>
<style>
{_REPORT_CSS}
</style>
</head>
<body>

<!-- ========== 报告头部 ========== -->
<div class="header">
    <h1>DriveGuard 驾驶注意力检测报告</h1>
    <div class="subtitle">基于面部特征与姿态分析的驾驶员状态评估</div>
    <div class="meta">任务ID: {task_id} &nbsp;|&nbsp; 生成时间: {now_str}</div>
</div>

<!-- ========== 综合风险评估 ========== -->
<div class="section">
    <div class="section-title">综合风险评估</div>
    <p>
        风险等级：
        <span class="risk-badge {_RISK_BADGE_CLASS.get(overall_risk, 'risk-low')}">
            {_RISK_LABELS.get(overall_risk, overall_risk)}
        </span>
    </p>
    <br>
    <div class="score-grid">
        <div class="score-card">
            <div class="label">疲劳评分</div>
            <div class="value {_score_class(fatigue_score)}">{fatigue_score}</div>
            <div class="unit">/ 100（越高越清醒）</div>
        </div>
        <div class="score-card">
            <div class="label">分心评分</div>
            <div class="value {_score_class(distraction_score)}">{distraction_score}</div>
            <div class="unit">/ 100（越高越专注）</div>
        </div>
    </div>
</div>

<!-- ========== 疲劳分析 ========== -->
<div class="section">
    <div class="section-title">疲劳检测分析</div>
    <div class="data-row">
        <div class="data-item">
            <div class="label">平均 EAR</div>
            <div class="value">{fatigue.get('avg_ear', fatigue.get('ear', '--'))}</div>
        </div>
        <div class="data-item">
            <div class="label">最低 EAR</div>
            <div class="value">{fatigue.get('min_ear', '--')}</div>
        </div>
        <div class="data-item">
            <div class="label">眨眼频率</div>
            <div class="value">{fatigue.get('blink_rate', '--')}</div>
        </div>
        <div class="data-item">
            <div class="label">点头次数</div>
            <div class="value">{fatigue.get('nodding_count', 0)}</div>
        </div>
    </div>
    <div class="data-row">
        <div class="data-item">
            <div class="label">平均 MAR</div>
            <div class="value">{fatigue.get('avg_mar', fatigue.get('mar', '--'))}</div>
        </div>
        <div class="data-item">
            <div class="label">闭眼检测</div>
            <div class="value">{'是' if fatigue.get('eye_closure') else '否'}</div>
        </div>
        <div class="data-item">
            <div class="label">哈欠检测</div>
            <div class="value">{'是' if fatigue.get('yawn') else '否'}</div>
        </div>
        <div class="data-item">
            <div class="label">检测到面部</div>
            <div class="value">{'是' if result_data.get('face_detected') else '否'}</div>
        </div>
    </div>
</div>

<!-- ========== 分心分析 ========== -->
<div class="section">
    <div class="section-title">分心检测分析</div>
    <div class="data-row">
        <div class="data-item">
            <div class="label">头部俯仰角 (Pitch)</div>
            <div class="value">{head_pose.get('pitch', '--')}&deg;</div>
        </div>
        <div class="data-item">
            <div class="label">头部偏航角 (Yaw)</div>
            <div class="value">{head_pose.get('yaw', '--')}&deg;</div>
        </div>
        <div class="data-item">
            <div class="label">头部翻滚角 (Roll)</div>
            <div class="value">{head_pose.get('roll', '--')}&deg;</div>
        </div>
    </div>
    <div class="data-row">
        <div class="data-item">
            <div class="label">视线偏离角</div>
            <div class="value">{gaze.get('gaze_angle', '--')}{'&deg;' if gaze.get('gaze_angle') is not None else ''}</div>
        </div>
        <div class="data-item">
            <div class="label">是否偏离</div>
            <div class="value">{'是' if gaze.get('is_deviated') else '否'}</div>
        </div>
        <div class="data-item">
            <div class="label">视线偏离次数</div>
            <div class="value">{len(distraction.get('gaze_deviation_events') or [])}</div>
        </div>
    </div>
    <div class="data-row">
        <div class="data-item" style="flex:2; text-align:left;">
            <div class="label">检测到的物体</div>
            <div class="tag-list">
"""

    # 检测物体标签
    objects = distraction.get('objects_detected') or []
    if objects:
        for obj in objects:
            cls_name = obj.get('class', '未知')
            count = obj.get('count', 1)
            html += f'                <span class="tag-item">{cls_name} ({count}次)</span>\n'
    else:
        html += '                <span class="no-data">未检测到异常物体</span>\n'

    html += """            </div>
        </div>
    </div>
</div>

<!-- ========== 生理信号 ========== -->
<div class="section">
    <div class="section-title">生理信号监测</div>
"""

    has_physio = any([
        physiological.get('heart_rate'),
        physiological.get('bp_systolic'),
        physiological.get('bp_diastolic'),
    ])

    if has_physio:
        hr = physiological.get('heart_rate', '--')
        bp_sys = physiological.get('bp_systolic', '--')
        bp_dia = physiological.get('bp_diastolic', '--')
        html += f"""    <div class="physio-grid">
        <div class="physio-card">
            <div class="label">心率</div>
            <div class="value">{hr}</div>
            <div class="unit">BPM</div>
        </div>
        <div class="physio-card">
            <div class="label">收缩压趋势</div>
            <div class="value">{bp_sys}</div>
            <div class="unit">mmHg</div>
        </div>
        <div class="physio-card">
            <div class="label">舒张压趋势</div>
            <div class="value">{bp_dia}</div>
            <div class="unit">mmHg</div>
        </div>
    </div>\n"""
    else:
        html += '    <p class="no-data">未启用生理信号检测或数据不足</p>\n'

    html += """</div>

<!-- ========== 告警记录 ========== -->
<div class="section">
    <div class="section-title">告警记录</div>
"""

    if alerts:
        html += """    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>时间</th>
                <th>来源</th>
                <th>类型</th>
                <th>严重性</th>
                <th>描述</th>
            </tr>
        </thead>
        <tbody>
"""
        for i, a in enumerate(alerts, 1):
            ts = a.get('timestamp', 0)
            if isinstance(ts, (int, float)):
                time_str = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            else:
                time_str = str(ts)
            source = _SOURCE_LABELS.get(a.get('source', ''), a.get('source', ''))
            atype = _ALERT_TYPE_LABELS.get(a.get('type', ''), a.get('type', ''))
            sev = a.get('severity', 'info')
            sev_label = _SEVERITY_LABELS.get(sev, sev)
            msg = a.get('message', '')

            html += f"""            <tr>
                <td>{i}</td>
                <td>{time_str}</td>
                <td>{source}</td>
                <td>{atype}</td>
                <td><span class="severity-dot {sev}"></span>{sev_label}</td>
                <td>{msg}</td>
            </tr>
"""
        html += """        </tbody>
    </table>
"""
    else:
        html += '    <p class="no-data">&#10004; 未检测到告警</p>\n'

    html += """</div>

<!-- ========== 页脚 ========== -->
<div class="footer">
    由 DriveGuard AI 检测系统自动生成 &nbsp;|&nbsp; 本报告仅供辅助参考，不构成医学诊断
</div>

</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def generate_report(result_data, output_path):
    """
    生成检测报告并保存。

    优先使用 weasyprint 输出 PDF（如果已安装），否则保存为 HTML 文件。

    Parameters
    ----------
    result_data : dict
        检测结果 JSON 数据，需含 task_id 字段或最外层包含它。
    output_path : str
        输出文件路径。如果使用 weasyprint，会输出 .pdf 文件；
        否则输出 .html 文件。

    Returns
    -------
    str
        实际生成的文件路径。
    """
    task_id = result_data.get('task_id', 'unknown')
    ensure_dir(Path(output_path).parent)

    html_content = _build_html(result_data, task_id)

    if _HAS_WEASYPRINT:
        pdf_path = output_path.replace('.html', '.pdf')
        WeasyHTML(string=html_content).write_pdf(pdf_path)
        return pdf_path
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return output_path
