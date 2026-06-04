"""
统一告警管理器
==============
负责收集、管理、去重所有检测模块产生的告警，并提供统计摘要。
"""

import time
from collections import defaultdict


class Alert:
    """单条告警记录"""

    def __init__(self, source, alert_type, severity, timestamp, message, metadata=None):
        self.source = source          # 'fatigue', 'head_pose', 'gaze', 'distraction', 'physiological'
        self.alert_type = alert_type  # 具体告警类型
        self.severity = severity      # 'info', 'warning', 'danger'
        self.timestamp = timestamp
        self.message = message
        self.metadata = metadata or {}

    def to_dict(self):
        return {
            'source': self.source,
            'type': self.alert_type,
            'severity': self.severity,
            'timestamp': self.timestamp,
            'message': self.message,
            'metadata': self.metadata
        }


class AlertManager:
    """统一告警管理器"""

    def __init__(self):
        self.alerts = []                          # 所有告警列表
        self.active_alerts = {}                   # 当前活跃告警 {alert_type: Alert}
        self.alert_counts = defaultdict(int)      # 各类型告警计数
        self.alert_by_source = defaultdict(int)   # 各来源告警计数

    def add_alert(self, alert_dict):
        """
        添加告警
        alert_dict 必须包含: source, alert_type, severity, timestamp, message
        可选: metadata
        """
        if alert_dict is None:
            return

        alert = Alert(
            source=alert_dict.get('source', 'unknown'),
            alert_type=alert_dict.get('type', alert_dict.get('alert_type', 'unknown')),
            severity=alert_dict.get('severity', 'info'),
            timestamp=alert_dict.get('timestamp', time.time()),
            message=alert_dict.get('message', ''),
            metadata=alert_dict.get('metadata', {})
        )

        # 加入告警列表
        self.alerts.append(alert)

        # 更新活跃告警（同类型覆盖）
        self.active_alerts[alert.alert_type] = alert

        # 更新计数
        self.alert_counts[alert.alert_type] += 1
        self.alert_by_source[alert.source] += 1

    def clear_alert(self, alert_type):
        """清除某类型的活跃告警"""
        if alert_type in self.active_alerts:
            del self.active_alerts[alert_type]

    def get_active_alerts(self):
        """获取当前活跃告警列表（字典格式）"""
        return [a.to_dict() for a in self.active_alerts.values()]

    def get_summary(self):
        """
        获取告警统计摘要
        返回:
        {
            'total_alerts': int,
            'active_alerts': int,
            'by_severity': {'danger': N, 'warning': N, 'info': N},
            'by_type': {...},
            'fatigue_score': int (0-100, 100=正常),
            'distraction_score': int (0-100, 100=正常),
            'overall_risk': 'low' | 'medium' | 'high'
        }
        """
        # 按严重性统计
        by_severity = {'danger': 0, 'warning': 0, 'info': 0}
        for alert in self.alerts:
            by_severity[alert.severity] = by_severity.get(alert.severity, 0) + 1

        # 按类型统计
        by_type = dict(self.alert_counts)

        # 计算疲劳评分（基于疲劳相关告警）
        fatigue_alerts = {
            'eye_closure': 30,       # 每扣30分
            'yawn': 15,              # 每扣15分
            'low_blink_rate': 20,    # 每扣20分
            'nodding': 25,           # 每扣25分
        }
        fatigue_score = 100
        for atype, penalty in fatigue_alerts.items():
            fatigue_score -= self.alert_counts.get(atype, 0) * penalty
        fatigue_score = max(0, min(100, fatigue_score))

        # 计算分心评分
        distraction_alerts = {
            'gaze_deviation': 20,
            'phone_usage': 25,
            'smoking': 20,
            'hands_off_wheel': 30,
            'body_turn': 25,
        }
        distraction_score = 100
        for atype, penalty in distraction_alerts.items():
            distraction_score -= self.alert_counts.get(atype, 0) * penalty
        distraction_score = max(0, min(100, distraction_score))

        # 综合风险等级
        active_danger = sum(1 for a in self.active_alerts.values() if a.severity == 'danger')
        active_warning = sum(1 for a in self.active_alerts.values() if a.severity == 'warning')

        if active_danger > 0:
            overall_risk = 'high'
        elif active_warning > 0:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'

        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(self.active_alerts),
            'by_severity': by_severity,
            'by_type': by_type,
            'fatigue_score': fatigue_score,
            'distraction_score': distraction_score,
            'overall_risk': overall_risk,
        }

    def get_all_alerts(self):
        """获取所有告警（字典格式）"""
        return [a.to_dict() for a in self.alerts]

    def get_alerts_by_source(self, source):
        """获取指定来源的告警"""
        return [a.to_dict() for a in self.alerts if a.source == source]

    def reset(self):
        """重置所有状态"""
        self.alerts.clear()
        self.active_alerts.clear()
        self.alert_counts.clear()
        self.alert_by_source.clear()
