"""
SQLite 检测历史数据库模块
=======================
提供检测结果的持久化存储，支持单条记录增删查和分页历史查询，
以及按日期范围聚合统计。

使用 Python 内置 sqlite3 模块，无需额外依赖。
线程安全：所有公开方法通过 threading.Lock 保证。
"""

import os
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

from utils.helpers import ensure_dir

# ---------------------------------------------------------------------------
# 数据库文件路径
# ---------------------------------------------------------------------------
DB_DIR = os.path.join(Path(__file__).resolve().parent.parent, 'models_data')
DB_PATH = os.path.join(DB_DIR, 'detection_history.db')


# ---------------------------------------------------------------------------
# 数据库连接管理
# ---------------------------------------------------------------------------

def _get_connection():
    """获取数据库连接（check_same_thread=False 允许跨线程使用）。"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# 表创建
# ---------------------------------------------------------------------------

def _create_table(conn):
    """创建 detection_history 表（如果不存在）。"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT UNIQUE,
            timestamp TEXT,
            file_name TEXT,
            file_type TEXT,
            overall_risk TEXT,
            fatigue_score REAL,
            distraction_score REAL,
            heart_rate REAL,
            bp_systolic REAL,
            bp_diastolic REAL,
            alert_count INTEGER,
            alerts_json TEXT,
            summary_json TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# 数据库封装类
# ---------------------------------------------------------------------------

class DetectionHistoryDB:
    """
    检测历史数据库操作封装。

    线程安全：所有公有方法均受 threading.Lock 保护。
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._db_lock = threading.Lock()
        # 确保目录和表存在
        ensure_dir(DB_DIR)
        conn = _get_connection()
        try:
            _create_table(conn)
        finally:
            conn.close()

    # -------------------------------------------------------------------
    # 保存结果
    # -------------------------------------------------------------------

    def save_result(self, task_id, result_dict):
        """
        保存（插入或替换）一条检测结果。

        Parameters
        ----------
        task_id : str
            任务唯一标识。
        result_dict : dict
            包含以下可选字段的字典：
            timestamp, file_name, file_type, overall_risk,
            fatigue_score, distraction_score, heart_rate,
            bp_systolic, bp_diastolic, alerts, summary
        """
        with self._db_lock:
            conn = _get_connection()
            try:
                summary = result_dict.get('summary', {}) or {}
                alerts = result_dict.get('alerts', []) or []
                physiological = result_dict.get('physiological', {}) or {}

                conn.execute("""
                    INSERT OR REPLACE INTO detection_history
                        (task_id, timestamp, file_name, file_type,
                         overall_risk, fatigue_score, distraction_score,
                         heart_rate, bp_systolic, bp_diastolic,
                         alert_count, alerts_json, summary_json,
                         created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        COALESCE(
                            (SELECT created_at FROM detection_history WHERE task_id = ?),
                            datetime('now')
                        )
                    )
                """, (
                    task_id,
                    result_dict.get('timestamp', datetime.now().isoformat()),
                    result_dict.get('file_name', ''),
                    result_dict.get('file_type', 'image'),
                    summary.get('overall_risk', 'low'),
                    summary.get('fatigue_score', 100.0),
                    summary.get('distraction_score', 100.0),
                    physiological.get('heart_rate'),
                    physiological.get('bp_systolic'),
                    physiological.get('bp_diastolic'),
                    len(alerts),
                    json.dumps(alerts, ensure_ascii=False, default=str),
                    json.dumps(summary, ensure_ascii=False, default=str),
                    task_id,
                ))
                conn.commit()
                return True
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # -------------------------------------------------------------------
    # 获取单条记录
    # -------------------------------------------------------------------

    def get_result(self, task_id):
        """
        获取单条检测结果。

        Parameters
        ----------
        task_id : str
            任务唯一标识。

        Returns
        -------
        dict or None
        """
        with self._db_lock:
            conn = _get_connection()
            try:
                row = conn.execute(
                    "SELECT * FROM detection_history WHERE task_id = ?",
                    (task_id,)
                ).fetchone()

                if row is None:
                    return None
                return _row_to_dict(row)
            finally:
                conn.close()

    # -------------------------------------------------------------------
    # 分页历史查询
    # -------------------------------------------------------------------

    def get_history(self, limit=50, offset=0):
        """
        分页获取检测历史记录（按创建时间倒序）。

        Parameters
        ----------
        limit : int
            每页条数，默认 50。
        offset : int
            偏移量，默认 0。

        Returns
        -------
        tuple[list[dict], int]
            (记录列表, 总条数)
        """
        with self._db_lock:
            conn = _get_connection()
            try:
                total_row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM detection_history"
                ).fetchone()
                total = total_row['cnt'] if total_row else 0

                rows = conn.execute(
                    "SELECT * FROM detection_history "
                    "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (limit, offset)
                ).fetchall()

                return [_row_to_dict(r) for r in rows], total
            finally:
                conn.close()

    # -------------------------------------------------------------------
    # 聚合统计
    # -------------------------------------------------------------------

    def get_stats(self, start_date=None, end_date=None):
        """
        获取聚合统计信息。

        Parameters
        ----------
        start_date : str or None
            开始日期 (YYYY-MM-DD)。
        end_date : str or None
            结束日期 (YYYY-MM-DD)。

        Returns
        -------
        dict
            包含 total_tasks, avg_fatigue_score, avg_distraction_score,
            risk_distribution, avg_heart_rate, avg_alert_count 等字段。
        """
        with self._db_lock:
            conn = _get_connection()
            try:
                conditions = []
                params = []

                if start_date:
                    conditions.append("created_at >= ?")
                    params.append(start_date + " 00:00:00")
                if end_date:
                    conditions.append("created_at <= ?")
                    params.append(end_date + " 23:59:59")

                where_clause = ""
                if conditions:
                    where_clause = "WHERE " + " AND ".join(conditions)

                def _build_where(extra_conditions=None):
                    all_conditions = conditions + (extra_conditions or [])
                    if not all_conditions:
                        return "", list(params)
                    return "WHERE " + " AND ".join(all_conditions), list(params)

                # 总任务数
                total_row = conn.execute(
                    f"SELECT COUNT(*) AS cnt FROM detection_history {where_clause}",
                    params
                ).fetchone()
                total_tasks = total_row['cnt'] if total_row else 0

                # 平均评分
                avg_fatigue = 100.0
                avg_distraction = 100.0
                avg_heart_rate = None
                avg_alert_count = 0.0

                if total_tasks > 0:
                    row = conn.execute(
                        f"SELECT AVG(fatigue_score) AS af, "
                        f"AVG(distraction_score) AS ad, "
                        f"AVG(heart_rate) AS ahr, "
                        f"AVG(alert_count) AS aac "
                        f"FROM detection_history {where_clause}",
                        params
                    ).fetchone()
                    avg_fatigue = round(row['af'], 1) if row['af'] is not None else 100.0
                    avg_distraction = round(row['ad'], 1) if row['ad'] is not None else 100.0
                    avg_heart_rate = round(row['ahr'], 1) if row['ahr'] is not None else None
                    avg_alert_count = round(row['aac'], 1) if row['aac'] is not None else 0.0

                # 风险等级分布
                risk_rows = conn.execute(
                    f"SELECT overall_risk, COUNT(*) AS cnt "
                    f"FROM detection_history {where_clause} "
                    f"GROUP BY overall_risk",
                    params
                ).fetchall()

                risk_distribution = {'low': 0, 'medium': 0, 'high': 0}
                for r in risk_rows:
                    risk_distribution[r['overall_risk']] = r['cnt']

                # 平均生理数据
                physio_where_clause, physio_params = _build_where(["heart_rate IS NOT NULL"])
                physio_row = conn.execute(
                    f"SELECT AVG(heart_rate) AS avg_hr, "
                    f"AVG(bp_systolic) AS avg_sys, "
                    f"AVG(bp_diastolic) AS avg_dia "
                    f"FROM detection_history {physio_where_clause}",
                    physio_params
                ).fetchone()

                return {
                    'total_tasks': total_tasks,
                    'avg_fatigue_score': avg_fatigue,
                    'avg_distraction_score': avg_distraction,
                    'risk_distribution': risk_distribution,
                    'avg_heart_rate': avg_heart_rate,
                    'avg_alert_count': avg_alert_count,
                    'avg_bp_systolic': round(physio_row['avg_sys'], 1) if physio_row and physio_row['avg_sys'] is not None else None,
                    'avg_bp_diastolic': round(physio_row['avg_dia'], 1) if physio_row and physio_row['avg_dia'] is not None else None,
                    'start_date': start_date,
                    'end_date': end_date,
                }
            finally:
                conn.close()

    # -------------------------------------------------------------------
    # 删除单条记录
    # -------------------------------------------------------------------

    def delete_result(self, task_id):
        """
        删除指定任务的检测记录。

        Parameters
        ----------
        task_id : str
            任务唯一标识。

        Returns
        -------
        bool
            是否成功删除。
        """
        with self._db_lock:
            conn = _get_connection()
            try:
                cursor = conn.execute(
                    "DELETE FROM detection_history WHERE task_id = ?",
                    (task_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _row_to_dict(row):
    """将 sqlite3.Row 转换为普通字典。"""
    return {
        'id': row['id'],
        'task_id': row['task_id'],
        'timestamp': row['timestamp'],
        'file_name': row['file_name'],
        'file_type': row['file_type'],
        'overall_risk': row['overall_risk'],
        'fatigue_score': row['fatigue_score'],
        'distraction_score': row['distraction_score'],
        'heart_rate': row['heart_rate'],
        'bp_systolic': row['bp_systolic'],
        'bp_diastolic': row['bp_diastolic'],
        'alert_count': row['alert_count'],
        'alerts': json.loads(row['alerts_json']) if row['alerts_json'] else [],
        'summary': json.loads(row['summary_json']) if row['summary_json'] else {},
        'created_at': row['created_at'],
    }
