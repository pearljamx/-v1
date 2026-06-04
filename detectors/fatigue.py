"""
疲劳驾驶检测模块
===============
基于眼部纵横比（EAR）和嘴部纵横比（MAR）实现对驾驶员闭眼、眨眼频率异常、
打哈欠等疲劳状态的实时检测与告警。

检测逻辑：
  - 闭眼告警：EAR < EAR_THRESHOLD 持续超过 EAR_DURATION 秒
  - 眨眼频率告警：最近60秒内眨眼次数 < BLINK_RATE_LOW 次/分钟
  - 哈欠告警：MAR > MAR_THRESHOLD 持续超过 MAR_DURATION 秒

告警格式:
  {'type': 'eye_closure'|'yawn'|'low_blink_rate',
   'severity': 'danger'|'warning',
   'message': str,
   'timestamp': float}
"""

from collections import deque
import time
from typing import Optional
from config import *


class FatigueDetector:
    """
    疲劳驾驶检测器
    功能: 闭眼检测(EAR)、眨眼频率统计、哈欠检测(MAR)

    使用方法:
        detector = FatigueDetector()
        result = detector.update(ear_value, mar_value, timestamp)
        if result['alerts']:
            for alert in result['alerts']:
                print(alert)
    """

    def __init__(self):
        # EAR历史记录 (最近60帧)
        self.ear_history = deque(maxlen=60)
        # MAR历史记录 (最近60帧)
        self.mar_history = deque(maxlen=60)
        # 眨眼时间戳记录
        self.blink_timestamps = deque()
        # 闭眼开始时间
        self.eye_closure_start = None
        # 哈欠开始时间
        self.yawn_start = None
        # 帧时间戳历史
        self.frame_times = deque(maxlen=60)
        # EAR阈值用于检测眨眼(单帧低于此值视为眨眼开始)
        self.blink_ear_threshold = 0.15

        # ---- 眨眼检测状态机 ----
        # 状态: 'open' -> 'closing' -> 'closed' -> 'opening' -> 'open'
        self._blink_state = 'open'

        # ---- 当前帧状态标记 ----
        self.eye_closure_active = False
        self.yawn_active = False

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def update(self, ear_value: float, mar_value: float, timestamp: float) -> dict:
        """
        处理一帧数据
        参数:
          ear_value: 当前帧的EAR值
          mar_value: 当前帧的MAR值
          timestamp: 当前时间戳(秒)
        返回: {'alerts': [Alert dicts], 'ear': float, 'mar': float, 'blink_rate': float}
        """
        alerts = []

        # 记录当前帧数据到滑动窗口
        self.ear_history.append(ear_value)
        self.mar_history.append(mar_value)
        self.frame_times.append(timestamp)

        # ---- 闭眼检测 ----
        eye_closure_alert = self._detect_eye_closure(ear_value, timestamp)
        if eye_closure_alert:
            alerts.append(eye_closure_alert)

        # ---- 眨眼检测 ----
        self._detect_blink(ear_value, timestamp)

        # ---- 眨眼频率计算与告警 ----
        blink_rate = self._calculate_blink_rate()
        if len(self.blink_timestamps) >= 3 and blink_rate < BLINK_RATE_LOW:
            alerts.append({
                'type': 'low_blink_rate',
                'severity': 'warning',
                'message': f'眨眼频率过低: {blink_rate:.1f}次/分钟 (阈值: {BLINK_RATE_LOW}次/分钟)',
                'timestamp': timestamp,
            })

        # ---- 哈欠检测 ----
        yawn_alert = self._detect_yawn(mar_value, timestamp)
        if yawn_alert:
            alerts.append(yawn_alert)

        return {
            'alerts': alerts,
            'ear': ear_value,
            'mar': mar_value,
            'blink_rate': blink_rate,
        }

    def reset(self):
        """重置所有状态"""
        self.ear_history.clear()
        self.mar_history.clear()
        self.blink_timestamps.clear()
        self.frame_times.clear()

        self.eye_closure_start = None
        self.yawn_start = None
        self._blink_state = 'open'
        self.eye_closure_active = False
        self.yawn_active = False

    # ------------------------------------------------------------------
    # 内部检测方法
    # ------------------------------------------------------------------

    def _detect_eye_closure(self, ear_value, timestamp) -> Optional[dict]:
        """
        检测持续闭眼
        EAR < EAR_THRESHOLD 持续 > EAR_DURATION -> 告警
        返回闭眼事件字典或None
        """
        if ear_value < EAR_THRESHOLD:
            # 进入或维持闭眼状态
            self.eye_closure_active = True
            if self.eye_closure_start is None:
                self.eye_closure_start = timestamp
            else:
                duration = timestamp - self.eye_closure_start
                if duration >= EAR_DURATION:
                    return {
                        'type': 'eye_closure',
                        'severity': 'danger',
                        'message': f'连续闭眼超过{duration:.1f}秒 (阈值: {EAR_DURATION}秒)',
                        'timestamp': timestamp,
                    }
        else:
            # 眼睛睁开，退出闭眼状态
            self.eye_closure_active = False
            self.eye_closure_start = None

        return None

    def _detect_blink(self, ear_value, timestamp):
        """
        检测眨眼事件
        使用波谷检测: EAR从正常值降到<0.15再回升
        记录眨眼时间戳

        状态转移:
          'open'    -- EAR降到 blink_ear_threshold 以下 --> 'closing'
          'closing' -- EAR降到 blink_ear_threshold 以下（维持） --> 'closed'
          'closed'  -- EAR回升到 blink_ear_threshold 以上 --> 'opening'
          'opening' -- 回到 'open'，记录一次眨眼
        """
        if self._blink_state == 'open':
            if ear_value < self.blink_ear_threshold:
                self._blink_state = 'closing'

        elif self._blink_state == 'closing':
            if ear_value < self.blink_ear_threshold:
                self._blink_state = 'closed'
            else:
                # EAR回升，假阳性，回到open
                self._blink_state = 'open'

        elif self._blink_state == 'closed':
            if ear_value >= self.blink_ear_threshold:
                self._blink_state = 'opening'

        elif self._blink_state == 'opening':
            # EAR恢复到 blink_ear_threshold 以上 -> 眨眼完成
            self.blink_timestamps.append(timestamp)
            self._blink_state = 'open'

    def _calculate_blink_rate(self) -> float:
        """
        计算眨眼频率 (次/分钟)
        使用最近60秒内的眨眼次数来估算
        """
        if len(self.blink_timestamps) < 2:
            return 0.0

        now = time.time()
        lookback = 60.0

        # 统计最近60秒内的眨眼次数
        count = sum(1 for ts in self.blink_timestamps if now - ts <= lookback)

        if count < 1:
            return 0.0

        # 找到最老有效时间戳对应的时间跨度
        oldest = None
        for ts in self.blink_timestamps:
            if now - ts <= lookback:
                oldest = ts
            else:
                break  # 时间戳从旧到新排列，遇到不在窗口内的可以停止

        if oldest is None:
            return 0.0

        span = now - oldest
        if span <= 0.0:
            return 0.0

        # 外推到每分钟
        blink_rate = (count / span) * 60.0
        return blink_rate

    def _detect_yawn(self, mar_value, timestamp) -> Optional[dict]:
        """
        检测哈欠
        MAR > MAR_THRESHOLD 持续 > MAR_DURATION -> 告警
        """
        if mar_value > MAR_THRESHOLD:
            # 进入或维持张嘴/哈欠状态
            self.yawn_active = True
            if self.yawn_start is None:
                self.yawn_start = timestamp
            else:
                duration = timestamp - self.yawn_start
                if duration >= MAR_DURATION:
                    return {
                        'type': 'yawn',
                        'severity': 'warning',
                        'message': f'检测到哈欠: 持续{duration:.1f}秒 (阈值: {MAR_DURATION}秒)',
                        'timestamp': timestamp,
                    }
        else:
            # 嘴巴闭合，退出哈欠状态
            self.yawn_active = False
            self.yawn_start = None

        return None
