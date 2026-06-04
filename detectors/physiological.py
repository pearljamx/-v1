"""
生理信号监测模块 — rPPG 心率 + 血压趋势估计
============================================

基于远程光电容积描记法 (remote Photoplethysmography, rPPG) 从面部视频中
提取脉搏信号，实时估算心率 (HR)、心率变异性 (HRV) 以及血压 (BP) 趋势。

核心流程：
  1. 从每帧面部 ROI 提取空间平均 RGB 三通道信号
  2. 使用 POS (Plane-Orthogonal-to-Skin) 算法消除运动/光照伪影，得到 PPG 波形
  3. FFT 主频分析 → 心率 BPM
  4. 波峰检测 → 心搏间期 → 心率变异性 (RMSSD)
  5. PPG 形态学特征 → LSTM 深度学习模型 → 血压趋势预测
     （线性经验模型作为后备）

参考文献：
  - Wang et al. "Algorithmic Principles of Remote PPG." IEEE TBME, 2017.
  - De Haan & Jeanne. "Robust Pulse Rate From Chrominance-Based rPPG."
    IEEE TBME, 2013.
  - Xing et al. "Blood Pressure Assessment with Differential
    Pulse Transit Time …" Frontiers in Physiology, 2021.
"""

import cv2
import numpy as np
import logging
import os
from collections import deque
from utils.signal_processing import (
    pos_algorithm, estimate_heart_rate_from_signal,
    butter_bandpass_filter, compute_fft, find_dominant_frequency,
    peak_detection, compute_hrv
)
from config import PPG_WINDOW, PPG_BANDPASS_LOW, PPG_BANDPASS_HIGH, FPS_TARGET

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 血压趋势估计 — 经验线性模型系数
# 基于文献中 PPG 波形形态与逐博血压关系的经验值（无创、无校准场景）。
# 参考：Xing et al. (2021), Elgendi et al. (2019)
# 注意：此为群体平均趋势系数，个体差异较大，仅用于趋势监测（非临床诊断）。
# ---------------------------------------------------------------------------

# 收缩压:  BP_sys  = A1 * amplitude_norm + A2 * rise_time  + B1
BP_SYS_A1 = 85.0    # 归一化幅度系数 (mmHg / unit)
BP_SYS_A2 = -12.0   # 上升时间系数  (mmHg / s)，上升越慢 → 血管僵硬度高 → SBP 偏低趋势
BP_SYS_B1 = 95.0    # 基线常数 (mmHg)

# 舒张压:  BP_dia  = C1 * amplitude_norm + C2 * fall_time  + B2
BP_DIA_C1 = 42.0    # 归一化幅度系数 (mmHg / unit)
BP_DIA_C2 = 8.0     # 下降时间系数  (mmHg / s)，下降越慢 → 外周阻力大 → DBP 偏高趋势
BP_DIA_B2 = 62.0    # 基线常数 (mmHg)

# 血压输出限幅（防止异常值）
BP_SYS_MIN, BP_SYS_MAX = 70.0, 200.0
BP_DIA_MIN, BP_DIA_MAX = 40.0, 130.0


class rPPGMonitor:
    """远程光电容积描记法 (rPPG) 生理信号监测器。

    从面部 ROI 视频序列中实时提取 PPG 信号，估算心率和血压趋势。
    所有计算基于滑动窗口内的历史帧，适合在线逐帧处理。

    Parameters
    ----------
    fps : float
        视频帧率 (Hz)，默认使用 config.FPS_TARGET。
    window_size : int
        RGB 信号滑动窗口长度（帧数），默认使用 config.PPG_WINDOW。
    """

    def __init__(self, fps=FPS_TARGET, window_size=PPG_WINDOW):
        self.fps = float(fps)
        self.window_size = int(window_size)

        # ---- RGB 信号缓冲区 ----
        # 每个元素为 [r, g, b] 三通道空间均值
        self.rgb_signals = deque(maxlen=window_size)
        # 时间戳缓冲区（秒）
        self.timestamps = deque(maxlen=window_size)

        # ---- 计算结果 ----
        self.heart_rate = None          # 心率 (BPM)
        self.hrv = None                 # 心率变异性 RMSSD (ms)
        self.bp_systolic = None         # 收缩压趋势 (mmHg)
        self.bp_diastolic = None        # 舒张压趋势 (mmHg)
        self.signal_quality = 0.0       # 信号质量 (0–1)

        # ---- PPG 信号缓存 ----
        # 保存最近一次计算出的完整 PPG 脉搏波形，供外部可视化或离线分析
        self.ppg_signal_history = deque(maxlen=window_size)

        # ---- PPG 形态特征缓存（跨帧保持，用于血压趋势平滑） ----
        self._bp_sys_history = deque(maxlen=30)
        self._bp_dia_history = deque(maxlen=30)

        # ---- LSTM 血压预测器 (延迟初始化) ----
        self._lstm_predictor = None

    # ------------------------------------------------------------------
    # 数据输入
    # ------------------------------------------------------------------

    def add_frame(self, face_roi, timestamp=None):
        """添加一帧的面部 ROI 并提取 RGB 均值。

        Parameters
        ----------
        face_roi : np.ndarray
            BGR 格式的面部（额头）区域图像。
        timestamp : float or None
            帧时间戳（秒）。若为 None，则根据 fps 自动递增。
        """
        rgb_mean = self._extract_rgb_mean(face_roi)

        # BGR -> RGB 转换：cv2.mean 返回顺序就是 BGR，需要反转
        # 因为 _extract_rgb_mean 内部已做转换，这里直接使用
        self.rgb_signals.append(rgb_mean)

        if timestamp is not None:
            self.timestamps.append(timestamp)
        else:
            # 自动生成时间戳
            if self.timestamps:
                self.timestamps.append(self.timestamps[-1] + 1.0 / self.fps)
            else:
                self.timestamps.append(0.0)

    def _extract_rgb_mean(self, roi):
        """提取 ROI 的空间平均 RGB 值。

        Parameters
        ----------
        roi : np.ndarray
            BGR 格式的图像区域。

        Returns
        -------
        list
            [r, g, b] 三个通道的均值。
        """
        if roi is None or roi.size == 0:
            return [0.0, 0.0, 0.0]

        # cv2.mean 返回 (B, G, R, A)，转为 [R, G, B]
        b, g, r, _ = cv2.mean(roi)
        return [r, g, b]

    # ------------------------------------------------------------------
    # 心率计算
    # ------------------------------------------------------------------

    def compute_heart_rate(self):
        """计算心率 (BPM) 和心率变异性 (HRV)。

        步骤：
        1. 将 RGB 信号缓冲区转为 numpy 数组
        2. 使用 POS 算法提取脉搏信号
        3. FFT 主频 → 心率 (BPM)
        4. 波峰检测 → 心搏间期 → HRV (RMSSD)
        5. 计算信号质量

        Returns
        -------
        float or None
            心率 (BPM)。若数据不足（少于最小帧数），返回 None。
        """
        min_frames = int(self.fps * 3.0)  # 至少 3 秒数据
        if len(self.rgb_signals) < max(min_frames, 30):
            return None

        # ---- 步骤 1: 构建 RGB 信号矩阵 ----
        rgb_array = np.array(self.rgb_signals, dtype=np.float64)

        # ---- 步骤 2: POS 算法提取 PPG 信号 ----
        try:
            ppg_signal = pos_algorithm(rgb_array, self.fps)
        except ValueError:
            return None

        if ppg_signal is None or len(ppg_signal) == 0:
            return None

        # ---- 步骤 3: 缓存 PPG 信号 ----
        self.ppg_signal_history.clear()
        for val in ppg_signal:
            self.ppg_signal_history.append(float(val))

        # ---- 步骤 4: 信号质量 ----
        self.signal_quality = self._compute_signal_quality(ppg_signal)

        # ---- 步骤 5: FFT 主频 → 心率 ----
        try:
            hr_bpm = estimate_heart_rate_from_signal(
                ppg_signal, self.fps,
                freq_range=(PPG_BANDPASS_LOW, PPG_BANDPASS_HIGH)
            )
        except Exception:
            return None

        if hr_bpm <= 0:
            self.heart_rate = None
            return None

        # 合理性检查：心率应在合理范围内
        if hr_bpm < 30 or hr_bpm > 220:
            self.heart_rate = None
            return None

        self.heart_rate = float(hr_bpm)

        # ---- 步骤 6: 波峰检测 → HRV ----
        min_distance = int(self.fps * 0.4)  # 最小波峰间隔 ≈ 0.4 s (150 BPM 上限)
        threshold = 0.3
        peak_indices = peak_detection(ppg_signal, threshold=threshold,
                                      min_distance=min_distance)

        if len(peak_indices) >= 3:
            self.hrv = compute_hrv(peak_indices, self.fps)
        else:
            self.hrv = None

        return self.heart_rate

    def _compute_signal_quality(self, ppg_signal):
        """基于信噪比 (SNR) 评估 PPG 信号质量。

        在频域中计算 PPG 通带 (0.7–4 Hz) 内功率与带外功率之比，
        映射为 0–1 的质量分数。

        Parameters
        ----------
        ppg_signal : np.ndarray
            一维 PPG 脉搏信号。

        Returns
        -------
        float
            信号质量分数 (0.0–1.0)。
        """
        if ppg_signal is None or len(ppg_signal) < 30:
            return 0.0

        freqs, magnitude = compute_fft(ppg_signal, self.fps)

        # 通带内功率
        band_mask = (freqs >= PPG_BANDPASS_LOW) & (freqs <= PPG_BANDPASS_HIGH)
        power_in_band = np.sum(magnitude[band_mask] ** 2)

        # 总功率 (DC 分量除外，freq > 0.1 Hz)
        ac_mask = freqs >= 0.1
        power_total = np.sum(magnitude[ac_mask] ** 2)

        if power_total < 1e-12:
            return 0.0

        snr_ratio = power_in_band / power_total

        # 映射到 0–1：SNR 超过 0.7 视为高质量
        quality = min(snr_ratio / 0.7, 1.0)
        return float(quality)

    # ------------------------------------------------------------------
    # 血压趋势估计
    # ------------------------------------------------------------------

    def estimate_blood_pressure(self):
        """使用 LSTM 模型预测血压趋势，线性模型作为后备。

        优先使用 LSTM 深度学习模型进行血压预测：
        1. 从 PPG 信号提取单帧形态学特征
        2. 将特征送入 LSTM 滑动窗口缓冲区
        3. 缓冲区满 30 帧时，使用 LSTM 模型前向推理
        4. 若 LSTM 不可用或预测失败，回退到经验线性模型

        Returns
        -------
        dict
            包含以下键：
            - 'sbp': float | None      收缩压趋势 (mmHg)
            - 'dbp': float | None      舒张压趋势 (mmHg)
            - 'method': str            方法标识 ('lstm' or 'linear')
            - 'confidence': float      置信度 (0.3–1.0)
            - 'ready': bool            是否已有足够数据进行预测
        """
        # ---- 信号数据检查 ----
        if len(self.ppg_signal_history) < int(self.fps * 3.0):
            return {'sbp': None, 'dbp': None, 'method': 'lstm', 'ready': False}

        # ---- 提取 PPG 形态学特征 ----
        features = self._extract_ppg_features()
        if features is None:
            return {'sbp': None, 'dbp': None, 'method': 'lstm', 'ready': False}

        # ---- 尝试 LSTM 预测 ----
        lstm_result = self._try_lstm_predict(features)
        if lstm_result is not None:
            self.bp_systolic = lstm_result['sbp']
            self.bp_diastolic = lstm_result['dbp']
            self._bp_sys_history.append(self.bp_systolic)
            self._bp_dia_history.append(self.bp_diastolic)
            return {
                'sbp': self.bp_systolic,
                'dbp': self.bp_diastolic,
                'method': 'lstm',
                'confidence': lstm_result['confidence'],
                'ready': True,
            }

        # ---- 回退到线性模型 ----
        bp_sys, bp_dia = self._estimate_bp_linear(features)
        self.bp_systolic = bp_sys
        self.bp_diastolic = bp_dia
        return {
            'sbp': bp_sys,
            'dbp': bp_dia,
            'method': 'linear',
            'confidence': self.signal_quality,
            'ready': bp_sys is not None,
        }

    def _extract_ppg_features(self):
        """从 PPG 波形信号中提取单帧形态学特征。

        基于波峰/波谷检测提取逐博特征，取中位数作为当前帧的代表值。

        Returns
        -------
        dict or None
            包含 amplitude, rise_time, fall_time, heart_rate, hrv 的字典。
            信号质量不足时返回 None。
        """
        if len(self.ppg_signal_history) < int(self.fps * 3.0):
            return None

        ppg = np.array(self.ppg_signal_history, dtype=np.float64)

        # ---- 波峰 & 波谷检测 ----
        min_distance = int(self.fps * 0.4)
        peak_indices = peak_detection(ppg, threshold=0.3,
                                      min_distance=min_distance)
        trough_indices = peak_detection(-ppg, threshold=0.3,
                                        min_distance=min_distance)

        if len(peak_indices) < 2 or len(trough_indices) < 2:
            return None

        # ---- 提取逐博形态学特征 ----
        amplitudes = []
        rise_times = []
        fall_times = []

        for pk in peak_indices:
            prev_troughs = [t for t in trough_indices if t < pk]
            if not prev_troughs:
                continue
            prev_t = max(prev_troughs)

            next_troughs = [t for t in trough_indices if t > pk]
            if not next_troughs:
                continue
            next_t = min(next_troughs)

            sig_std = np.std(ppg)
            if sig_std < 1e-9:
                continue
            amp = (ppg[pk] - ppg[prev_t]) / sig_std
            if amp <= 0:
                continue

            rise_t = (pk - prev_t) / self.fps
            fall_t = (next_t - pk) / self.fps

            amplitudes.append(amp)
            rise_times.append(rise_t)
            fall_times.append(fall_t)

        if len(amplitudes) < 1:
            return None

        # ---- 取中位数 ----
        return {
            'amplitude': float(np.median(amplitudes)),
            'rise_time': float(np.median(rise_times)),
            'fall_time': float(np.median(fall_times)),
            'heart_rate': self.heart_rate or 75.0,
            'hrv': self.hrv or 30.0,
        }

    def _estimate_bp_linear(self, features):
        """使用经验线性模型估计血压（后备方法）。

        基于 PPG 波形形态特征与血压的经验关系：
          BP_sys = A1 * amplitude + A2 * rise_time + B1
          BP_dia = C1 * amplitude + C2 * fall_time + B2

        Parameters
        ----------
        features : dict
            _extract_ppg_features() 返回的特征字典。

        Returns
        -------
        tuple of (float or None, float or None)
            (systolic, diastolic) 血压趋势值 (mmHg)。
        """
        amp = features['amplitude']
        rise = features['rise_time']
        fall = features['fall_time']

        # ---- 线性模型 ----
        sys_raw = BP_SYS_A1 * amp + BP_SYS_A2 * rise + BP_SYS_B1
        dia_raw = BP_DIA_C1 * amp + BP_DIA_C2 * fall + BP_DIA_B2

        # 限幅
        sys_raw = np.clip(sys_raw, BP_SYS_MIN, BP_SYS_MAX)
        dia_raw = np.clip(dia_raw, BP_DIA_MIN, BP_DIA_MAX)

        # 确保收缩压 > 舒张压
        if sys_raw <= dia_raw:
            sys_raw = dia_raw + 25.0

        # ---- 时序平滑 ----
        self._bp_sys_history.append(sys_raw)
        self._bp_dia_history.append(dia_raw)

        bp_sys = float(np.mean(self._bp_sys_history))
        bp_dia = float(np.mean(self._bp_dia_history))

        # 再次限幅
        bp_sys = float(np.clip(bp_sys, BP_SYS_MIN, BP_SYS_MAX))
        bp_dia = float(np.clip(bp_dia, BP_DIA_MIN, BP_DIA_MAX))

        return (bp_sys, bp_dia)

    def _try_lstm_predict(self, features):
        """尝试使用 LSTM 模型进行血压预测。

        延迟初始化 LSTM 预测器，将当前帧特征送入滑动窗口，
        缓冲区满 30 帧后进行前向推理。

        Parameters
        ----------
        features : dict
            _extract_ppg_features() 返回的特征字典。

        Returns
        -------
        dict or None
            LSTM 预测结果 {'sbp', 'dbp', 'confidence'}，
            若 LSTM 未就绪或预测失败返回 None。
        """
        try:
            # ---- 延迟初始化 LSTM 预测器 ----
            if self._lstm_predictor is None:
                from models.lstm_bp import BPLSTMPredictor
                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    'models_data', 'bp_lstm.pt'
                )
                if os.path.exists(model_path):
                    self._lstm_predictor = BPLSTMPredictor(model_path=model_path)
                else:
                    self._lstm_predictor = BPLSTMPredictor(model_path=None)
                    logger.warning(
                        "LSTM model file not found at %s, "
                        "using heuristic weights (BP ≈ 120/80).", model_path
                    )

            # ---- 添加特征到滑动窗口 ----
            self._lstm_predictor.add_frame(features)

            # ---- 检查是否足以预测 ----
            if not self._lstm_predictor.is_ready():
                return None

            # ---- 前向推理 ----
            result = self._lstm_predictor.predict()
            return result

        except Exception as e:
            logger.warning("LSTM BP prediction failed, using linear fallback: %s", e)
            return None

    # ------------------------------------------------------------------
    # 结果输出 & 重置
    # ------------------------------------------------------------------

    def get_results(self):
        """获取所有生理检测结果。

        Returns
        -------
        dict
            包含以下键的字典：
            - 'heart_rate'       : float | None  心率 (BPM)
            - 'hrv'              : float | None  心率变异性 RMSSD (ms)
            - 'bp_systolic'      : float | None  收缩压趋势 (mmHg)
            - 'bp_diastolic'     : float | None  舒张压趋势 (mmHg)
            - 'signal_quality'   : float         信号质量 (0–1)
            - 'ppg_signal'       : list[float]   最近的 PPG 波形数据
        """
        return {
            'heart_rate': self.heart_rate,
            'hrv': self.hrv,
            'bp_systolic': self.bp_systolic,
            'bp_diastolic': self.bp_diastolic,
            'signal_quality': self.signal_quality,
            'ppg_signal': list(self.ppg_signal_history),
        }

    def reset(self):
        """重置所有内部缓冲区与计算结果。"""
        self.rgb_signals.clear()
        self.timestamps.clear()
        self.ppg_signal_history.clear()
        self._bp_sys_history.clear()
        self._bp_dia_history.clear()

        if self._lstm_predictor is not None:
            self._lstm_predictor.reset()

        self.heart_rate = None
        self.hrv = None
        self.bp_systolic = None
        self.bp_diastolic = None
        self.signal_quality = 0.0
