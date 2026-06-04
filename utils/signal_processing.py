"""
信号处理工具模块 — 用于 rPPG 心率提取。

提供巴特沃斯带通滤波、FFT 分析、滑动窗口平均、波峰检测、
心率变异性(HRV)计算、POS 脉搏信号提取以及心率估算等工具函数。
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq


# ---------------------------------------------------------------------------
# 巴特沃斯带通滤波器
# ---------------------------------------------------------------------------

def butter_bandpass(lowcut, highcut, fs, order=4):
    """设计巴特沃斯带通滤波器，返回 (b, a) 系数。

    Parameters
    ----------
    lowcut : float
        低截止频率 (Hz)
    highcut : float
        高截止频率 (Hz)
    fs : float
        采样率 (Hz)
    order : int
        滤波器阶数

    Returns
    -------
    b : ndarray
        分子系数
    a : ndarray
        分母系数
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """对一维信号应用巴特沃斯带通滤波。

    Parameters
    ----------
    data : ndarray
        输入一维信号
    lowcut : float
        低截止频率 (Hz)
    highcut : float
        高截止频率 (Hz)
    fs : float
        采样率 (Hz)
    order : int
        滤波器阶数

    Returns
    -------
    y : ndarray
        滤波后信号
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = signal.filtfilt(b, a, data)
    return y


# ---------------------------------------------------------------------------
# FFT 分析
# ---------------------------------------------------------------------------

def compute_fft(signal_data, fs):
    """计算实信号的单边 FFT 并返回频率轴与幅度谱。

    Parameters
    ----------
    signal_data : ndarray
        输入一维信号
    fs : float
        采样率 (Hz)

    Returns
    -------
    freqs : ndarray
        频率轴 (Hz)
    magnitude : ndarray
        对应频率的幅度
    """
    n = len(signal_data)
    magnitude = np.abs(rfft(signal_data)) / n
    freqs = rfftfreq(n, d=1.0 / fs)
    return freqs, magnitude


def find_dominant_frequency(signal_data, fs, freq_range=(0.7, 4.0)):
    """在指定频率范围内找到主频（幅度最大的频率分量）。

    Parameters
    ----------
    signal_data : ndarray
        输入一维信号
    fs : float
        采样率 (Hz)
    freq_range : tuple of (float, float)
        搜索的频率范围 (low, high)，单位 Hz

    Returns
    -------
    dominant_frequency : float
        主频 (Hz)
    """
    freqs, magnitude = compute_fft(signal_data, fs)

    low, high = freq_range
    mask = (freqs >= low) & (freqs <= high)
    freqs_in_range = freqs[mask]
    mag_in_range = magnitude[mask]

    if len(freqs_in_range) == 0:
        return 0.0

    idx = np.argmax(mag_in_range)
    return float(freqs_in_range[idx])


# ---------------------------------------------------------------------------
# 滑动窗口平均
# ---------------------------------------------------------------------------

def sliding_window_average(signal_data, window_size):
    """对一维信号做滑动窗口平均（移动平均）。

    Parameters
    ----------
    signal_data : ndarray
        输入一维信号
    window_size : int
        窗口大小（样本数）

    Returns
    -------
    avg : ndarray
        平滑后的信号，长度与输入相同。前 window_size-1 个值可能存在边缘效应。
    """
    if window_size < 1:
        raise ValueError("window_size 必须 >= 1")

    window = np.ones(window_size) / window_size
    avg = np.convolve(signal_data, window, mode='same')
    return avg


# ---------------------------------------------------------------------------
# 波峰检测
# ---------------------------------------------------------------------------

def peak_detection(signal_data, threshold=0.5, min_distance=5):
    """使用 scipy.signal.find_peaks 进行波峰检测。

    Parameters
    ----------
    signal_data : ndarray
        输入一维信号
    threshold : float
        波峰高度阈值（相对于信号极差的比值：0~1 之间）。
        仅保留高度 >= threshold*peak-to-peak 的峰。
    min_distance : int
        相邻波峰的最小间隔（样本数）

    Returns
    -------
    peak_indices : list of int
        检测到的波峰索引列表
    """
    peak_to_peak = np.ptp(signal_data)
    height = threshold * peak_to_peak if peak_to_peak > 0 else 0.0

    peaks, properties = signal.find_peaks(
        signal_data,
        height=height,
        distance=min_distance,
    )
    return peaks.tolist()


# ---------------------------------------------------------------------------
# 心率变异性 (HRV)
# ---------------------------------------------------------------------------

def compute_hrv(peak_indices, fs):
    """计算心率变异性 (HRV) — 基于相邻波峰间隔的 RMSSD。

    RMSSD = sqrt( mean( (IBI_i - IBI_{i-1})^2 ) )  [单位: ms]

    Parameters
    ----------
    peak_indices : list of int
        波峰索引列表
    fs : float
        采样率 (Hz)

    Returns
    -------
    hrv : float
        HRV 值 (ms)。如果波峰不足（< 3 个），返回 0.0。
    """
    if len(peak_indices) < 3:
        return 0.0

    # 相邻波峰间隔 — 转换为毫秒
    ibi = np.diff(peak_indices).astype(np.float64) * 1000.0 / fs

    # RMSSD
    squared_diff = (np.diff(ibi)) ** 2
    rmssd = np.sqrt(np.mean(squared_diff))
    return float(rmssd)


# ---------------------------------------------------------------------------
# POS 算法
# ---------------------------------------------------------------------------

def pos_algorithm(rgb_signals, fps):
    """POS (Plane-Orthogonal-to-Skin) 算法提取脉搏信号。

    步骤：
    1. 逐帧按通道均值归一化: C_n[t] = rgb[t] / mean(rgb[t])
    2. 投影矩阵 S = [[0, 1, -1], [-2, 1, 1]]
    3. X[t] = dot(S[0], C_n[t]), Y[t] = dot(S[1], C_n[t])
    4. alpha = std(X) / std(Y), h = X + alpha * Y
    5. 带通滤波 (0.7–4 Hz)

    Parameters
    ----------
    rgb_signals : ndarray
        (N_frames, 3) — 每帧的平均 RGB 值
    fps : float
        采样率 (帧率)

    Returns
    -------
    h_filtered : ndarray
        滤波后的脉搏信号 (长度 N_frames)
    """
    if rgb_signals.ndim != 2 or rgb_signals.shape[1] != 3:
        raise ValueError("rgb_signals 必须是 (N_frames, 3) 的二维数组")

    # 步骤 1: 逐帧按通道均值归一化
    temporal_mean = np.mean(rgb_signals, axis=1, keepdims=True)
    temporal_mean[temporal_mean == 0] = 1e-9  # 避免除零
    C_n = rgb_signals / temporal_mean

    # 步骤 2: 投影矩阵
    # S = [[ 0,  1, -1],
    #      [-2,  1,  1]]
    X = C_n[:, 1] - C_n[:, 2]          # dot([0, 1, -1], C_n[t])
    Y = -2.0 * C_n[:, 0] + C_n[:, 1] + C_n[:, 2]  # dot([-2, 1, 1], C_n[t])

    # 步骤 4: alpha = std(X) / std(Y)，合成 h
    std_X = np.std(X)
    std_Y = np.std(Y)
    if std_Y == 0:
        std_Y = 1e-9
    alpha = std_X / std_Y
    h = X + alpha * Y

    # 步骤 5: 带通滤波 (0.7 — 4 Hz)
    h_filtered = butter_bandpass_filter(h, 0.7, 4.0, fps, order=4)

    return h_filtered


# ---------------------------------------------------------------------------
# 心率估算
# ---------------------------------------------------------------------------

def estimate_heart_rate_from_signal(ppg_signal, fps, freq_range=(0.7, 4.0)):
    """从 PPG 脉搏信号估算心率 (BPM)。

    Parameters
    ----------
    ppg_signal : ndarray
        一维 PPG / 脉搏信号
    fps : float
        采样率 (Hz)
    freq_range : tuple of (float, float)
        有效心率频率范围 (Hz)

    Returns
    -------
    hr_bpm : float
        心率，单位 次/分钟 (BPM)
    """
    dominant_freq = find_dominant_frequency(ppg_signal, fps, freq_range)
    hr_bpm = dominant_freq * 60.0
    return hr_bpm
