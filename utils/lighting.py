"""
环境光照自适应模块
==================
根据图像亮度动态调整检测阈值，提高不同光照条件下的检测鲁棒性。
"""

import cv2
import numpy as np


def compute_brightness(frame_bgr):
    """
    计算图像平均亮度 (0-255)

    参数:
        frame_bgr: BGR格式的numpy图像数组

    返回:
        float: 平均亮度值，范围 0-255
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def get_lighting_level(brightness):
    """
    根据平均亮度返回光照等级

    参数:
        brightness: 平均亮度值 (0-255)

    返回:
        str: 光照等级，可选值: 'dark', 'dim', 'normal', 'bright'
    """
    if brightness < 50:
        return 'dark'
    elif brightness < 100:
        return 'dim'
    elif brightness < 180:
        return 'normal'
    else:
        return 'bright'


def get_adaptive_thresholds(brightness, base_ear=0.2, base_mar=0.5):
    """
    根据光照等级返回自适应阈值

    在暗光环境下适当放宽阈值，减少因图像噪声导致的误检；
    在正常和明亮环境下使用基准阈值。

    参数:
        brightness: 平均亮度值 (0-255)
        base_ear: 基准EAR阈值 (默认 0.2)
        base_mar: 基准MAR阈值 (默认 0.5)

    返回:
        dict: {
            'ear_threshold': float,      # 自适应EAR阈值
            'mar_threshold': float,      # 自适应MAR阈值
            'lighting_level': str,       # 光照等级
            'brightness': float,         # 原始平均亮度值
        }
    """
    level = get_lighting_level(brightness)

    if level == 'dark':
        # 暗光环境: 阈值大幅放宽，减少噪声误检
        ear_offset = 0.05
        mar_offset = 0.05
    elif level == 'dim':
        # 昏暗环境: 阈值适度放宽
        ear_offset = 0.03
        mar_offset = 0.03
    else:
        # 正常/明亮环境: 使用基准阈值
        ear_offset = 0.0
        mar_offset = 0.0

    return {
        'ear_threshold': round(base_ear + ear_offset, 4),
        'mar_threshold': round(base_mar + mar_offset, 4),
        'lighting_level': level,
        'brightness': round(brightness, 2),
    }
