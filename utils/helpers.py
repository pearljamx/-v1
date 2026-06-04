"""
通用工具函数模块

包含性能计时、文件/目录操作、图像编码、JSON读写、
视频信息获取等通用辅助函数。
"""

import os
import json
import uuid
import time
import base64

import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# 性能计时上下文管理器
# ---------------------------------------------------------------------------

class Timer:
    """性能计时上下文管理器，支持 __enter__/__exit__ 和 elapsed 属性。

    用法:
        with Timer() as t:
            do_something()
        print(f"耗时: {t.elapsed:.2f}s")

        t = Timer()
        t.start()
        ...
        t.stop()
        print(t.elapsed)
    """

    def __init__(self):
        self._start_time: float | None = None
        self._end_time: float | None = None
        self.elapsed: float = 0.0

    def start(self):
        """手动启动计时。"""
        self._start_time = time.perf_counter()
        self._end_time = None
        self.elapsed = 0.0

    def stop(self):
        """手动停止计时并记录耗时。"""
        if self._start_time is not None:
            self._end_time = time.perf_counter()
            self.elapsed = self._end_time - self._start_time
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False  # 不抑制异常


# ---------------------------------------------------------------------------
# 文件 / 目录工具
# ---------------------------------------------------------------------------

def ensure_dir(path):
    """确保目录存在，不存在则创建。

    Args:
        path: 目录路径（字符串或 Path 对象）
    """
    Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 文件扩展名校验
# ---------------------------------------------------------------------------

# 默认允许的图像扩展名
DEFAULT_IMAGE_EXTENSIONS = {
    'jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp', 'gif',
}

# 默认允许的视频扩展名
DEFAULT_VIDEO_EXTENSIONS = {
    'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm',
}

# 综合默认允许列表
DEFAULT_ALLOWED_EXTENSIONS = DEFAULT_IMAGE_EXTENSIONS | DEFAULT_VIDEO_EXTENSIONS


def allowed_file(filename, extensions=None):
    """校验文件扩展名是否允许，默认支持图像和视频格式。

    Args:
        filename: 文件名或路径
        extensions: 允许的扩展名集合（不含点），为 None 时使用默认集合

    Returns:
        bool: 是否允许
    """
    if extensions is None:
        extensions = DEFAULT_ALLOWED_EXTENSIONS

    # 提取扩展名并转小写
    ext = Path(filename).suffix.lstrip('.').lower()
    return ext in extensions


# ---------------------------------------------------------------------------
# 任务 ID 生成
# ---------------------------------------------------------------------------

def generate_task_id():
    """生成唯一任务 ID（UUID4 的 hex 字符串）。

    Returns:
        str: 32 位十六进制字符串
    """
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# 图像编码 / 解码
# ---------------------------------------------------------------------------

def frame_to_base64(frame):
    """将 OpenCV 帧（BGR）编码为 JPEG 的 Base64 字符串。

    Args:
        frame: numpy ndarray, BGR 格式图像

    Returns:
        str: Base64 编码的 JPEG 图像字符串
    """
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        raise ValueError("图像编码失败：cv2.imencode 返回 False")
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_image(b64_string):
    """将 Base64 字符串解码为 OpenCV 图像（numpy array，BGR 格式）。

    Args:
        b64_string: Base64 编码的图像字符串

    Returns:
        numpy ndarray, or None 如果解码失败
    """
    try:
        img_data = base64.b64decode(b64_string)
        np_arr = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


# ---------------------------------------------------------------------------
# JSON 文件读写
# ---------------------------------------------------------------------------

def safe_json_load(filepath, default=None):
    """安全读取 JSON 文件，出错返回 default。

    Args:
        filepath: JSON 文件路径
        default: 读取失败时的返回值

    Returns:
        解析后的 Python 对象，或 default
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def safe_json_dump(data, filepath):
    """安全写入 JSON 文件，自动创建目录。

    Args:
        data: 要序列化的 Python 对象
        filepath: 目标 JSON 文件路径
    """
    ensure_dir(Path(filepath).parent)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# 时间格式化
# ---------------------------------------------------------------------------

def format_timestamp(seconds):
    """秒数格式化为 mm:ss 字符串。

    Args:
        seconds: 秒数（int 或 float）

    Returns:
        str: 格式化后的 "mm:ss" 字符串，例如 "03:45"
    """
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# 视频信息
# ---------------------------------------------------------------------------

def get_video_info(video_path):
    """获取视频信息。

    Args:
        video_path: 视频文件路径

    Returns:
        dict: {
            'fps': float,
            'frame_count': int,
            'width': int,
            'height': int,
            'duration': float,   # 秒
        }
        如果无法打开视频则返回空字典。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps > 0 and frame_count > 0:
        duration = frame_count / fps
    else:
        duration = 0.0

    cap.release()

    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration,
    }


# ---------------------------------------------------------------------------
# 图像缩放
# ---------------------------------------------------------------------------

def resize_frame(frame, max_width=640):
    """按最大宽度等比缩放帧。

    Args:
        frame: numpy ndarray, BGR 格式图像
        max_width: 最大宽度（像素），默认 640

    Returns:
        numpy ndarray: 缩放后的图像。若原图宽度已 <= max_width 则直接返回原图。
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    ratio = max_width / w
    new_w = max_width
    new_h = int(h * ratio)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
