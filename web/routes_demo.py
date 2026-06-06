"""
答辩演示辅助路由
================
提供固定样例清单，帮助无摄像头环境下走图像/视频上传演示路径。
"""

from pathlib import Path

from flask import abort, jsonify, send_from_directory

from config import BASE_DIR
from . import bp


SAMPLE_IMAGE_GLOBS = [
    "dataset/val/images/*.jpg",
    "dataset/test/images/*.jpg",
    "dataset/train/images/*.jpg",
]
MAX_SAMPLES = 6


def _collect_sample_images():
    samples = []
    root = Path(BASE_DIR)
    for pattern in SAMPLE_IMAGE_GLOBS:
        for path in sorted(root.glob(pattern)):
            rel = path.relative_to(root).as_posix()
            samples.append({
                "type": "image",
                "name": path.name,
                "path": rel,
                "url": f"/demo/sample/{rel}",
            })
            if len(samples) >= MAX_SAMPLES:
                return samples
    return samples


@bp.route("/api/demo/samples")
def list_demo_samples():
    """返回固定答辩样例清单。"""
    return jsonify({
        "samples": _collect_sample_images(),
        "upload_url": "/#upload-section",
        "camera_url": "/camera",
        "note": "无摄像头时可下载样例后在首页上传，或直接选择本地 dataset/val/images 示例图。",
    })


@bp.route("/demo/sample/<path:relative_path>")
def demo_sample_file(relative_path):
    """安全提供 dataset 下的固定演示样例文件。"""
    root = Path(BASE_DIR).resolve()
    target = (root / relative_path).resolve()
    allowed_roots = [
        (root / "dataset" / split / "images").resolve()
        for split in ("train", "val", "test")
    ]
    if not any(str(target).startswith(str(allowed) + "/") for allowed in allowed_roots):
        abort(404)
    if target.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".mp4", ".avi", ".mov", ".webm"}:
        abort(404)
    if not target.exists():
        abort(404)
    return send_from_directory(str(target.parent), target.name)
