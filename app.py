"""
Flask 应用主入口
===============
驾驶注意力检测系统的 Web 应用入口文件。
负责创建 Flask 应用实例、注册扩展与蓝图、配置文件服务路由及错误处理器。

运行方式:
    python app.py
    或通过 Flask CLI:  set FLASK_APP=app.py && flask run --port 5000
"""

import os

from flask import (
    Flask,
    send_from_directory,
    render_template,
    jsonify,
    request,
)

from config import BASE_DIR, MAX_CONTENT_LENGTH
from extensions import init_app


# ============================================================================
# 应用工厂函数
# ============================================================================


def create_app():
    """
    创建并配置 Flask 应用实例。

    遵循 Flask 应用工厂模式:
        1. 创建 Flask 实例
        2. 加载基础配置
        3. 初始化扩展 (CORS 等)
        4. 注册文件服务路由
        5. 注册 web 蓝图
        6. 注册错误处理器

    Returns
    -------
    flask.Flask
        已完全配置的 Flask 应用实例。
    """
    app = Flask(__name__)

    # -----------------------------------------------------------------------
    # 基础配置
    # -----------------------------------------------------------------------
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or os.urandom(24).hex()
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["CORS_ORIGINS"] = os.environ.get("CORS_ORIGINS", "*")

    # 上传与输出目录绝对路径
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

    # -----------------------------------------------------------------------
    # 确保必要的目录存在（应用启动时立即创建）
    # -----------------------------------------------------------------------
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # -----------------------------------------------------------------------
    # 初始化 Flask 扩展 (CORS 等)
    # -----------------------------------------------------------------------
    init_app(app)

    # -----------------------------------------------------------------------
    # 静态文件服务路由
    # 为 uploads 和 outputs 目录提供 HTTP 文件访问
    # -----------------------------------------------------------------------
    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename):
        """提供上传目录中文件的下载/预览服务。"""
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

    @app.route("/outputs/<path:filename>")
    def output_file(filename):
        """提供输出目录（检测结果）中文件的下载/预览服务。"""
        return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

    # -----------------------------------------------------------------------
    # 首页路由
    # -----------------------------------------------------------------------
    @app.route("/")
    def index():
        """系统首页 —— 文件上传与检测面板。"""
        return render_template("index.html")

    # -----------------------------------------------------------------------
    # 模型预加载 (应用启动时预热检测器，避免首次请求延迟)
    # -----------------------------------------------------------------------
    with app.app_context():
        try:
            from detectors.face_detector import FaceDetector
            _detector = FaceDetector()
            print(f"[启动] 面部检测器预热完成 (后端: {_detector.backend})")
        except Exception as e:
            print(f"[启动] 面部检测器预热跳过: {e}")

        # 尝试预加载 dlib 后端 (如果可用)
        try:
            import dlib
            import face_recognition
            dlib.get_frontal_face_detector()
            print("[启动] face_recognition/dlib 面部检测可用")
        except Exception:
            print("[启动] face_recognition/dlib 不可用，使用 MediaPipe/Yunet 后端")

        # 检查 YOLO 模型是否可用
        try:
            from config import YOLO_HANDHELD_MODEL, YOLO_DRIVER_STATE_MODEL, YOLO_POSE_MODEL
            if os.path.exists(YOLO_DRIVER_STATE_MODEL):
                print(f"[启动] YOLO 驾驶状态模型就绪: {YOLO_DRIVER_STATE_MODEL}")
            else:
                print("[启动] 提示: YOLO 驾驶状态模型未找到，可运行 python -m yolo.train_driver_state 训练")
            if os.path.exists(YOLO_HANDHELD_MODEL):
                print(f"[启动] YOLO 手持物检测模型就绪: {YOLO_HANDHELD_MODEL}")
            else:
                print(f"[启动] 提示: YOLO 手持物模型未找到，运行 python -m yolo.train_handheld 训练")
            if os.path.exists(YOLO_POSE_MODEL):
                print(f"[启动] YOLO 姿态估计模型就绪: {YOLO_POSE_MODEL}")
        except Exception as e:
            print(f"[启动] YOLO 模型检查跳过: {e}")

    # -----------------------------------------------------------------------
    # 注册蓝图（延迟导入以避免循环依赖）
    #
    # web 包的蓝图在 web/__init__.py 中定义，名为 bp。
    # 延迟导入确保 web 包内模块 (routes.py 等) 可以安全地引用
    # app.py 所在模块中的对象，而不会触发循环 import。
    # -----------------------------------------------------------------------
    from web import bp as web_bp

    app.register_blueprint(web_bp)

    # -----------------------------------------------------------------------
    # 错误处理器
    # -----------------------------------------------------------------------
    @app.errorhandler(400)
    def bad_request(error):
        """400 - 请求参数错误。"""
        if _is_api_request():
            return jsonify({"code": 400, "message": "请求参数有误"}), 400
        return render_template("index.html"), 400

    @app.errorhandler(404)
    def not_found(error):
        """404 - 资源未找到。"""
        if _is_api_request():
            return jsonify({"code": 404, "message": "请求的资源不存在"}), 404
        return render_template("index.html"), 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        """413 - 上传文件大小超过限制 (500 MB)。"""
        return jsonify({
            "code": 413,
            "message": "上传文件大小超过限制 (最大 500 MB)",
        }), 413

    @app.errorhandler(500)
    def internal_server_error(error):
        """500 - 服务器内部错误。"""
        if _is_api_request():
            return jsonify({"code": 500, "message": "服务器内部错误，请稍后重试"}), 500
        return render_template("index.html"), 500

    def _is_api_request():
        """判断当前请求是否为 API 请求（路径以 /api/ 开头）。"""
        return request.path.startswith("/api/")

    return app


# ============================================================================
# 模块级应用实例
# 供 WSGI 服务器 (gunicorn / waitress) 直接使用:  from app import app
# ============================================================================
app = create_app()


# ============================================================================
# 开发模式直接运行入口
# ============================================================================
if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    print(f"网站地址: http://{host}:{port}")
    if host == "0.0.0.0":
        print(f"本机访问: http://127.0.0.1:{port}")
    app.run(debug=debug, host=host, port=port)
