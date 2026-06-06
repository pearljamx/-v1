"""
Web 蓝图包
==========
定义 Flask 蓝图 'web'，注册所有子路由模块。
各路由模块通过 import 时的副作用自动注册到蓝图。
"""

from flask import Blueprint

bp = Blueprint('web', __name__)

# 延迟导入路由模块（在蓝图定义后导入，避免循环依赖）
from . import routes_main
from . import routes_upload
from . import routes_detect
from . import routes_api
from . import routes_camera
from . import routes_demo
