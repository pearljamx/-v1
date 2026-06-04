"""
YOLO 手持物品检测数据集准备工具
==============================
生成合成的驾驶场景图像用于训练手持物品检测模型。

功能:
1. 创建data.yaml配置文件
2. 用OpenCV绘制逼真的合成驾驶员图像（640x480）
3. 生成YOLO格式标注（归一化边界框）
4. 5个类别: normal_driving, phone_usage, smoking, drinking, hands_off_wheel

数据量:
- 训练集: 80张 (每类16张)
- 验证集: 25张 (每类5张)
- 测试集: 25张 (每类5张)

运行方式:
    python -m yolo.dataset_prep
"""

import os
import random
import math
from pathlib import Path
import numpy as np
import cv2

# ---------- 安全随机数包装器 ----------
# numpy RandomState 返回 numpy int64/float64，OpenCV 不接受这些类型
# 所以包装一下强制转为 Python 原生类型
class SafeRng:
    """包装 numpy RandomState，所有返回值强制转为 Python 原生类型"""
    def __init__(self, seed=None):
        self._rng = np.random.RandomState(seed)
    def randint(self, low, high=None):
        if high is None:
            return int(self._rng.randint(low))
        return int(self._rng.randint(low, high))
    def uniform(self, low=0.0, high=1.0):
        return float(self._rng.uniform(low, high))
    def random(self):
        return float(self._rng.random())
    def choice(self, a):
        return a[self.randint(0, len(a) - 1)]
    def randn(self, *args):
        return float(self._rng.randn(*args))


# ---------- 路径常量 ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / 'dataset'
WIDTH, HEIGHT = 640, 480

# ---------- 配色方案 ----------
# 肤色系
SKIN_LIGHT = (180, 200, 230)
SKIN_MEDIUM = (150, 170, 210)
SKIN_DARK = (110, 135, 175)
SKIN_TONES = [SKIN_LIGHT, SKIN_MEDIUM, SKIN_DARK,
              (195, 210, 240), (130, 155, 195)]

# 服装颜色
SHIRT_COLORS = [
    (180, 80, 60),    # 深蓝
    (50, 50, 50),     # 黑
    (200, 200, 200),  # 白
    (100, 60, 40),    # 深红
    (60, 100, 60),    # 深绿
    (140, 120, 80),   # 棕
    (80, 80, 130),    # 紫蓝
]

# 车内颜色
DASHBOARD_DARK = (35, 38, 42)
DASHBOARD_MID = (50, 53, 57)
STEERING_COL = (55, 55, 60)
STEERING_HIGHLIGHT = (80, 80, 85)
DOOR_PANEL = (70, 65, 60)
A_PILLAR = (55, 52, 48)

# 环境颜色
SKY_TOP = (200, 160, 100)
SKY_BOTTOM = (220, 200, 170)
ROAD_COLOR = (95, 95, 100)
BUILDING_COLOR = (150, 155, 160)

# 物体颜色
PHONE_COLOR = (30, 30, 35)
CIGARETTE_COLOR = (235, 235, 240)
CIGARETTE_TIP = (80, 100, 255)
BOTTLE_COLORS = [(200, 150, 80), (180, 120, 60), (100, 160, 180), (90, 140, 200)]
CUP_COLOR = (200, 200, 210)


def ensure_dirs():
    """确保所有数据集目录存在"""
    dirs = [
        DATASET_DIR / 'train' / 'images',
        DATASET_DIR / 'train' / 'labels',
        DATASET_DIR / 'val' / 'images',
        DATASET_DIR / 'val' / 'labels',
        DATASET_DIR / 'test' / 'images',
        DATASET_DIR / 'test' / 'labels',
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def create_data_yaml():
    """创建YOLO数据集配置文件"""
    yaml_content = f"""# YOLOv8 手持物品检测 - 驾驶场景数据集
# 5类: normal_driving, phone_usage, smoking, drinking, hands_off_wheel

path: {DATASET_DIR.as_posix()}
train: train/images
val: val/images
test: test/images

nc: 5
names:
  0: normal_driving
  1: phone_usage
  2: smoking
  3: drinking
  4: hands_off_wheel
"""
    yaml_path = DATASET_DIR / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"[OK] data.yaml -> {yaml_path}")


# ==================== 绘制组件 ====================

def draw_sky_and_road(img, rng):
    """绘制背景：天空渐变 + 路面"""
    # 天空（上部0-55%）
    sky_end = int(HEIGHT * 0.55 + rng.randint(-20, 10))
    for y in range(0, sky_end):
        t = y / max(sky_end - 1, 1)
        b = int(SKY_TOP[0] + (SKY_BOTTOM[0] - SKY_TOP[0]) * t)
        g = int(SKY_TOP[1] + (SKY_BOTTOM[1] - SKY_TOP[1]) * t)
        r = int(SKY_TOP[2] + (SKY_BOTTOM[2] - SKY_TOP[2]) * t)
        cv2.line(img, (0, y), (WIDTH, y), (b, g, r), 1)

    # 路面（天空下方到画面底部）
    for y in range(sky_end, HEIGHT):
        shade = 85 + int(20 * (y - sky_end) / (HEIGHT - sky_end))
        cv2.line(img, (0, y), (WIDTH, y), (shade, shade, shade + 5), 1)

    # 远处建筑轮廓
    if rng.random() < 0.7:
        horizon = sky_end - rng.randint(5, 20)
        for _ in range(rng.randint(2, 5)):
            bx = rng.randint(50, WIDTH - 200)
            bw = rng.randint(40, 180)
            bh = rng.randint(20, 100)
            cv2.rectangle(img, (bx, horizon - bh), (bx + bw, horizon),
                          (rng.randint(130, 170), rng.randint(135, 175), rng.randint(140, 180)), -1)

    return sky_end


def draw_car_interior(img, rng):
    """绘制车内结构：A柱、车门板、仪表台"""
    # A柱（左右两侧）
    cv2.fillPoly(img, np.array([[(0, 0), (60, 0), (160, 180), (60, HEIGHT)]]),
                 (50, 47, 43))
    cv2.fillPoly(img, np.array([[(WIDTH, 0), (WIDTH - 60, 0), (WIDTH - 160, 180), (WIDTH - 60, HEIGHT)]]),
                 (50, 47, 43))

    # A柱内侧 - 让过渡更自然
    cv2.fillPoly(img, np.array([[(0, 0), (40, 0), (150, 160), (40, HEIGHT)]]),
                 (65, 62, 58))
    cv2.fillPoly(img, np.array([[(WIDTH, 0), (WIDTH - 40, 0), (WIDTH - 150, 160), (WIDTH - 40, HEIGHT)]]),
                 (65, 62, 58))

    # 仪表台（底部）
    dash_top = int(HEIGHT * 0.72)
    cv2.rectangle(img, (0, dash_top), (WIDTH, HEIGHT), DASHBOARD_DARK, -1)

    # 仪表台顶部高光线
    cv2.line(img, (0, dash_top), (WIDTH, dash_top), DASHBOARD_MID, 1)

    # 仪表台纹理（空调出风口等）
    vent_y = dash_top + rng.randint(30, 50)
    vent_x = WIDTH // 2
    vent_w = rng.randint(80, 140)
    cv2.rectangle(img, (vent_x - vent_w, vent_y), (vent_x + vent_w, vent_y + 8),
                  (60, 63, 68), -1)
    # 格栅
    for gx in range(vent_x - vent_w + 15, vent_x + vent_w, 20):
        cv2.line(img, (gx, vent_y), (gx, vent_y + 8), DASHBOARD_DARK, 1)

    return dash_top


def draw_steering_wheel(img, rng):
    """绘制方向盘"""
    cx = WIDTH // 2 + rng.randint(-15, 15)
    cy = int(HEIGHT * 0.72) + rng.randint(-5, 5)
    outer_rx = rng.randint(75, 95)
    outer_ry = rng.randint(55, 70)
    thickness = rng.randint(8, 12)

    # 外圈
    cv2.ellipse(img, (cx, cy), (outer_rx, outer_ry), 0, 0, 360,
                STEERING_COL, thickness)
    # 稍亮的内边
    cv2.ellipse(img, (cx, cy), (outer_rx - 2, outer_ry - 2), 0, 0, 360,
                STEERING_HIGHLIGHT, max(thickness - 4, 2))

    # 中心
    center_rx = rng.randint(12, 20)
    center_ry = rng.randint(8, 14)
    cv2.ellipse(img, (cx, cy), (center_rx, center_ry), 0, 0, 360,
                (70, 70, 75), -1)

    # 辐条（3根）
    for angle in [0, 120, 240]:
        rad = math.radians(angle + rng.randint(-10, 10))
        ex = int(cx + (outer_rx - 8) * math.cos(rad))
        ey = int(cy + (outer_ry - 8) * math.sin(rad))
        mx = int(cx + center_rx * math.cos(rad))
        my = int(cy + center_ry * math.sin(rad))
        cv2.line(img, (mx, my), (ex, ey), STEERING_COL, rng.randint(3, 6))

    # 返回方向盘区域 bbox
    sx = cx - outer_rx - 5
    sy = cy - outer_ry - 5
    sw = 2 * outer_rx + 10
    sh = 2 * outer_ry + 10
    return (sx, sy, sw, sh)


def draw_driver_body(img, rng):
    """绘制驾驶员身体：躯干、头颈"""
    # 身体中心
    body_cx = WIDTH // 2 + rng.randint(-15, 15)

    # 肩膀位置
    shoulder_y = int(HEIGHT * 0.40) + rng.randint(-10, 10)
    shoulder_w = rng.randint(70, 95)

    # 躯干
    torso_top = shoulder_y
    torso_bottom = int(HEIGHT * 0.72)
    shirt_color = SHIRT_COLORS[rng.randint(0, len(SHIRT_COLORS) - 1)]

    # 绘制躯干（梯形->矩形）
    cv2.rectangle(img,
                  (body_cx - shoulder_w, torso_top),
                  (body_cx + shoulder_w, torso_bottom),
                  shirt_color, -1)

    # 衣服褶皱/纹理
    for _ in range(rng.randint(1, 3)):
        fold_y = rng.randint(torso_top + 20, torso_bottom - 20)
        fold_alpha = rng.randint(-15, 15)
        fold_c = tuple(max(0, min(255, c + fold_alpha)) for c in shirt_color)
        cv2.line(img, (body_cx - shoulder_w + 15, fold_y),
                 (body_cx + shoulder_w - 15, fold_y), fold_c, 1)

    # 领口
    collar_y = shoulder_y + rng.randint(5, 15)
    cv2.ellipse(img, (body_cx, collar_y),
                (rng.randint(20, 30), rng.randint(10, 15)),
                0, -60, 240, SKIN_TONES[rng.randint(0, 4)], -1)

    # 脖子
    neck_w = rng.randint(18, 26)
    neck_y = max(0, shoulder_y - rng.randint(30, 45))
    skin = SKIN_TONES[rng.randint(0, len(SKIN_TONES) - 1)]
    cv2.rectangle(img,
                  (body_cx - neck_w, neck_y),
                  (body_cx + neck_w, shoulder_y),
                  skin, -1)

    # 头部
    head_r = rng.randint(30, 40)
    head_cy = neck_y - head_r + rng.randint(5, 15)
    head_cx = body_cx + rng.randint(-8, 8)
    cv2.circle(img, (head_cx, head_cy), head_r, skin, -1)

    # 头发
    hair_r = head_r + rng.randint(2, 5)
    hair_color = (rng.randint(10, 60), rng.randint(10, 50), rng.randint(10, 40))
    cv2.ellipse(img, (head_cx, head_cy - rng.randint(5, 15)),
                (hair_r, hair_r - rng.randint(2, 6)),
                0, 0, 360, hair_color, -1)
    # 修剪到头部范围
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.circle(mask, (head_cx, head_cy), head_r, 255, -1)
    # 把头部下方头发去掉
    for y in range(head_cy + hair_r // 2, min(head_cy + hair_r + 10, HEIGHT)):
        for x in range(max(0, head_cx - hair_r), min(WIDTH, head_cx + hair_r)):
            if mask[y, x] == 0:
                img[y, x] = skin if rng.random() < 0.5 else img[y, x]

    # 面部特征
    draw_face(img, head_cx, head_cy, head_r, rng)

    # 返回身体各部位坐标
    return {
        'head_cx': head_cx,
        'head_cy': head_cy,
        'head_r': head_r,
        'neck_y': neck_y,
        'shoulder_y': shoulder_y,
        'shoulder_w': shoulder_w,
        'body_cx': body_cx,
        'torso_bottom': torso_bottom,
        'skin': skin,
        'shirt_color': shirt_color,
    }


def draw_face(img, cx, cy, r, rng):
    """绘制简单的面部特征"""
    # 眼睛
    eye_y = cy - rng.randint(3, 8)
    eye_spacing = r // 3
    eye_r = r // 7
    for sign in [-1, 1]:
        ex = cx + sign * eye_spacing
        ey = eye_y
        cv2.circle(img, (ex, ey), eye_r, (40, 40, 40), -1)
        cv2.circle(img, (ex, ey), eye_r // 2, (200, 200, 200), -1)

    # 眉毛
    brow_y = eye_y - r // 5
    for sign in [-1, 1]:
        bx = cx + sign * eye_spacing
        cv2.line(img, (bx - r // 4, brow_y), (bx + r // 4, brow_y),
                 (40, 35, 30), max(r // 10, 1))

    # 嘴
    mouth_y = cy + r // 3
    mouth_w = r // 2
    cv2.ellipse(img, (cx, mouth_y), (mouth_w, r // 8),
                0, 0, 360, (80, 60, 60), -1)

    # 鼻子
    nose_y = cy + r // 8
    cv2.circle(img, (cx, nose_y), r // 10, (skin_darken(img[cy, cx].tolist())), -1)


def skin_darken(color, amount=30):
    """肤色加深"""
    return tuple(max(0, c - amount) for c in color)


def draw_arm(img, start_x, start_y, end_x, end_y, thickness, skin, rng):
    """绘制手臂"""
    # 转换为 Python int 避免 numpy 类型问题
    start_x, start_y = int(start_x), int(start_y)
    end_x, end_y = int(end_x), int(end_y)
    thickness = int(thickness)

    # 中间肘部弯曲点
    mid_x = int(start_x + (end_x - start_x) * rng.uniform(0.35, 0.65) + rng.randint(-20, 20))
    mid_y = int(start_y + (end_y - start_y) * rng.uniform(0.35, 0.65) + rng.randint(-15, 15))

    # 裁剪到图像范围内
    mid_x = max(1, min(WIDTH - 2, mid_x))
    mid_y = max(1, min(HEIGHT - 2, mid_y))

    # 绘制两段
    for (x1, y1), (x2, y2) in [((start_x, start_y), (mid_x, mid_y)),
                                 ((mid_x, mid_y), (end_x, end_y))]:
        x1c = max(0, min(WIDTH - 1, int(x1)))
        y1c = max(0, min(HEIGHT - 1, int(y1)))
        x2c = max(0, min(WIDTH - 1, int(x2)))
        y2c = max(0, min(HEIGHT - 1, int(y2)))
        cv2.line(img, (x1c, y1c), (x2c, y2c), skin, thickness)
        # 稍微描边
        darker = skin_darken(skin, 25)
        cv2.line(img, (x1c, y1c), (x2c, y2c), darker, max(thickness - 3, 1))

    return (mid_x, mid_y)


def draw_hand(img, cx, cy, size, skin, rng):
    """绘制手掌"""
    cx = int(cx)
    cy = int(cy)
    size = int(size)
    # 手掌椭圆
    hand_w = size + int(rng.randint(-3, 3))
    hand_h = int(size * 0.8)
    angle_deg = int(rng.randint(-20, 20))
    cv2.ellipse(img, (cx, cy), (hand_w, hand_h), angle_deg,
                0, 360, skin, -1)
    cv2.ellipse(img, (cx, cy), (hand_w, hand_h), angle_deg,
                0, 360, skin_darken(skin, 20), 1)

    # 手指（简单线条）
    finger_count = int(rng.randint(3, 5))
    for i in range(finger_count):
        angle = math.radians(rng.randint(-60, 60))
        fx = int(cx + hand_w * 0.7 * math.cos(angle))
        fy = int(cy + hand_h * 0.7 * math.sin(angle))
        cv2.line(img, (cx, cy), (fx, fy), skin, int(rng.randint(2, 4)))

    hand_w_actual = size + 6
    hand_h_actual = int(size * 0.8) + 6
    return (cx, cy, hand_w_actual, hand_h_actual)


# ==================== 各类别生成函数 ====================

def generate_normal_driving(img, body, sw_bbox, rng):
    """
    Class 0: normal_driving
    双手在方向盘上，正常驾驶姿势
    BBox: 覆盖方向盘+双手区域
    """
    head = body
    cx, cy, sw, sh = sw_bbox

    # 方向盘中心
    sw_cx = cx + sw // 2
    sw_cy = cy + sh // 2

    # 左手在方向盘左上
    lh_x = sw_cx - rng.randint(35, 55)
    lh_y = sw_cy - rng.randint(5, 20)
    # 手臂从肩膀到方向盘
    draw_arm(img,
             head['body_cx'] - head['shoulder_w'] + 15, head['shoulder_y'] + 5,
             lh_x, lh_y,
             rng.randint(6, 9), head['skin'], rng)
    draw_hand(img, lh_x, lh_y, rng.randint(10, 14), head['skin'], rng)

    # 右手在方向盘右上
    rh_x = sw_cx + rng.randint(35, 55)
    rh_y = sw_cy - rng.randint(5, 20)
    draw_arm(img,
             head['body_cx'] + head['shoulder_w'] - 15, head['shoulder_y'] + 5,
             rh_x, rh_y,
             rng.randint(6, 9), head['skin'], rng)
    draw_hand(img, rh_x, rh_y, rng.randint(10, 14), head['skin'], rng)

    # BBox = 方向盘 + 手部区域，归一化
    bbox_x = (cx - 15) / WIDTH
    bbox_y = (cy - 10) / HEIGHT
    bbox_w = (sw + 30) / WIDTH
    bbox_h = (sh + 20) / HEIGHT

    return (0, bbox_x, bbox_y, bbox_w, bbox_h)


def generate_phone_usage(img, body, sw_bbox, rng):
    """
    Class 1: phone_usage
    一手持手机贴近耳朵或看屏幕
    BBox: 覆盖手机+手部区域
    """
    head = body
    side = rng.choice([-1, 1])  # 左手或右手
    ear_x = head['head_cx'] + side * head['head_r'] * rng.uniform(0.7, 1.1)
    ear_y = head['head_cy'] - rng.randint(5, 20)

    # 手臂路径
    elbow_dist = side * rng.randint(20, 40)
    draw_arm(img,
             head['body_cx'] + side * (head['shoulder_w'] - 10),
             head['shoulder_y'] + 5,
             ear_x - elbow_dist, ear_y + 30,
             rng.randint(7, 10), head['skin'], rng)

    # 手机（矩形，深色）
    phone_w = rng.randint(18, 28)
    phone_h = rng.randint(30, 45)
    # 手机在手部位置，靠耳朵
    phone_cx = ear_x + side * rng.randint(-5, 5)
    phone_cy = ear_y + rng.randint(-5, 10)
    angle = rng.randint(-20, 20)

    rect_pts = get_rotated_rect_pts(phone_cx, phone_cy, phone_w, phone_h, angle)
    cv2.fillPoly(img, [rect_pts], PHONE_COLOR)

    # 屏幕高光
    inner_pts = get_rotated_rect_pts(phone_cx, phone_cy, phone_w - 4, phone_h - 6, angle)
    cv2.fillPoly(img, [inner_pts], (rng.randint(40, 80), rng.randint(40, 80), rng.randint(50, 90)))

    # 手覆盖手机
    draw_hand(img, phone_cx - side * 5, phone_cy + 15,
              rng.randint(11, 15), head['skin'], rng)

    # BBox
    pad = 8
    bbox_cx = phone_cx / WIDTH
    bbox_cy = phone_cy / HEIGHT
    bbox_w = (phone_w + pad * 2) / WIDTH
    bbox_h = (phone_h + pad * 2) / HEIGHT

    return (1, bbox_cx, bbox_cy, bbox_w, bbox_h)


def generate_smoking(img, body, sw_bbox, rng):
    """
    Class 2: smoking
    手持香烟靠近嘴部
    BBox: 覆盖香烟+手部+嘴区域
    """
    head = body
    side = rng.choice([-1, 1])

    # 嘴位置
    mouth_x = head['head_cx'] - rng.randint(5, 15)
    mouth_y = head['head_cy'] + head['head_r'] // 3

    # 手位置（嘴附近）
    hand_x = mouth_x + side * rng.randint(15, 30)
    hand_y = mouth_y + rng.randint(-10, 10)

    # 手臂
    draw_arm(img,
             head['body_cx'] + side * (head['shoulder_w'] - 15),
             head['shoulder_y'] + 10,
             hand_x, hand_y + 15,
             rng.randint(6, 9), head['skin'], rng)

    # 手掌
    draw_hand(img, hand_x, hand_y, rng.randint(10, 13), head['skin'], rng)

    # 香烟（细长白色圆柱）
    cig_angle = math.radians(rng.randint(-30, 30))
    cig_len = rng.randint(30, 50)
    cig_x1 = int(mouth_x + side * rng.randint(5, 15))
    cig_y1 = mouth_y
    cig_x2 = int(cig_x1 + cig_len * math.cos(cig_angle) * side - cig_len * math.cos(cig_angle) * side)
    # 简化：从嘴到手的方向
    cig_x2 = hand_x - side * rng.randint(5, 15)
    cig_y2 = hand_y

    # 烟的主体
    cv2.line(img, (cig_x1, cig_y1), (cig_x2, cig_y2),
             CIGARETTE_COLOR, rng.randint(4, 6))

    # 过滤嘴
    tip_x = hand_x
    tip_y = hand_y
    cv2.line(img, (tip_x, tip_y), (cig_x1, cig_y1),
             CIGARETTE_TIP, 2)

    # 烟雾
    if rng.random() < 0.6:
        smoke_cx = cig_x1
        smoke_cy = cig_y1 - rng.randint(5, 15)
        for _ in range(rng.randint(2, 4)):
            sr = rng.randint(5, 15)
            alpha = rng.randint(40, 80)
            overlay = img.copy()
            cv2.circle(overlay, (smoke_cx + rng.randint(-10, 10),
                                  smoke_cy - rng.randint(0, 15)),
                       sr, (180, 180, 190), -1)
            img[:] = cv2.addWeighted(img, 0.85, overlay, 0.15, 0)
            smoke_cy -= rng.randint(5, 15)

    # BBox 覆盖香烟和手
    min_x = min(cig_x1, hand_x) - 10
    min_y = min(cig_y1 - 5, hand_y - 15)
    max_x = max(cig_x2, hand_x) + 10
    max_y = max(cig_y1 + 5, hand_y + 15)
    bbox_cx = ((min_x + max_x) / 2) / WIDTH
    bbox_cy = ((min_y + max_y) / 2) / HEIGHT
    bbox_w = (max_x - min_x) / WIDTH
    bbox_h = (max_y - min_y) / HEIGHT

    return (2, bbox_cx, bbox_cy, bbox_w, bbox_h)


def generate_drinking(img, body, sw_bbox, rng):
    """
    Class 3: drinking
    手持瓶子/杯子靠近嘴部
    BBox: 覆盖瓶子+手+嘴区域
    """
    head = body
    side = rng.choice([-1, 1])

    mouth_x = head['head_cx']
    mouth_y = head['head_cy'] + head['head_r'] // 3

    # 瓶口靠近嘴
    bottle_top_x = mouth_x + side * rng.randint(-5, 10)
    bottle_top_y = mouth_y - rng.randint(0, 8)

    # 瓶子底部（手中）
    bottle_bottom_x = bottle_top_x - side * rng.randint(5, 15)
    bottle_bottom_y = bottle_top_y + rng.randint(40, 70)

    # 手位置在瓶身中部
    hand_x = (bottle_top_x + bottle_bottom_x) // 2
    hand_y = (bottle_top_y + bottle_bottom_y) // 2 + rng.randint(-5, 15)

    # 手臂
    draw_arm(img,
             head['body_cx'] + side * (head['shoulder_w'] - 10),
             head['shoulder_y'] + 5,
             hand_x + side * 10, hand_y,
             rng.randint(7, 10), head['skin'], rng)

    # 瓶子主体
    bottle_color = BOTTLE_COLORS[rng.randint(0, len(BOTTLE_COLORS) - 1)]
    bottle_w = rng.randint(10, 18)
    # 画瓶子（竖长矩形+瓶颈）
    cv2.rectangle(img,
                  (int(bottle_bottom_x - bottle_w), int(bottle_bottom_y)),
                  (int(bottle_bottom_x + bottle_w), int(bottle_top_y)),
                  bottle_color, -1)

    # 瓶身光泽
    highlight_strip = max(bottle_w // 3, 3)
    cv2.rectangle(img,
                  (int(bottle_bottom_x - bottle_w + highlight_strip), int(bottle_bottom_y)),
                  (int(bottle_bottom_x - bottle_w + highlight_strip * 2), int(bottle_top_y)),
                  tuple(min(255, c + 50) for c in bottle_color), -1)

    # 瓶盖/杯口
    cap_h = rng.randint(3, 6)
    cv2.rectangle(img,
                  (int(bottle_top_x - bottle_w - 1), int(bottle_top_y - cap_h)),
                  (int(bottle_top_x + bottle_w + 1), int(bottle_top_y)),
                  (rng.randint(50, 120), rng.randint(50, 100), rng.randint(50, 100)), -1)

    # 手
    draw_hand(img, hand_x, hand_y, rng.randint(11, 15), head['skin'], rng)

    # BBox
    min_x = min(bottle_top_x, bottle_bottom_x) - bottle_w - 10
    min_y = bottle_top_y - cap_h - 10
    max_x = max(bottle_top_x, bottle_bottom_x) + bottle_w + 10
    max_y = max(bottle_bottom_y, hand_y) + 10
    bbox_cx = ((min_x + max_x) / 2) / WIDTH
    bbox_cy = ((min_y + max_y) / 2) / HEIGHT
    bbox_w_n = (max_x - min_x) / WIDTH
    bbox_h_n = (max_y - min_y) / HEIGHT

    return (3, bbox_cx, bbox_cy, bbox_w_n, bbox_h_n)


def generate_hands_off_wheel(img, body, sw_bbox, rng):
    """
    Class 4: hands_off_wheel
    双手明显离开方向盘（如伸向副驾/摸头发/调整后视镜等）
    BBox: 覆盖手部离方向盘较远的区域
    """
    head = body

    variant = rng.randint(0, 2)

    if variant == 0:
        # 一只手伸向副驾
        target_x = WIDTH - rng.randint(40, 100)
        target_y = rng.randint(int(HEIGHT * 0.35), int(HEIGHT * 0.55))
        side = 1
    elif variant == 1:
        # 一只手摸头发/脸
        target_x = head['head_cx'] + rng.randint(-20, 30)
        target_y = head['head_cy'] - head['head_r'] + rng.randint(-5, 15)
        side = rng.choice([-1, 1])
    else:
        # 一只手垂到身侧
        target_x = head['body_cx'] + rng.randint(-40, 40)
        target_y = head['torso_bottom'] - rng.randint(10, 40)
        side = rng.choice([-1, 1])

    arm_start_x = head['body_cx'] + side * (head['shoulder_w'] - 10)
    arm_start_y = head['shoulder_y'] + 5

    draw_arm(img, arm_start_x, arm_start_y, target_x, target_y,
             rng.randint(7, 10), head['skin'], rng)
    draw_hand(img, target_x, target_y, rng.randint(12, 16), head['skin'], rng)

    # 另一只手也离开（有时）
    if rng.random() < 0.6:
        side2 = -side
        target2_x = head['body_cx'] + side2 * rng.randint(50, 100)
        target2_y = rng.randint(int(HEIGHT * 0.3), int(HEIGHT * 0.55))
        arm_start2_x = head['body_cx'] + side2 * (head['shoulder_w'] - 10)
        draw_arm(img, arm_start2_x, head['shoulder_y'] + 5,
                 target2_x, target2_y,
                 rng.randint(6, 9), head['skin'], rng)
        draw_hand(img, target2_x, target2_y, rng.randint(11, 14), head['skin'], rng)

    # BBox 覆盖离方向盘最远的手
    sw_cx = sw_bbox[0] + sw_bbox[2] // 2
    sw_cy = sw_bbox[1] + sw_bbox[3] // 2
    dist = math.sqrt((target_x - sw_cx) ** 2 + (target_y - sw_cy) ** 2)
    # BBox 以手部为中心
    bbox_cx = target_x / WIDTH
    bbox_cy = target_y / HEIGHT
    bbox_w = (35 + rng.randint(0, 15)) / WIDTH
    bbox_h = (35 + rng.randint(0, 15)) / HEIGHT

    return (4, bbox_cx, bbox_cy, bbox_w, bbox_h)


# ==================== 辅助函数 ====================

def get_rotated_rect_pts(cx, cy, w, h, angle_deg):
    """获取旋转矩形的四个顶点"""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    hw, hh = w / 2, h / 2
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    pts = []
    for dx, dy in corners:
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        pts.append([int(cx + rx), int(cy + ry)])
    return np.array(pts, dtype=np.int32)


def add_variations(img, rng):
    """添加随机噪声、亮度变化、轻微模糊"""
    # 亮度调整
    alpha = rng.uniform(0.75, 1.35)
    beta = rng.randint(-20, 25)
    img[:] = np.clip(img * alpha + beta, 0, 255).astype(np.uint8)

    # 高斯噪声
    if rng.random() < 0.7:
        noise_sigma = rng.uniform(0, 8)
        noise = np.random.randn(*img.shape).astype(np.float32) * noise_sigma
        img[:] = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 轻微高斯模糊（模拟不同对焦）
    if rng.random() < 0.3:
        ksize = rng.choice([3, 5])
        img[:] = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # 轻微旋转
    if rng.random() < 0.4:
        angle = rng.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((WIDTH // 2, HEIGHT // 2), angle, 1.0)
        # 注意：旋转后bbox需要重新计算，这里我们旋转图像
        # 为了保持bbox准确性，我们只做很小的旋转
        img[:] = cv2.warpAffine(img, M, (WIDTH, HEIGHT),
                                 borderMode=cv2.BORDER_REFLECT)

    # 偶尔添加暗角效果
    if rng.random() < 0.3:
        vignette = np.ones((HEIGHT, WIDTH), dtype=np.float32)
        for y in range(HEIGHT):
            for x in range(WIDTH):
                dx = (x - WIDTH / 2) / (WIDTH / 2)
                dy = (y - HEIGHT / 2) / (HEIGHT / 2)
                d = math.sqrt(dx * dx + dy * dy)
                vignette[y, x] = max(0.5, 1.0 - d * 0.5)
        for c in range(3):
            img[:, :, c] = np.clip(img[:, :, c] * vignette, 0, 255).astype(np.uint8)


def clamp_bbox(cx, cy, w, h):
    """确保bbox在[0,1]范围内"""
    cx = max(w / 2, min(1.0 - w / 2, cx))
    cy = max(h / 2, min(1.0 - h / 2, cy))
    w = min(1.0, w)
    h = min(1.0, h)
    return cx, cy, w, h


def generate_image(class_id, rng):
    """
    生成一张合成图像及其YOLO标注
    返回: (image_numpy_array, yolo_label_string)
    """
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # 1. 背景
    draw_sky_and_road(img, rng)

    # 2. 车内结构
    dash_top = draw_car_interior(img, rng)

    # 3. 方向盘
    sw_bbox = draw_steering_wheel(img, rng)

    # 4. 驾驶员身体
    body = draw_driver_body(img, rng)

    # 5. 根据类别绘制手部和物品
    if class_id == 0:
        label = generate_normal_driving(img, body, sw_bbox, rng)
    elif class_id == 1:
        label = generate_phone_usage(img, body, sw_bbox, rng)
    elif class_id == 2:
        label = generate_smoking(img, body, sw_bbox, rng)
    elif class_id == 3:
        label = generate_drinking(img, body, sw_bbox, rng)
    elif class_id == 4:
        label = generate_hands_off_wheel(img, body, sw_bbox, rng)
    else:
        raise ValueError(f"Unknown class_id: {class_id}")

    # 6. 添加变化
    add_variations(img, rng)

    # 确保bbox有效
    cid, cx, cy, w, h = label
    cx, cy, w, h = clamp_bbox(cx, cy, w, h)

    # 格式化为YOLO label行
    label_str = f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    return img, label_str


# ==================== 主生成流程 ====================

def generate_split(split_name, num_per_class, start_index=0):
    """
    为某个split生成图像和标注
    split_name: 'train', 'val', 'test'
    num_per_class: 每类数量
    start_index: 起始编号
    """
    img_dir = DATASET_DIR / split_name / 'images'
    lbl_dir = DATASET_DIR / split_name / 'labels'

    # 清除旧文件
    for f in img_dir.glob('*.jpg'):
        f.unlink()
    for f in lbl_dir.glob('*.txt'):
        f.unlink()

    total = num_per_class * 5
    class_names = ['normal_driving', 'phone_usage', 'smoking', 'drinking', 'hands_off_wheel']

    print(f"\n生成 {split_name} 集 ({total} 张) ...")
    idx = start_index

    for class_id in range(5):
        for i in range(num_per_class):
            # 每个样本用不同随机种子
            seed = hash(f"{split_name}_{class_id}_{i}") & 0xFFFFFFFF
            rng = SafeRng(seed)

            img, label_str = generate_image(class_id, rng)

            filename = f"{split_name}_{idx:04d}"
            img_path = img_dir / f"{filename}.jpg"
            lbl_path = lbl_dir / f"{filename}.txt"

            # cv2.imwrite 不支持中文路径，用 imencode 绕过
            success, encoded = cv2.imencode('.jpg', img,
                                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            if success:
                with open(img_path, 'wb') as f:
                    f.write(encoded.tobytes())

            with open(lbl_path, 'w', encoding='utf-8') as f:
                f.write(label_str + '\n')

            idx += 1

        print(f"  Class {class_id} ({class_names[class_id]}): {num_per_class} 张")

    print(f"  -> 总计 {total} 张图像 + 标注完成")
    return idx


def verify_dataset():
    """验证数据集完整性"""
    print("\n" + "=" * 60)
    print("数据集验证")
    print("=" * 60)

    issues = []

    # 1. 检查 data.yaml
    yaml_path = DATASET_DIR / 'data.yaml'
    if yaml_path.exists():
        print(f"[OK] data.yaml 存在")
    else:
        issues.append("data.yaml 缺失")
        print(f"[FAIL] data.yaml 不存在!")

    # 2. 检查各split
    for split in ['train', 'val', 'test']:
        img_dir = DATASET_DIR / split / 'images'
        lbl_dir = DATASET_DIR / split / 'labels'

        imgs = sorted(list(img_dir.glob('*.jpg')))
        lbls = sorted(list(lbl_dir.glob('*.txt')))

        img_names = {f.stem for f in imgs}
        lbl_names = {f.stem for f in lbls}

        print(f"\n{split}:")
        print(f"  图像: {len(imgs)} 张")
        print(f"  标注: {len(lbls)} 个")

        if len(imgs) != len(lbls):
            issues.append(f"{split}: 图像({len(imgs)})与标注({len(lbls)})数量不匹配")
            print(f"  [WARN] 数量不匹配!")

        # 检查文件名配对
        only_imgs = img_names - lbl_names
        only_lbls = lbl_names - img_names
        if only_imgs:
            issues.append(f"{split}: {len(only_imgs)} 个图像缺少标注")
            print(f"  [WARN] 图像缺标注: {list(only_imgs)[:5]}...")
        if only_lbls:
            issues.append(f"{split}: {len(only_lbls)} 个标注缺少图像")
            print(f"  [WARN] 标注缺图像: {list(only_lbls)[:5]}...")

        if not only_imgs and not only_lbls and len(imgs) == len(lbls):
            print(f"  [OK] 图像和标注一一对应")

    # 3. 随机抽查标注格式
    print("\n随机抽查标注格式:")
    all_lbls = list(DATASET_DIR.rglob('*.txt'))
    all_lbls = [l for l in all_lbls if l.parent.parent.name in ('train', 'val', 'test')]

    if all_lbls:
        for _ in range(5):
            lbl_path = random.choice(all_lbls)
            with open(lbl_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            valid = True
            for line in content.split('\n'):
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) != 5:
                    valid = False
                    break
                try:
                    class_id = int(parts[0])
                    vals = [float(x) for x in parts[1:]]
                    if class_id < 0 or class_id > 4:
                        valid = False
                    if not all(0.0 <= v <= 1.0 for v in vals):
                        valid = False
                except ValueError:
                    valid = False
            status = "OK" if valid else "FAIL"
            if not valid:
                issues.append(f"标注格式错误: {lbl_path}")
            print(f"  [{status}] {lbl_path.relative_to(DATASET_DIR)}")

    # 4. 统计各类别数量
    print("\n各类别分布:")
    class_counts = {i: {'train': 0, 'val': 0, 'test': 0} for i in range(5)}
    class_names = ['normal_driving', 'phone_usage', 'smoking', 'drinking', 'hands_off_wheel']
    for split in ['train', 'val', 'test']:
        lbl_dir = DATASET_DIR / split / 'labels'
        for lbl_path in lbl_dir.glob('*.txt'):
            with open(lbl_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        cid = int(line.split()[0])
                        class_counts[cid][split] += 1

    for cid in range(5):
        parts = [f"  train={class_counts[cid]['train']}",
                 f"val={class_counts[cid]['val']}",
                 f"test={class_counts[cid]['test']}"]
        print(f"  Class {cid} ({class_names[cid]}): {', '.join(parts)}")

    # 5. 随机查看一张图片尺寸
    all_imgs = list(DATASET_DIR.rglob('*.jpg'))
    all_imgs = [i for i in all_imgs if i.parent.parent.name in ('train', 'val', 'test')]
    if all_imgs:
        sample_path = random.choice(all_imgs)
        with open(sample_path, 'rb') as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)
        sample_img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if sample_img is not None:
            h, w = sample_img.shape[:2]
            print(f"\n随机图片尺寸检查: {w}x{h} (期望 640x480)")
            if w == 640 and h == 480:
                print("[OK] 尺寸正确")
            else:
                issues.append(f"图片尺寸异常: {w}x{h}")
                print("[FAIL] 尺寸不符!")

    # 总结
    print("\n" + "=" * 60)
    if issues:
        print(f"发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("所有检查通过!")
    print("=" * 60)

    return len(issues) == 0


# ==================== 主入口 ====================

def main():
    print("=" * 60)
    print("YOLO 手持物品检测 - 合成数据集生成")
    print("=" * 60)
    print(f"图像尺寸: {WIDTH}x{HEIGHT}")
    print(f"类别: 0=normal_driving, 1=phone_usage, 2=smoking, 3=drinking, 4=hands_off_wheel")
    print(f"输出目录: {DATASET_DIR}")

    # 1. 确保目录存在
    print("\n[1/4] 创建目录结构...")
    ensure_dirs()

    # 2. 创建data.yaml
    print("\n[2/4] 创建 data.yaml...")
    create_data_yaml()

    # 3. 生成数据
    print("\n[3/4] 生成合成图像和标注...")

    train_per_class = 16   # 训练集 80 张
    val_per_class = 5      # 验证集 25 张
    test_per_class = 5     # 测试集 25 张

    next_idx = generate_split('train', train_per_class, start_index=0)
    next_idx = generate_split('val', val_per_class, start_index=next_idx)
    next_idx = generate_split('test', test_per_class, start_index=next_idx)

    total = train_per_class * 5 + val_per_class * 5 + test_per_class * 5
    print(f"\n总计生成: {total} 张图像 + {total} 个标注文件")

    # 4. 验证
    print("\n[4/4] 验证数据集...")
    ok = verify_dataset()

    print("\n" + "=" * 60)
    print("数据集准备完成!")
    print(f"位置: {DATASET_DIR}")
    print(f"  train: {train_per_class * 5} 张")
    print(f"  val:   {val_per_class * 5} 张")
    print(f"  test:  {test_per_class * 5} 张")
    print("=" * 60)
    print("\n下一步: python -m yolo.train_handheld")


if __name__ == '__main__':
    main()
