"""
处理流水线 - 单帧编排器
======================
负责协调所有检测器对单帧图像进行处理，串联面部检测、疲劳分析、
头部姿态估计、视线估计、分心检测和生理信号监测。
"""

import time
import logging
import numpy as np
from config import (
    FPS_TARGET, HEAD_TURN_DURATION, HEAD_TURN_YAW_THRESHOLD, PPG_WINDOW
)

logger = logging.getLogger(__name__)


class FramePipeline:
    """
    单帧处理编排器
    串联所有检测器，对每一帧进行完整的驾驶注意力分析
    """

    def __init__(self, enable_modules=None):
        """
        初始化所有检测器

        参数:
            enable_modules: dict 控制启用哪些模块
                {'fatigue': True, 'pose': True, 'gaze': True,
                 'distraction': True, 'physio': False}
        """
        # 默认启用设置
        if enable_modules is None:
            enable_modules = {
                'fatigue': True,
                'pose': True,
                'gaze': True,
                'distraction': True,
                'physio': False
            }
        self.enable_modules = enable_modules
        self.demo_mode = bool(enable_modules.get('demo_mode', False))

        # 延迟导入检测器模块（避免加载时的循环依赖）
        from detectors.face_detector import FaceDetector

        self.face_detector = FaceDetector()

        # 按需加载其他检测器
        self.fatigue_detector = None
        self.head_pose_estimator = None
        self.gaze_estimator = None
        self.distraction_detector = None
        self.rppg_monitor = None

        if enable_modules.get('fatigue', True):
            from detectors.fatigue import FatigueDetector
            self.fatigue_detector = FatigueDetector()

        if enable_modules.get('pose', True):
            from detectors.head_pose import HeadPoseEstimator
            self.head_pose_estimator = HeadPoseEstimator(demo_mode=self.demo_mode)

        if enable_modules.get('gaze', True):
            from detectors.gaze import GazeEstimator
            self.gaze_estimator = GazeEstimator()

        if enable_modules.get('distraction', True):
            from detectors.distraction import DistractionDetector
            self.distraction_detector = DistractionDetector(demo_mode=self.demo_mode)

        if enable_modules.get('physio', False):
            from detectors.physiological import rPPGMonitor
            self.rppg_monitor = rPPGMonitor()

        # 告警管理器（始终启用）
        from alert_manager import AlertManager
        self.alert_manager = AlertManager()

        # 时序数据记录（限制最大长度防止内存泄漏）
        self._max_history = 300  # 5FPS下约60秒数据
        self.ear_history = []
        self.mar_history = []
        self.pitch_history = []
        self.yaw_history = []
        self.roll_history = []
        self.ppg_history = []
        self.frame_count = 0
        self.start_time = None
        self.last_face_result = None

    def process_frame(self, frame, timestamp=None):
        """
        处理单帧图像

        参数:
            frame: BGR格式的numpy图像数组
            timestamp: 时间戳(秒)，None则自动生成

        返回:
            ProcessedFrame 字典:
            {
                'frame': 原始帧或标注帧,
                'face_detected': bool,
                'fatigue': {ear, mar, blink_rate, ...},
                'head_pose': {pitch, yaw, roll},
                'gaze': {gaze_vector, gaze_angle, is_deviated},
                'distraction': {objects, hands_off, body_turn},
                'physio': {heart_rate, ...},
                'alerts': [...],
                'summary': {...}
            }
        """
        if self.start_time is None:
            self.start_time = timestamp or time.time()
        if timestamp is None:
            timestamp = time.time() - self.start_time

        self.frame_count += 1

        result = {
            'frame': frame,
            'timestamp': timestamp,
            'frame_count': self.frame_count,
            'face_detected': False,
            'fatigue': {},
            'head_pose': {},
            'gaze': {},
            'distraction': {},
            'physio': {},
            'alerts': [],
            'summary': {}
        }

        # 1. 面部检测
        try:
            face_result = self.face_detector.detect(frame)
        except Exception as e:
            logger.error("面部检测失败: %s", e, exc_info=True)
            face_result = None
        self.last_face_result = face_result

        if face_result is None:
            # 无面部检测到，仅做分心检测（不需要面部）
            if self.distraction_detector:
                try:
                    dist_result = self.distraction_detector.update_and_check(
                        frame, None, timestamp
                    )
                    if dist_result:
                        result['distraction'] = {
                            'objects': dist_result.get('objects', []),
                            'hands_off_wheel': dist_result.get('hands_off_wheel', False),
                            'hand_status': dist_result.get('hand_status', {}),
                            'body_turn': dist_result.get('body_turn', False),
                            'body_turn_info': dist_result.get('body_turn_info', {}),
                            'pose_keypoints': dist_result.get('pose_keypoints', []),
                        }
                        for alert in dist_result.get('alerts', []):
                            self.alert_manager.add_alert(alert)
                            result['alerts'].append(alert)
                except Exception as e:
                    logger.error("分心检测失败 (无面部): %s", e, exc_info=True)
            return result

        result['face_detected'] = True

        # 2. 疲劳检测 (EAR, MAR, 眨眼)
        if self.fatigue_detector:
            try:
                left_eye_pts, right_eye_pts = self.face_detector.get_eye_regions(face_result)
                mouth_pts = self.face_detector.get_mouth_region(face_result)

                from utils.geometry import compute_ear_both, compute_mar
                ear_val = compute_ear_both(left_eye_pts, right_eye_pts)
                mar_val = compute_mar(mouth_pts)

                fat_result = self.fatigue_detector.update(ear_val, mar_val, timestamp)

                result['fatigue'] = {
                    'ear': ear_val,
                    'mar': mar_val,
                    'blink_rate': fat_result.get('blink_rate', 0),
                    'eye_closure': fat_result.get('eye_closure', False),
                    'yawn': fat_result.get('yawn', False),
                }

                for alert in fat_result.get('alerts', []):
                    alert['source'] = 'fatigue'
                    self.alert_manager.add_alert(alert)
                    result['alerts'].append(alert)

                # 记录时序数据（限制长度防止内存泄漏）
                self.ear_history.append({'t': timestamp, 'v': ear_val})
                self.mar_history.append({'t': timestamp, 'v': mar_val})
                if len(self.ear_history) > self._max_history:
                    self.ear_history = self.ear_history[-self._max_history:]
                    self.mar_history = self.mar_history[-self._max_history:]
            except Exception as e:
                logger.error("疲劳检测失败: %s", e, exc_info=True)

        # 3. 头部姿态估计
        head_pose = (0.0, 0.0, 0.0)
        if self.head_pose_estimator:
            try:
                hp_points = self.face_detector.get_head_pose_points(face_result)
                pitch, yaw, roll = self.head_pose_estimator.estimate(hp_points)
                head_pose = (pitch, yaw, roll)

                result['head_pose'] = {
                    'pitch': pitch,
                    'yaw': yaw,
                    'roll': roll,
                    'direction': self._head_direction(pitch, yaw),
                    'head_turning': abs(yaw) > HEAD_TURN_YAW_THRESHOLD,
                    'head_turn_threshold': HEAD_TURN_YAW_THRESHOLD,
                    'head_turn_duration_threshold': getattr(
                        self.head_pose_estimator,
                        'head_turn_duration',
                        HEAD_TURN_DURATION,
                    ),
                    'demo_mode': bool(self.demo_mode),
                }

                # 点头检测
                nod_alert = self.head_pose_estimator.check_nodding(pitch, timestamp)
                if nod_alert:
                    nod_alert['source'] = 'head_pose'
                    self.alert_manager.add_alert(nod_alert)
                    result['alerts'].append(nod_alert)

                # 转头检测
                head_turn_alert = self.head_pose_estimator.check_head_turn(yaw, timestamp)
                if head_turn_alert:
                    head_turn_alert['source'] = 'head_pose'
                    self.alert_manager.add_alert(head_turn_alert)
                    result['alerts'].append(head_turn_alert)

                self.pitch_history.append({'t': timestamp, 'v': pitch})
                self.yaw_history.append({'t': timestamp, 'v': yaw})
                self.roll_history.append({'t': timestamp, 'v': roll})
            except Exception as e:
                logger.error("头部姿态估计失败: %s", e, exc_info=True)

        # 4. 视线方向检测
        if self.gaze_estimator:
            try:
                gaze_result = self.gaze_estimator.estimate(face_result, head_pose)

                result['gaze'] = {
                    'gaze_angle': gaze_result.get('gaze_angle', 0.0),
                    'is_deviated': gaze_result.get('is_deviated', False),
                    'gaze_vector': gaze_result.get('gaze_vector', None),
                }

                # 视线偏离检测
                gaze_alert = self.gaze_estimator.check_gaze_deviation(
                    gaze_result.get('gaze_angle', 0.0), timestamp
                )
                if gaze_alert:
                    gaze_alert['source'] = 'gaze'
                    self.alert_manager.add_alert(gaze_alert)
                    result['alerts'].append(gaze_alert)
            except Exception as e:
                logger.error("视线检测失败: %s", e, exc_info=True)

        # 5. 分心检测
        if self.distraction_detector:
            try:
                dist_result = self.distraction_detector.update_and_check(
                    frame, face_result, timestamp
                )
                if dist_result:
                    result['distraction'] = {
                        'objects': dist_result.get('objects', []),
                        'hands_off_wheel': dist_result.get('hands_off_wheel', False),
                        'hand_status': dist_result.get('hand_status', {}),
                        'body_turn': dist_result.get('body_turn', False),
                        'body_turn_info': dist_result.get('body_turn_info', {}),
                        'pose_keypoints': dist_result.get('pose_keypoints', []),
                    }
                    for alert in dist_result.get('alerts', []):
                        alert['source'] = 'distraction'
                        self.alert_manager.add_alert(alert)
                        result['alerts'].append(alert)
            except Exception as e:
                logger.error("分心检测失败: %s", e, exc_info=True)

        # 6. 生理信号监测
        if self.rppg_monitor:
            try:
                forehead_roi = self.face_detector.get_forehead_roi(face_result, frame)
                if forehead_roi is not None and forehead_roi.size > 0:
                    self.rppg_monitor.add_frame(forehead_roi, timestamp)

                    # 每PPG_WINDOW帧计算一次心率
                    if self.frame_count % PPG_WINDOW == 0 and self.frame_count >= PPG_WINDOW:
                        hr = self.rppg_monitor.compute_heart_rate()
                        physio = self.rppg_monitor.get_results()

                        # rPPG 信号质量门控: 低质量时标记而非报警
                        signal_quality = physio.get('signal_quality', 0)
                        if signal_quality < 0.3:
                            result['physio'] = {
                                **physio,
                                'status': 'poor_signal',
                                'message': '信号质量不佳，请保持面部正对摄像头',
                            }
                        else:
                            bp = self.rppg_monitor.estimate_blood_pressure()
                            result['physio'] = {**physio, **bp}

                            if hr and (hr < 50 or hr > 120):
                                alert = {
                                    'source': 'physiological',
                                    'type': 'heart_rate_abnormal',
                                    'severity': 'warning',
                                    'timestamp': timestamp,
                                    'message': f'心率异常: {hr:.0f} BPM'
                                }
                                self.alert_manager.add_alert(alert)
                                result['alerts'].append(alert)

                # 保存PPG信号
                if hasattr(self.rppg_monitor, 'ppg_signal_history'):
                    self.ppg_history = list(self.rppg_monitor.ppg_signal_history)
            except Exception as e:
                logger.error("生理信号监测失败: %s", e, exc_info=True)

        # 7. 更新摘要
        result['summary'] = self.alert_manager.get_summary()

        return result

    def get_annotated_frame(self, frame, face_result, result):
        """获取标注后的帧"""
        annotated = frame.copy()

        if face_result:
            self.face_detector.draw_landmarks(annotated, face_result)

            # 绘制EAR/MAR
            if result.get('fatigue'):
                left_eye_pts, right_eye_pts = self.face_detector.get_eye_regions(face_result)
                mouth_pts = self.face_detector.get_mouth_region(face_result)
                from utils.visualization import draw_eye_contours, draw_mouth_contour
                draw_eye_contours(annotated, left_eye_pts, right_eye_pts,
                                  result['fatigue'].get('ear'))
                draw_mouth_contour(annotated, mouth_pts,
                                   result['fatigue'].get('mar'))

            # 绘制头部姿态轴
            if result.get('head_pose') and self.head_pose_estimator:
                nose_tip = self.face_detector.get_nose_tip(face_result)
                hp = result['head_pose']
                from utils.visualization import draw_head_pose_axes
                draw_head_pose_axes(annotated,
                                    (hp['pitch'], hp['yaw'], hp['roll']),
                                    nose_tip)

            # 绘制视线方向
            if result.get('gaze') and result['gaze'].get('gaze_vector') is not None:
                nose_bridge = self.face_detector.get_nose_bridge(face_result)
                from utils.visualization import draw_gaze_vector
                draw_gaze_vector(annotated, nose_bridge,
                                 result['gaze']['gaze_vector'])

        # 绘制检测框
        if result.get('distraction') and result['distraction'].get('objects'):
            from utils.visualization import draw_detection_boxes
            draw_detection_boxes(annotated, result['distraction']['objects'],
                                 {0: 'normal', 1: 'phone', 2: 'smoking',
                                  3: 'drinking', 4: 'hands_off'})

        # 绘制告警覆盖
        if result.get('alerts'):
            from utils.visualization import draw_alert_overlay
            draw_alert_overlay(annotated, result['alerts'])

        return annotated

    def build_overlay(self, face_result, result, frame_shape):
        """构建前端 canvas 使用的纯图形 overlay 数据。"""
        h, w = frame_shape[:2]
        overlay = {
            'frame_size': {'width': int(w), 'height': int(h)},
            'face_bbox': None,
            'eye_contours': [],
            'mouth_contour': [],
            'head_pose_axes': [],
            'gaze_arrow': None,
            'object_boxes': [],
            'pose_keypoints': [],
            'pose_skeleton': [],
            'wheel_roi': None,
            'virtual_wheel': None,
            'wheel_state': None,
            'wrists': {},
            'shoulders': {},
            'body_turn_vector': None,
            'alert_regions': [],
        }

        if face_result:
            bbox = face_result.get('bbox')
            if bbox is not None:
                x, y, bw, bh = bbox
                overlay['face_bbox'] = [int(x), int(y), int(x + bw), int(y + bh)]

            try:
                left_eye, right_eye = self.face_detector.get_eye_regions(face_result)
                overlay['eye_contours'] = [
                    self._points_to_list(left_eye),
                    self._points_to_list(right_eye),
                ]
            except Exception:
                pass

            try:
                mouth = self.face_detector.get_mouth_region(face_result)
                overlay['mouth_contour'] = self._points_to_list(mouth)
            except Exception:
                pass

            try:
                nose_tip = self.face_detector.get_nose_tip(face_result)
                hp = result.get('head_pose') or {}
                if hp:
                    overlay['head_pose_axes'] = self._head_pose_axes(
                        nose_tip,
                        (hp.get('pitch', 0.0), hp.get('yaw', 0.0), hp.get('roll', 0.0)),
                    )
            except Exception:
                pass

            try:
                gaze = result.get('gaze') or {}
                direction = gaze.get('gaze_vector')
                if direction is not None:
                    origin = self.face_detector.get_nose_bridge(face_result)
                    overlay['gaze_arrow'] = self._arrow_from_vector(origin, direction)
            except Exception:
                pass

        distraction = result.get('distraction') or {}
        overlay['object_boxes'] = [
            {
                'bbox': [int(v) for v in obj.get('bbox', (0, 0, 0, 0))],
                'class_id': int(obj.get('class_id', -1)),
                'severity': self._object_severity(obj),
            }
            for obj in distraction.get('objects', [])
            if obj.get('bbox') is not None
        ]

        for box in overlay['object_boxes']:
            if box['severity'] in {'warning', 'danger'}:
                overlay['alert_regions'].append({
                    'kind': 'object',
                    'bbox': box['bbox'],
                    'severity': box['severity'],
                })

        keypoints = distraction.get('pose_keypoints') or []
        if keypoints:
            overlay['pose_keypoints'] = [
                [float(kp[0]), float(kp[1]), float(kp[2])] for kp in keypoints
            ]
            overlay['pose_skeleton'] = self._pose_skeleton(overlay['pose_keypoints'])

        hand_status = distraction.get('hand_status') or {}
        if hand_status:
            roi = hand_status.get('roi')
            if roi:
                overlay['wheel_roi'] = [int(v) for v in roi]
            wheel = hand_status.get('wheel')
            if wheel:
                overlay['virtual_wheel'] = {
                    'center': [float(v) for v in wheel.get('center', [0, 0])],
                    'radius_x': float(wheel.get('radius_x', 0)),
                    'radius_y': float(wheel.get('radius_y', 0)),
                    'bbox': [int(v) for v in wheel.get('bbox', [])],
                    'grip_left': [float(v) for v in wheel.get('grip_left', [])],
                    'grip_right': [float(v) for v in wheel.get('grip_right', [])],
                }
            overlay['wheel_state'] = {
                'state': hand_status.get('state', 'unknown'),
                'severity': hand_status.get('alert_level') or (
                    'warning' if hand_status.get('state') in {'left_off', 'right_off'} else 'success'
                ),
                'left_on_wheel': bool(hand_status.get('left_on_wheel', hand_status.get('left_in_roi', False))),
                'right_on_wheel': bool(hand_status.get('right_on_wheel', hand_status.get('right_in_roi', False))),
                'duration': float(hand_status.get('duration', 0.0)),
                'threshold_seconds': float(hand_status.get('threshold_seconds', 0.0) or 0.0),
            }
            overlay['wrists'] = hand_status.get('wrists', {})
            if hand_status.get('alert_level'):
                overlay['alert_regions'].append({
                    'kind': 'hands',
                    'bbox': overlay['wheel_roi'],
                    'severity': hand_status.get('alert_level'),
                })

        body_info = distraction.get('body_turn_info') or {}
        if body_info:
            left = body_info.get('left_shoulder')
            right = body_info.get('right_shoulder')
            if left and right:
                overlay['shoulders'] = {'left': left, 'right': right}
            center = body_info.get('center')
            baseline = body_info.get('baseline')
            if center and baseline:
                overlay['body_turn_vector'] = {
                    'start': baseline,
                    'end': center,
                    'active': bool(distraction.get('body_turn')),
                }

        return overlay

    @staticmethod
    def _head_direction(pitch, yaw):
        """将头部 pitch/yaw 转为页面展示用方向状态。"""
        pitch = float(pitch)
        yaw = float(yaw)
        if abs(yaw) > HEAD_TURN_YAW_THRESHOLD:
            return 'right' if yaw > 0 else 'left'
        if abs(pitch) > 15.0:
            return 'up' if pitch > 0 else 'down'
        return 'forward'

    @staticmethod
    def _points_to_list(points):
        if points is None:
            return []
        return [[float(p[0]), float(p[1])] for p in points]

    @staticmethod
    def _arrow_from_vector(origin, direction, length=100):
        dx, dy, dz = [float(v) for v in direction]
        norm = (dx * dx + dy * dy + dz * dz) ** 0.5
        if norm < 1e-8:
            return None
        ox, oy = float(origin[0]), float(origin[1])
        return {
            'start': [ox, oy],
            'end': [ox + (dx / norm) * length, oy - (dy / norm) * length],
        }

    @staticmethod
    def _head_pose_axes(origin, pose, scale=80):
        import math

        pitch, yaw, roll = [math.radians(float(v)) for v in pose]
        ox, oy = float(origin[0]), float(origin[1])
        axes = [
            ('x', math.cos(roll) * math.cos(yaw), math.sin(roll) * math.cos(pitch)),
            ('y', -math.sin(roll) * math.cos(yaw), math.cos(roll) * math.cos(pitch)),
            ('z', -math.cos(pitch) * math.sin(yaw), math.sin(pitch)),
        ]
        return [
            {
                'axis': name,
                'start': [ox, oy],
                'end': [ox + vx * scale, oy - vy * scale],
            }
            for name, vx, vy in axes
        ]

    @staticmethod
    def _pose_skeleton(keypoints):
        pairs = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16),
        ]
        segments = []
        for a, b in pairs:
            if a < len(keypoints) and b < len(keypoints):
                if keypoints[a][2] >= 0.3 and keypoints[b][2] >= 0.3:
                    segments.append([a, b])
        return segments

    @staticmethod
    def _object_severity(obj):
        class_id = int(obj.get('class_id', -1))
        if class_id in {1, 4}:
            return 'danger'
        if class_id in {2, 3, 5, 6, 7}:
            return 'warning'
        return 'info'

    def get_timeseries_data(self):
        """获取时序数据用于前端图表"""
        return {
            'ear_values': [d['v'] for d in self.ear_history],
            'mar_values': [d['v'] for d in self.mar_history],
            'pitch_values': [d['v'] for d in self.pitch_history],
            'yaw_values': [d['v'] for d in self.yaw_history],
            'roll_values': [d['v'] for d in self.roll_history],
            'ppg_signal': self.ppg_history,
            'timestamps': [d['t'] for d in self.ear_history] if self.ear_history else [],
        }

    def get_final_results(self):
        """获取最终检测结果"""
        physio_results = {}
        if self.rppg_monitor:
            physio_results = self.rppg_monitor.get_results()

        return {
            'summary': self.alert_manager.get_summary(),
            'fatigue': {
                'ear_values': [d['v'] for d in self.ear_history],
                'mar_values': [d['v'] for d in self.mar_history],
                'min_ear': min((d['v'] for d in self.ear_history), default=0),
                'avg_ear': (sum(d['v'] for d in self.ear_history) / len(self.ear_history)
                            if self.ear_history else 0),
                'blink_rate': 0,
                'eye_closure_events': [],
                'yawn_events': [],
            },
            'head_pose': {
                'pitch_values': [d['v'] for d in self.pitch_history],
                'yaw_values': [d['v'] for d in self.yaw_history],
                'roll_values': [d['v'] for d in self.roll_history],
            },
            'distraction': {
                'gaze_deviation_events': [],
                'objects_detected': [],
                'hands_off_wheel_events': [],
            },
            'physiological': physio_results,
            'alerts': self.alert_manager.get_all_alerts(),
        }

    def reset(self):
        """重置所有检测器状态"""
        if self.fatigue_detector:
            self.fatigue_detector.reset()
        if self.head_pose_estimator:
            self.head_pose_estimator.reset()
        if self.gaze_estimator:
            self.gaze_estimator.reset()
        if self.distraction_detector:
            self.distraction_detector.reset()
        if self.rppg_monitor:
            self.rppg_monitor.reset()
        self.alert_manager.reset()

        self.ear_history = []
        self.mar_history = []
        self.pitch_history = []
        self.yaw_history = []
        self.roll_history = []
        self.ppg_history = []
        self.frame_count = 0
        self.start_time = None
        self.last_face_result = None
