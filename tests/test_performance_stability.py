import io
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from flask import Flask


class _FakeAlertManager:
    def __init__(self):
        self.alerts = []

    def add_alert(self, alert):
        self.alerts.append(alert)

    def get_summary(self):
        return {"total_alerts": len(self.alerts), "overall_risk": "low"}

    def get_all_alerts(self):
        return list(self.alerts)

    def get_active_alerts(self):
        return list(self.alerts)

    def reset(self):
        self.alerts.clear()


class _CountingFaceDetector:
    def __init__(self):
        self.detect_calls = 0
        self.face_result = {"bbox": (0, 0, 10, 10)}

    def detect(self, frame):
        self.detect_calls += 1
        return self.face_result


def _minimal_pipeline():
    from processors.frame_pipeline import FramePipeline

    pipeline = FramePipeline.__new__(FramePipeline)
    pipeline.enable_modules = {}
    pipeline.face_detector = _CountingFaceDetector()
    pipeline.fatigue_detector = None
    pipeline.head_pose_estimator = None
    pipeline.gaze_estimator = None
    pipeline.distraction_detector = None
    pipeline.rppg_monitor = None
    pipeline.alert_manager = _FakeAlertManager()
    pipeline.ear_history = []
    pipeline.mar_history = []
    pipeline.pitch_history = []
    pipeline.yaw_history = []
    pipeline.roll_history = []
    pipeline.ppg_history = []
    pipeline.frame_count = 0
    pipeline.start_time = None
    return pipeline


class FrameReuseTests(unittest.TestCase):
    def test_pipeline_exposes_last_face_result_from_current_frame(self):
        pipeline = _minimal_pipeline()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        result = pipeline.process_frame(frame, timestamp=0.0)

        self.assertTrue(result["face_detected"])
        self.assertEqual(pipeline.face_detector.detect_calls, 1)
        self.assertIs(pipeline.last_face_result, pipeline.face_detector.face_result)

        pipeline.reset()
        self.assertIsNone(pipeline.last_face_result)

    def test_realtime_processor_returns_overlay_without_annotated_frame(self):
        from processors import realtime_processor as realtime_module

        class FakePipeline:
            def __init__(self, enable_modules=None):
                self.last_face_result = {"bbox": (0, 0, 10, 10)}

            def process_frame(self, frame, timestamp):
                return {
                    "face_detected": True,
                    "fatigue": {},
                    "head_pose": {},
                    "gaze": {},
                    "distraction": {},
                    "physio": {},
                    "alerts": [],
                    "summary": {},
                }

            def build_overlay(self, face_result, result, frame_shape):
                self.overlay_face_result = face_result
                return {"face_bbox": [0, 0, 10, 10]}

            def get_timeseries_data(self):
                return {}

            def reset(self):
                pass

        with patch.object(realtime_module, "FramePipeline", FakePipeline):
            processor = realtime_module.RealtimeProcessor()
            result = processor.process_frame(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual(processor.pipeline.overlay_face_result, {"bbox": (0, 0, 10, 10)})
        self.assertEqual(result["overlay"]["face_bbox"], [0, 0, 10, 10])
        self.assertNotIn("annotated_frame_b64", result)


class VideoProcessorTests(unittest.TestCase):
    def test_video_processing_reads_sequentially_and_reuses_face_result(self):
        from processors import video_processor as video_module

        class FakeCapture:
            def __init__(self, path):
                self.index = 0
                self.set_calls = []

            def isOpened(self):
                return True

            def get(self, prop):
                if prop == video_module.cv2.CAP_PROP_FRAME_COUNT:
                    return 6
                if prop == video_module.cv2.CAP_PROP_FPS:
                    return 30
                if prop == video_module.cv2.CAP_PROP_FRAME_WIDTH:
                    return 320
                if prop == video_module.cv2.CAP_PROP_FRAME_HEIGHT:
                    return 240
                return 0

            def set(self, prop, value):
                self.set_calls.append((prop, value))
                return True

            def read(self):
                if self.index >= 6:
                    return False, None
                self.index += 1
                return True, np.zeros((240, 320, 3), dtype=np.uint8)

            def release(self):
                pass

        class RaisingFaceDetector:
            def detect(self, frame):
                raise AssertionError("should reuse pipeline.last_face_result")

        class FakePipeline:
            def __init__(self):
                self.face_detector = RaisingFaceDetector()
                self.last_face_result = {"bbox": (0, 0, 10, 10)}
                self.alert_manager = _FakeAlertManager()

            def reset(self):
                pass

            def process_frame(self, frame, timestamp):
                return {
                    "face_detected": True,
                    "fatigue": {},
                    "head_pose": {},
                    "gaze": {},
                    "distraction": {},
                    "alerts": [],
                    "summary": {},
                }

            def get_annotated_frame(self, frame, face_result, result):
                self.annotated_face_result = face_result
                return frame

            def get_final_results(self):
                return {"summary": {}, "alerts": []}

        fake_capture = FakeCapture("demo.mp4")

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.object(video_module.cv2, "VideoCapture", return_value=fake_capture), \
             patch.object(video_module.cv2, "imencode", return_value=(True, np.array([1, 2, 3], dtype=np.uint8))), \
             patch.object(video_module, "BASE_DIR", tmpdir):
            processor = video_module.VideoProcessor(FakePipeline())
            result = processor.process_video("demo.mp4", "task-1")

        self.assertEqual(fake_capture.set_calls, [])
        self.assertEqual(result["video_info"]["processed_frames"], 3)
        self.assertEqual(processor.pipeline.annotated_face_result, {"bbox": (0, 0, 10, 10)})


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        import models.database as database_module

        self.database_module = database_module
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        database_module.DB_DIR = self.tmpdir.name
        database_module.DB_PATH = os.path.join(self.tmpdir.name, "history.db")
        database_module.DetectionHistoryDB._instance = None

    def test_get_stats_supports_optional_date_filters(self):
        db = self.database_module.DetectionHistoryDB()
        db.save_result("old", {
            "file_name": "old.jpg",
            "file_type": "image",
            "summary": {"overall_risk": "low", "fatigue_score": 90, "distraction_score": 95},
            "physiological": {"heart_rate": 70, "bp_systolic": 120, "bp_diastolic": 80},
        })
        db.save_result("new", {
            "file_name": "new.jpg",
            "file_type": "image",
            "summary": {"overall_risk": "high", "fatigue_score": 60, "distraction_score": 50},
            "physiological": {"heart_rate": 90, "bp_systolic": 130, "bp_diastolic": 85},
        })

        with sqlite3.connect(self.database_module.DB_PATH) as conn:
            conn.execute("UPDATE detection_history SET created_at = ? WHERE task_id = ?", ("2026-05-01 10:00:00", "old"))
            conn.execute("UPDATE detection_history SET created_at = ? WHERE task_id = ?", ("2026-05-31 10:00:00", "new"))
            conn.commit()

        all_stats = db.get_stats()
        day_stats = db.get_stats(start_date="2026-05-31", end_date="2026-05-31")
        from_stats = db.get_stats(start_date="2026-05-02")

        self.assertEqual(all_stats["total_tasks"], 2)
        self.assertEqual(day_stats["total_tasks"], 1)
        self.assertEqual(day_stats["risk_distribution"]["high"], 1)
        self.assertEqual(from_stats["total_tasks"], 1)
        self.assertEqual(day_stats["avg_heart_rate"], 90)


class RoutesAndAlertTests(unittest.TestCase):
    def test_save_to_database_uses_uploaded_filename_metadata(self):
        import models.database as database_module
        from web import routes_detect

        saved_payloads = []

        class FakeDB:
            def save_result(self, task_id, payload):
                saved_payloads.append((task_id, payload))

        with patch.object(routes_detect, "_get_task_meta", return_value={"filename": "driver.jpg", "file_type": "image"}), \
             patch.object(database_module, "DetectionHistoryDB", return_value=FakeDB()):
            routes_detect._save_to_database("task-1", {
                "summary": {"overall_risk": "low"},
                "alerts": [],
                "physiological": {},
            })

        self.assertEqual(saved_payloads[0][1]["file_name"], "driver.jpg")

    def test_alert_summary_scores_and_risk(self):
        from alert_manager import AlertManager

        manager = AlertManager()
        manager.add_alert({
            "source": "fatigue",
            "type": "eye_closure",
            "severity": "danger",
            "timestamp": 1.0,
            "message": "闭眼",
        })
        manager.add_alert({
            "source": "distraction",
            "type": "phone_usage",
            "severity": "warning",
            "timestamp": 2.0,
            "message": "手机",
        })

        summary = manager.get_summary()

        self.assertEqual(summary["total_alerts"], 2)
        self.assertEqual(summary["fatigue_score"], 70)
        self.assertEqual(summary["distraction_score"], 75)
        self.assertEqual(summary["overall_risk"], "high")


class CameraRealtimeTests(unittest.TestCase):
    def test_draw_alert_overlay_accepts_strings_and_dict_alerts(self):
        from utils.visualization import draw_alert_overlay

        cases = [
            ["闭眼告警"],
            [{"type": "gaze_deviation", "severity": "warning", "message": "视线偏离"}],
            [{"type": "gaze_deviation", "severity": "warning"}],
        ]

        for alerts in cases:
            with self.subTest(alerts=alerts):
                image = np.zeros((120, 220, 3), dtype=np.uint8)
                draw_alert_overlay(image, alerts)

    def test_camera_frame_with_dict_alert_overlay_does_not_return_500(self):
        from web import bp
        from web import routes_camera

        class FakeProcessor:
            def process_frame(self, frame):
                return {
                    "face_detected": True,
                    "fatigue": {},
                    "head_pose": {},
                    "gaze": {},
                    "distraction": {},
                    "physio": {},
                    "alerts": [{
                        "type": "gaze_deviation",
                        "severity": "warning",
                        "message": "视线偏离",
                    }],
                    "summary": {},
                    "overlay": {
                        "gaze_arrow": {"start": [10, 10], "end": [30, 10]},
                        "object_boxes": [],
                    },
                }

        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(bp)
        client = app.test_client()

        ok, buffer = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
        self.assertTrue(ok)

        with patch.object(routes_camera, "_get_or_create_processor", return_value=FakeProcessor()):
            response = client.post(
                "/camera/frame",
                data={
                    "session_id": "test-session",
                    "frame": (io.BytesIO(buffer.tobytes()), "frame.jpg"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertTrue(payload["face_detected"])
        self.assertIn("overlay", payload)
        self.assertNotIn("annotated_frame_b64", payload)


class NoTextVisualizationTests(unittest.TestCase):
    def test_visualization_helpers_do_not_draw_text_on_detection_frames(self):
        from utils import visualization

        image = np.zeros((160, 220, 3), dtype=np.uint8)
        with patch.object(visualization.cv2, "putText", side_effect=AssertionError("text is forbidden")):
            visualization.draw_eye_contours(
                image,
                np.array([[10, 10], [20, 10], [20, 18], [10, 18]]),
                np.array([[40, 10], [50, 10], [50, 18], [40, 18]]),
                ear_value=0.18,
            )
            visualization.draw_mouth_contour(
                image,
                np.array([[70, 70], [90, 70], [90, 90], [70, 90]]),
                mar_value=0.6,
            )
            visualization.draw_detection_boxes(
                image,
                [{"class_id": 1, "confidence": 0.9, "bbox": (5, 5, 40, 40)}],
                {1: "phone"},
            )
            visualization.draw_alert_overlay(
                image,
                [{"type": "phone_usage", "severity": "danger", "message": "手机"}],
            )


class CameraFrontendStaticTests(unittest.TestCase):
    def test_camera_template_uses_single_js_event_binding(self):
        html = Path("templates/camera.html").read_text(encoding="utf-8")

        self.assertNotIn('onclick="startCamera()"', html)
        self.assertNotIn('onclick="stopCamera()"', html)

    def test_camera_js_has_start_guard_and_adaptive_real_fps(self):
        js = Path("static/js/camera.js").read_text(encoding="utf-8")

        self.assertIn("let isStarting = false;", js)
        self.assertIn("MIN_SEND_INTERVAL_MS", js)
        self.assertIn("MAX_SEND_INTERVAL_MS", js)
        self.assertIn("currentSendIntervalMs", js)
        self.assertIn("calculateActualFps", js)
        self.assertNotIn("dom.fpsDisplay.textContent = (1000 / SEND_INTERVAL_MS).toFixed(0);", js)

    def test_camera_page_uses_single_video_with_canvas_overlay(self):
        html = Path("templates/camera.html").read_text(encoding="utf-8")
        js = Path("static/js/camera.js").read_text(encoding="utf-8")

        self.assertIn('id="camera-overlay-canvas"', html)
        self.assertNotIn('id="camera-output"', html)
        self.assertIn("drawOverlay", js)
        self.assertIn("drawVirtualSteeringWheel", js)
        self.assertIn("overlayCanvas", js)
        self.assertIn("enable-camera-distraction", html)
        self.assertIn("metric-hand-state", html)
        self.assertIn("metric-head-turn-state", html)
        self.assertNotIn("annotated_frame_b64", js)

    def test_camera_page_exposes_demo_mode_and_no_camera_samples(self):
        html = Path("templates/camera.html").read_text(encoding="utf-8")
        js = Path("static/js/camera.js").read_text(encoding="utf-8")

        self.assertIn("enable-demo-mode", html)
        self.assertIn("camera-demo-fallback", html)
        self.assertIn("btn-load-demo-samples", html)
        self.assertIn("demo_mode", js)
        self.assertIn("loadDemoSamples", js)
        self.assertIn("/api/demo/samples", js)

    def test_upload_js_is_safe_on_pages_without_upload_controls(self):
        js = Path("static/js/upload.js").read_text(encoding="utf-8")

        self.assertIn("if (dropZone && fileInput)", js)


class StartupAndDependencyStaticTests(unittest.TestCase):
    def test_app_prints_configured_site_url_on_startup(self):
        source = Path("app.py").read_text(encoding="utf-8")

        self.assertIn("FLASK_HOST", source)
        self.assertIn("FLASK_PORT", source)
        self.assertIn("网站地址:", source)

    def test_requirements_include_face_recognition_and_roboflow(self):
        requirements = Path("requirements.txt").read_text(encoding="utf-8")

        self.assertIn("face-recognition", requirements)
        self.assertIn("roboflow", requirements)


class YoloDatasetScriptTests(unittest.TestCase):
    def test_roboflow_dataset_import_script_documents_target_model_and_mapping(self):
        script = Path("yolo/roboflow_driver_dataset.py").read_text(encoding="utf-8")

        self.assertIn("ROBOFLOW_API_KEY", script)
        self.assertIn("yolo_driver_state.pt", script)
        self.assertIn("driver using phone", script)
        self.assertIn("driver smoking", script)
        self.assertIn("driver drowsy", script)

    def test_real_dataset_entry_lists_and_dry_runs(self):
        script = Path("yolo/train_real_datasets.py").read_text(encoding="utf-8")

        self.assertIn("Driver fatigue and distraction", script)
        self.assertIn("Novel Driver Distractions Dataset With Low Lighting Support", script)
        self.assertIn("n7i5x9/driver-drowsiness-dataset", script)
        self.assertIn("--dry-run", script)
        self.assertIn("yolo_driver_state.pt", script)
        self.assertIn("driver_distraction_cls.pt", script)

    def test_mendeley_dataset_prepare_and_classifier_scripts_exist(self):
        prepare = Path("yolo/mendeley_distraction_dataset.py").read_text(encoding="utf-8")
        train = Path("yolo/train_mendeley_distraction_cls.py").read_text(encoding="utf-8")

        self.assertIn("ykmr99nrsg", prepare)
        self.assertIn("CC BY-NC 3.0", prepare)
        self.assertIn("safe_driving", prepare)
        self.assertIn("talking_to_passenger", prepare)
        self.assertIn("YOLO_DRIVER_CLASSIFIER_MODEL", train)

    def test_camera_start_makes_preview_video_visible(self):
        camera_js = Path("static/js/camera.js").read_text(encoding="utf-8")

        self.assertIn("dom.video.style.display = 'block'", camera_js)
        self.assertIn("dom.overlayCanvas.style.display = 'block'", camera_js)

    def test_models_api_surfaces_driver_classifier(self):
        route = Path("web/routes_main.py").read_text(encoding="utf-8")

        self.assertIn("driver_distraction_cls", route)
        self.assertIn("YOLO_DRIVER_CLASSIFIER_MODEL", route)


class YoloModelCacheTests(unittest.TestCase):
    def test_distraction_detector_instances_share_yolo_models(self):
        import detectors.distraction as distraction_module

        class FakeYOLO:
            load_count = 0

            def __init__(self, path):
                FakeYOLO.load_count += 1
                self.path = path

        with patch.object(distraction_module.os.path, "exists", return_value=True), \
             patch.object(distraction_module, "YOLO", FakeYOLO):
            first = distraction_module.DistractionDetector()
            second = distraction_module.DistractionDetector()

        self.assertIs(first.handheld_model, second.handheld_model)
        self.assertIs(first.driver_state_model, second.driver_state_model)
        self.assertIs(first.driver_classifier_model, second.driver_classifier_model)
        self.assertIs(first.steering_hand_model, second.steering_hand_model)
        self.assertIs(first.pose_model, second.pose_model)
        self.assertEqual(FakeYOLO.load_count, 5)

    def test_single_and_double_hand_off_have_separate_alert_levels(self):
        import detectors.distraction as distraction_module

        with patch.object(distraction_module.DistractionDetector, "_init_models", lambda self: None):
            detector = distraction_module.DistractionDetector()

        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[distraction_module.KP_LEFT_WRIST] = [100, 100, 0.9]
        keypoints[distraction_module.KP_RIGHT_WRIST] = [320, 360, 0.9]

        status = detector.check_hands_on_wheel(keypoints, 640, 480, 0.0)
        self.assertEqual(status["state"], "left_off")
        self.assertEqual(status["threshold_seconds"], 8.0)
        self.assertIn("wheel", status)
        status = detector.check_hands_on_wheel(keypoints, 640, 480, 8.1)
        self.assertEqual(status["alert_level"], "warning")

        detector.reset()
        keypoints[distraction_module.KP_LEFT_WRIST] = [100, 100, 0.9]
        keypoints[distraction_module.KP_RIGHT_WRIST] = [100, 120, 0.9]
        status = detector.check_hands_on_wheel(keypoints, 640, 480, 0.0)
        self.assertEqual(status["state"], "both_off")
        self.assertEqual(status["threshold_seconds"], 5.0)
        status = detector.check_hands_on_wheel(keypoints, 640, 480, 5.1)
        self.assertEqual(status["alert_level"], "danger")

    def test_demo_mode_shortens_hand_off_durations_only_for_demo(self):
        import detectors.distraction as distraction_module

        with patch.object(distraction_module.DistractionDetector, "_init_models", lambda self: None):
            detector = distraction_module.DistractionDetector(demo_mode=True)

        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[distraction_module.KP_LEFT_WRIST] = [100, 100, 0.9]
        keypoints[distraction_module.KP_RIGHT_WRIST] = [320, 360, 0.9]

        status = detector.check_hands_on_wheel(keypoints, 640, 480, 0.0)
        self.assertEqual(status["state"], "left_off")
        self.assertEqual(status["threshold_seconds"], 2.0)
        self.assertTrue(status["demo_mode"])
        status = detector.check_hands_on_wheel(keypoints, 640, 480, 2.1)
        self.assertEqual(status["alert_level"], "warning")

        detector.reset()
        keypoints[distraction_module.KP_RIGHT_WRIST] = [100, 120, 0.9]
        status = detector.check_hands_on_wheel(keypoints, 640, 480, 0.0)
        self.assertEqual(status["threshold_seconds"], 1.5)
        status = detector.check_hands_on_wheel(keypoints, 640, 480, 1.6)
        self.assertEqual(status["alert_level"], "danger")

    def test_head_turn_uses_sustained_threshold(self):
        from detectors.head_pose import HeadPoseEstimator

        estimator = HeadPoseEstimator()
        self.assertIsNone(estimator.check_head_turn(40.0, 0.0))
        self.assertIsNone(estimator.check_head_turn(40.0, 1.0))
        alert = estimator.check_head_turn(40.0, 2.1)
        self.assertIsNotNone(alert)
        self.assertEqual(alert["type"], "head_turn")
        self.assertEqual(alert["severity"], "warning")

        estimator.reset()
        estimator.check_head_turn(60.0, 0.0)
        alert = estimator.check_head_turn(60.0, 2.1)
        self.assertEqual(alert["severity"], "danger")

    def test_demo_mode_shortens_head_turn_duration(self):
        from detectors.head_pose import HeadPoseEstimator

        estimator = HeadPoseEstimator(demo_mode=True)
        self.assertIsNone(estimator.check_head_turn(40.0, 0.0))
        alert = estimator.check_head_turn(40.0, 1.1)
        self.assertIsNotNone(alert)
        self.assertEqual(alert["metadata"]["threshold_seconds"], 1.0)
        self.assertTrue(alert["metadata"]["demo_mode"])


class DemoRoutesTests(unittest.TestCase):
    def test_demo_samples_api_returns_upload_route_and_safe_urls(self):
        from web import bp

        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(bp)
        client = app.test_client()

        response = client.get("/api/demo/samples")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["upload_url"], "/#upload-section")
        self.assertIn("samples", payload)
        for sample in payload["samples"]:
            self.assertTrue(sample["url"].startswith("/demo/sample/dataset/"))

        if payload["samples"]:
            sample_response = client.get(payload["samples"][0]["url"])
            try:
                self.assertEqual(sample_response.status_code, 200)
            finally:
                sample_response.close()


class ApiFlowTests(unittest.TestCase):
    def setUp(self):
        from web import bp
        from web import routes_detect

        self.routes_detect = routes_detect
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

        app = Flask(__name__)
        app.config["TESTING"] = True
        app.config["UPLOAD_FOLDER"] = os.path.join(self.tmpdir.name, "uploads")
        app.config["OUTPUT_FOLDER"] = os.path.join(self.tmpdir.name, "outputs")
        app.register_blueprint(bp)
        self.client = app.test_client()

    def _wait_for_done(self, task_id):
        for _ in range(50):
            resp = self.client.get(f"/api/status/{task_id}")
            data = resp.get_json()
            if data.get("status") in {"done", "error"}:
                return data
            time.sleep(0.02)
        self.fail(f"task did not finish: {task_id}")

    def test_upload_detect_and_fetch_image_result(self):
        class FakeFramePipeline:
            def __init__(self, enable_modules=None):
                self.enable_modules = enable_modules

        class FakeImageProcessor:
            def __init__(self, pipeline):
                self.pipeline = pipeline

            def process_image(self, image_path, task_id):
                return {
                    "task_id": task_id,
                    "annotated_image_base64": "abc",
                    "face_detected": True,
                    "summary": {"overall_risk": "low"},
                    "alerts": [],
                }

        with patch("processors.frame_pipeline.FramePipeline", FakeFramePipeline), \
             patch("processors.video_processor.ImageProcessor", FakeImageProcessor), \
             patch.object(self.routes_detect, "_save_to_database"):
            upload_resp = self.client.post(
                "/upload",
                data={"file": (io.BytesIO(b"fake image"), "driver.jpg"), "mode": "auto"},
                content_type="multipart/form-data",
            )
            self.assertEqual(upload_resp.status_code, 200)
            task_id = upload_resp.get_json()["task_id"]

            detect_resp = self.client.post("/detect/auto", json={"task_id": task_id})
            self.assertEqual(detect_resp.status_code, 200)

            status = self._wait_for_done(task_id)
            self.assertEqual(status["status"], "done")
            result_resp = self.client.get(f"/api/results/{task_id}")
            self.assertEqual(result_resp.status_code, 200)
            self.assertTrue(result_resp.get_json()["face_detected"])

        self.assertNotIn(task_id, self.routes_detect._running_tasks)

    def test_upload_detect_and_fetch_video_result(self):
        class FakeFramePipeline:
            def __init__(self, enable_modules=None):
                self.enable_modules = enable_modules

        class FakeVideoProcessor:
            def __init__(self, pipeline):
                self.pipeline = pipeline

            def process_video(self, video_path, task_id, progress_callback=None, output_video=False):
                if progress_callback:
                    progress_callback({
                        "task_id": task_id,
                        "status": "processing",
                        "progress": 50,
                        "current_frame": 1,
                        "total_frames": 1,
                        "current_alerts": [],
                        "preview_frame": None,
                        "summary": {"overall_risk": "low"},
                    })
                return {
                    "summary": {"overall_risk": "low"},
                    "fatigue": {},
                    "head_pose": {},
                    "distraction": {},
                    "physiological": {},
                    "alerts": [],
                    "annotated_frames": [],
                    "video_info": {"processed_frames": 1},
                }

        with patch("processors.frame_pipeline.FramePipeline", FakeFramePipeline), \
             patch("processors.video_processor.VideoProcessor", FakeVideoProcessor), \
             patch.object(self.routes_detect, "_save_to_database"):
            upload_resp = self.client.post(
                "/upload",
                data={"file": (io.BytesIO(b"fake video"), "demo.mp4"), "mode": "auto"},
                content_type="multipart/form-data",
            )
            self.assertEqual(upload_resp.status_code, 200)
            task_id = upload_resp.get_json()["task_id"]

            detect_resp = self.client.post("/detect/auto", json={"task_id": task_id})
            self.assertEqual(detect_resp.status_code, 200)

            status = self._wait_for_done(task_id)
            self.assertEqual(status["status"], "done")
            result_resp = self.client.get(f"/api/results/{task_id}")
            self.assertEqual(result_resp.status_code, 200)
            self.assertEqual(result_resp.get_json()["video_info"]["processed_frames"], 1)

        self.assertNotIn(task_id, self.routes_detect._running_tasks)


if __name__ == "__main__":
    unittest.main()
