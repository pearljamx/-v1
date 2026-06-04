/**
 * camera.js — 摄像头实时检测逻辑
 * =================================
 * 通过 getUserMedia 捕获摄像头，自适应发送帧到后端，
 * 本地视频保持流畅预览，后端结果更新检测指标和标注小窗。
 */

// ===== 状态变量 =====
let mediaStream = null;
let videoElement = null;
let canvasElement = null;
let canvasCtx = null;
let overlayCanvas = null;
let overlayCtx = null;
let sessionId = null;
let sendTimer = null;
let inFlightController = null;
let isRunning = false;
let isStarting = false;
let frameCount = 0;
let totalLatency = 0;

// 配置
const CAMERA_WIDTH = 640;           // 浏览器预览请求宽度
const CAMERA_HEIGHT = 480;          // 浏览器预览请求高度
const CAPTURE_WIDTH = 512;          // 后端检测帧宽度
const CAPTURE_HEIGHT = 384;         // 后端检测帧高度
const MIN_SEND_INTERVAL_MS = 100;   // 最快约10 FPS
const BASE_SEND_INTERVAL_MS = 120;  // 默认约8 FPS
const MAX_SEND_INTERVAL_MS = 350;   // 过载时自动降到约3 FPS
const FETCH_TIMEOUT_MS = 3000;      // 单帧请求超时
const MAX_CONSECUTIVE_ERRORS = 10;  // 连续错误阈值

let consecutiveErrors = 0;          // 连续错误计数
let currentSendIntervalMs = BASE_SEND_INTERVAL_MS;
const JPEG_QUALITY = 0.55;          // JPEG压缩质量
const frameTimestamps = [];

// 时序数据
const earHistory = [];
const marHistory = [];
const maxHistoryPoints = 100;

// ===== DOM元素缓存 =====
const dom = {};

function cacheDomElements() {
    dom.video = document.getElementById('camera-video');
    dom.canvas = document.getElementById('camera-canvas');
    dom.overlayCanvas = document.getElementById('camera-overlay-canvas');
    dom.startBtn = document.getElementById('btn-start-camera');
    dom.stopBtn = document.getElementById('btn-stop-camera');
    dom.deviceSelect = document.getElementById('camera-device-select');
    dom.statusDot = document.getElementById('live-status-dot');
    dom.statusText = document.getElementById('live-status-text');
    dom.faceStatus = document.getElementById('metric-face-status');
    dom.earValue = document.getElementById('metric-ear');
    dom.marValue = document.getElementById('metric-mar');
    dom.blinkRate = document.getElementById('metric-blink-rate');
    dom.headPitch = document.getElementById('metric-head-pitch');
    dom.headYaw = document.getElementById('metric-head-yaw');
    dom.gazeAngle = document.getElementById('metric-gaze-angle');
    dom.lightingLevel = document.getElementById('metric-lighting-level');
    dom.brightnessValue = document.getElementById('metric-brightness-value');
    dom.adaptiveEar = document.getElementById('metric-adaptive-ear');
    dom.adaptiveMar = document.getElementById('metric-adaptive-mar');
    dom.alertList = document.getElementById('active-alerts-list');
    dom.fpsDisplay = document.getElementById('fps-display');
    dom.latencyDisplay = document.getElementById('latency-display');
    dom.cameraPlaceholder = document.getElementById('camera-placeholder');
    dom.cameraContainer = document.getElementById('camera-container');
}

// ===== 摄像头初始化 =====
async function enumerateDevices() {
    try {
        // 先请求权限以获取设备标签
        const tempStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        tempStream.getTracks().forEach(t => t.stop());
    } catch (e) {
        console.warn('[Camera] 权限预检失败:', e.message);
        // 权限被拒绝，仍尝试列出设备（可能没有标签名）
    }

    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');

        if (dom.deviceSelect) {
            if (videoDevices.length === 0) {
                dom.deviceSelect.innerHTML = '<option value="">未检测到摄像头</option>';
            } else {
                dom.deviceSelect.innerHTML = videoDevices.map((d, i) =>
                    `<option value="${d.deviceId}">${d.label || '摄像头 ' + (i + 1)}</option>`
                ).join('');
            }
        }
        console.log('[Camera] 找到', videoDevices.length, '个摄像头');
        return videoDevices;
    } catch (e) {
        console.error('[Camera] 枚举设备失败:', e);
        return [];
    }
}

async function startCamera() {
    if (isRunning || isStarting) return;
    isStarting = true;

    cacheDomElements();
    if (dom.startBtn) {
        dom.startBtn.disabled = true;
    }

    // 检查浏览器支持
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showCameraError('您的浏览器不支持摄像头访问，请使用最新版Chrome/Firefox/Edge，并通过localhost或HTTPS访问');
        isStarting = false;
        if (dom.startBtn) {
            dom.startBtn.disabled = false;
        }
        return;
    }

    const deviceId = dom.deviceSelect?.value || undefined;
    const constraints = {
        video: {
            deviceId: deviceId ? { exact: deviceId } : undefined,
            width: { ideal: CAMERA_WIDTH },
            height: { ideal: CAMERA_HEIGHT },
            facingMode: 'user',
        },
        audio: false,
    };

    try {
        console.log('[Camera] 正在请求摄像头...', constraints);
        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        console.log('[Camera] 摄像头已获取, tracks:', mediaStream.getVideoTracks().length);

        // 设置video元素
        videoElement = dom.video || videoElement;
        if (!videoElement) {
            videoElement = document.createElement('video');
            videoElement.setAttribute('playsinline', '');
            videoElement.setAttribute('autoplay', '');
            videoElement.muted = true;
            dom.cameraContainer?.appendChild(videoElement);
        }
        videoElement.srcObject = mediaStream;

        // 设置canvas
        if (!canvasElement) {
            canvasElement = document.createElement('canvas');
            canvasElement.width = CAPTURE_WIDTH;
            canvasElement.height = CAPTURE_HEIGHT;
            canvasCtx = canvasElement.getContext('2d', { willReadFrequently: true });
        }
        overlayCanvas = dom.overlayCanvas || overlayCanvas;
        if (overlayCanvas && !overlayCtx) {
            overlayCtx = overlayCanvas.getContext('2d');
        }

        // 等待video准备好
        await videoElement.play();
        console.log('[Camera] Video playing, readyState:', videoElement.readyState);

        // 生成session ID
        sessionId = 'cam-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8);

        // 显示本地视频作为唯一主画面，canvas负责纯图形检测叠加
        dom.video.style.display = '';
        if (dom.overlayCanvas) {
            dom.overlayCanvas.style.display = '';
            resizeOverlayCanvas();
        }
        dom.cameraPlaceholder.style.display = 'none';

        // 更新UI
        isRunning = true;
        isStarting = false;
        frameCount = 0;
        totalLatency = 0;
        consecutiveErrors = 0;
        currentSendIntervalMs = BASE_SEND_INTERVAL_MS;
        frameTimestamps.length = 0;
        earHistory.length = 0;
        marHistory.length = 0;

        dom.startBtn.classList.add('hide-mobile');
        dom.startBtn.disabled = false;
        dom.stopBtn.style.display = '';
        dom.statusDot.className = 'live-dot active';
        dom.statusText.textContent = '检测中';

        // 开始发送循环
        sendFrame();

        console.log('[Camera] 摄像头已启动, session:', sessionId);

    } catch (error) {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        if (videoElement) {
            videoElement.pause();
            videoElement.srcObject = null;
        }
        console.error('[Camera] 启动失败:', error.name, error.message);
        let msg = '无法访问摄像头: ' + error.message;
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            msg = '摄像头权限被拒绝，请在浏览器设置中允许访问摄像头，并使用localhost或HTTPS访问';
        } else if (error.name === 'NotFoundError') {
            msg = '未检测到摄像头设备，请确认摄像头已连接';
        } else if (error.name === 'NotReadableError') {
            msg = '摄像头被其他应用占用，请关闭其他使用摄像头的程序';
        }
        showCameraError(msg);
        isStarting = false;
        if (dom.startBtn) {
            dom.startBtn.disabled = false;
        }
    }
}

function stopCamera() {
    isRunning = false;
    isStarting = false;

    // 停止发送
    if (sendTimer) {
        clearTimeout(sendTimer);
        sendTimer = null;
    }

    if (inFlightController) {
        inFlightController.abort();
        inFlightController = null;
    }

    // 停止媒体流
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    if (videoElement) {
        videoElement.pause();
        videoElement.srcObject = null;
    }

    // 通知服务器释放session
    if (sessionId) {
        fetch('/camera/session/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId }),
        }).catch(() => {});
    }
    sessionId = null;

    // 更新UI
    dom.startBtn.disabled = false;
    dom.startBtn.classList.remove('hide-mobile');
    dom.stopBtn.style.display = 'none';
    if (dom.video) {
        dom.video.style.display = 'none';
    }
    if (dom.overlayCanvas) {
        clearOverlay();
        dom.overlayCanvas.style.display = 'none';
    }
    if (dom.cameraPlaceholder) {
        dom.cameraPlaceholder.style.display = 'flex';
    }
    dom.statusDot.className = 'live-dot inactive';
    dom.statusText.textContent = '已停止';
    dom.faceStatus.textContent = '--';
    dom.faceStatus.className = 'metric-value';

    console.log('[Camera] 摄像头已停止, 共处理', frameCount, '帧');
}

// ===== 帧发送循环 =====
async function sendFrame() {
    if (!isRunning) return;

    const startTime = performance.now();
    let timeoutId = null;

    try {
        // 检查video是否就绪
        if (!videoElement || videoElement.readyState < 2) {
            scheduleNext(startTime);
            return;
        }

        // 从video绘制到canvas
        canvasCtx.drawImage(videoElement, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);

        // Canvas转Blob
        const blob = await new Promise((resolve) => {
            canvasElement.toBlob(resolve, 'image/jpeg', JPEG_QUALITY);
        });

        if (!blob || !isRunning) return;

        // 构建请求
        const formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');
        formData.append('session_id', sessionId);

        // 发送到服务器（带超时）
        inFlightController = new AbortController();
        timeoutId = setTimeout(() => inFlightController.abort(), FETCH_TIMEOUT_MS);

        const resp = await fetch('/camera/frame', {
            method: 'POST',
            body: formData,
            signal: inFlightController.signal,
        });

        clearTimeout(timeoutId);
        timeoutId = null;
        inFlightController = null;

        if (!resp.ok) {
            consecutiveErrors++;
            increaseSendInterval();
            console.warn('[Camera] 服务器响应异常:', resp.status, '(连续错误:', consecutiveErrors, ')');
            if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
                updateCameraStatus('error', '服务器异常，正在重试...');
            }
            scheduleNext(startTime);
            return;
        }

        const data = await resp.json();

        // 成功响应，重置错误计数
        consecutiveErrors = 0;
        updateCameraStatus('active', '实时检测中');
        frameCount++;

        // 计算延迟
        const latency = performance.now() - startTime;
        totalLatency += latency;
        updateAdaptiveInterval(latency);

        // 更新UI
        updateCameraUI(data, latency);

    } catch (error) {
        if (timeoutId) {
            clearTimeout(timeoutId);
        }
        inFlightController = null;
        if (!isRunning && error.name === 'AbortError') {
            return;
        }
        consecutiveErrors++;
        increaseSendInterval();
        console.error('[Camera] 帧发送失败:', error.message, '(连续错误:', consecutiveErrors, ')');

        if (error.name === 'AbortError') {
            console.warn('[Camera] 请求超时，服务器可能过载');
            updateCameraStatus('warning', '服务器响应慢...');
        } else if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
            updateCameraStatus('error', '连接断开，正在重连...');
        }

        // 发生错误时增加间隔，避免雪崩
        const backoffDelay = Math.min(consecutiveErrors * 200, 2000);
        scheduleNext(startTime, backoffDelay);
        return;
    }

    scheduleNext(startTime);
}

// ===== 摄像头状态更新 =====
function updateCameraStatus(status, message) {
    const dot = dom.statusDot || document.getElementById('live-status-dot');
    const text = dom.statusText || document.getElementById('live-status-text');
    if (dot) {
        dot.className = 'live-dot ' + status;
    }
    if (text) {
        text.textContent = message;
    }
}

function updateAdaptiveInterval(latency) {
    if (latency < 80 && currentSendIntervalMs > MIN_SEND_INTERVAL_MS) {
        currentSendIntervalMs = Math.max(MIN_SEND_INTERVAL_MS, currentSendIntervalMs - 10);
    } else if (latency > 160) {
        increaseSendInterval();
    }
}

function increaseSendInterval() {
    currentSendIntervalMs = Math.min(MAX_SEND_INTERVAL_MS, currentSendIntervalMs + 40);
}

function calculateActualFps(now) {
    frameTimestamps.push(now);
    while (frameTimestamps.length > 0 && now - frameTimestamps[0] > 1000) {
        frameTimestamps.shift();
    }
    return frameTimestamps.length;
}

function scheduleNext(startTime, extraDelay = 0) {
    if (!isRunning) return;

    const elapsed = performance.now() - startTime;
    const delay = Math.max(20, currentSendIntervalMs - elapsed) + extraDelay;
    sendTimer = setTimeout(sendFrame, delay);
}

// ===== UI更新 =====
function updateCameraUI(data, latency) {
    drawOverlay(data.overlay || {});

    // 人脸状态
    if (dom.faceStatus) {
        dom.faceStatus.textContent = data.face_detected ? '已检测到' : '未检测到';
        dom.faceStatus.className = 'metric-value ' + (data.face_detected ? 'success' : 'danger');
    }

    // 疲劳指标
    if (data.fatigue) {
        dom.earValue.textContent = data.fatigue.ear ? data.fatigue.ear.toFixed(3) : '--';
        dom.marValue.textContent = data.fatigue.mar ? data.fatigue.mar.toFixed(3) : '--';
        dom.blinkRate.textContent = data.fatigue.blink_rate ? data.fatigue.blink_rate.toFixed(1) : '--';

        // 颜色指示
        dom.earValue.className = 'metric-value ' + (data.fatigue.ear < 0.2 ? 'danger' : 'success');
        dom.marValue.className = 'metric-value ' + (data.fatigue.mar > 0.5 ? 'warning' : 'success');

        // 记录时序
        earHistory.push(data.fatigue.ear || 0);
        marHistory.push(data.fatigue.mar || 0);
        if (earHistory.length > maxHistoryPoints) earHistory.shift();
        if (marHistory.length > maxHistoryPoints) marHistory.shift();
    }

    // 头姿
    if (data.head_pose) {
        dom.headPitch.textContent = data.head_pose.pitch ? data.head_pose.pitch.toFixed(1) + '°' : '--';
        dom.headYaw.textContent = data.head_pose.yaw ? data.head_pose.yaw.toFixed(1) + '°' : '--';

        const pitchAbs = Math.abs(data.head_pose.pitch || 0);
        dom.headPitch.className = 'metric-value ' + (pitchAbs > 15 ? 'danger' : 'success');
    }

    // 视线
    if (data.gaze) {
        dom.gazeAngle.textContent = data.gaze.gaze_angle ? data.gaze.gaze_angle.toFixed(1) + '°' : '--';
        dom.gazeAngle.className = 'metric-value ' +
            (data.gaze.gaze_angle > 30 ? 'warning' : 'success');
    }

    // 环境光照
    if (data.lighting) {
        const levelMap = {
            'dark': { label: '🌙 暗光', cls: 'danger', desc: '(阈值已放宽)' },
            'dim': { label: '🌆 昏暗', cls: 'warning', desc: '(阈值适度放宽)' },
            'normal': { label: '☀️ 正常', cls: 'success', desc: '(默认阈值)' },
            'bright': { label: '🔆 明亮', cls: 'success', desc: '(默认阈值)' },
        };
        const info = levelMap[data.lighting.lighting_level] || { label: data.lighting.lighting_level, cls: '', desc: '' };

        dom.lightingLevel.textContent = info.label + ' ' + info.desc;
        dom.lightingLevel.className = 'metric-value ' + info.cls;
        dom.brightnessValue.textContent = data.lighting.brightness.toFixed(1);
        dom.adaptiveEar.textContent = data.lighting.ear_threshold.toFixed(3);
        dom.adaptiveMar.textContent = data.lighting.mar_threshold.toFixed(3);
    }

    // 活跃告警
    updateAlertList(data.alerts || []);

    // FPS和延迟
    if (dom.fpsDisplay) {
        const actualFps = calculateActualFps(performance.now());
        dom.latencyDisplay.textContent = latency.toFixed(0) + 'ms';
        dom.fpsDisplay.textContent = actualFps.toFixed(0);
    }
}

// ===== 纯图形叠加绘制（检测画面禁止绘制任何文字） =====
function resizeOverlayCanvas() {
    if (!overlayCanvas || !dom.cameraContainer) return;
    const rect = dom.cameraContainer.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    overlayCanvas.width = Math.max(1, Math.round(rect.width * dpr));
    overlayCanvas.height = Math.max(1, Math.round(rect.height * dpr));
    overlayCanvas.style.width = rect.width + 'px';
    overlayCanvas.style.height = rect.height + 'px';
    if (overlayCtx) {
        overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
}

function clearOverlay() {
    if (!overlayCtx || !overlayCanvas) return;
    const rect = overlayCanvas.getBoundingClientRect();
    overlayCtx.clearRect(0, 0, rect.width, rect.height);
}

function overlayTransform(overlay) {
    const frame = overlay.frame_size || {};
    const srcW = frame.width || CAPTURE_WIDTH;
    const srcH = frame.height || CAPTURE_HEIGHT;
    const rect = dom.cameraContainer.getBoundingClientRect();
    const scale = Math.min(rect.width / srcW, rect.height / srcH);
    const drawW = srcW * scale;
    const drawH = srcH * scale;
    return {
        scale,
        ox: (rect.width - drawW) / 2,
        oy: (rect.height - drawH) / 2,
    };
}

function mapPoint(pt, t) {
    return [t.ox + pt[0] * t.scale, t.oy + pt[1] * t.scale];
}

function colorForSeverity(severity) {
    if (severity === 'danger') return '#ff4560';
    if (severity === 'warning') return '#ffb547';
    if (severity === 'success') return '#22c55e';
    return '#4d9fff';
}

function drawOverlay(overlay) {
    if (!overlayCanvas || !overlayCtx) return;
    resizeOverlayCanvas();
    clearOverlay();
    const t = overlayTransform(overlay);

    (overlay.alert_regions || []).forEach(region => {
        if (region.bbox) drawBox(region.bbox, t, colorForSeverity(region.severity), 3, true);
    });
    if (overlay.face_bbox) drawBox(overlay.face_bbox, t, '#22c55e', 2, false);
    (overlay.eye_contours || []).forEach(points => drawPolyline(points, t, '#4dd0ff', true, 2));
    drawPolyline(overlay.mouth_contour || [], t, '#ffd166', true, 2);
    (overlay.object_boxes || []).forEach(box => drawBox(box.bbox, t, colorForSeverity(box.severity), 2, false));
    if (overlay.wheel_roi) drawDashedBox(overlay.wheel_roi, t, '#ff4560');
    drawPose(overlay, t);
    (overlay.head_pose_axes || []).forEach(axis => drawLine(axis.start, axis.end, t, axisColor(axis.axis), 2));
    if (overlay.gaze_arrow) drawArrow(overlay.gaze_arrow.start, overlay.gaze_arrow.end, t, '#ff9f1c', 3);
    if (overlay.body_turn_vector) {
        drawArrow(overlay.body_turn_vector.start, overlay.body_turn_vector.end, t,
            overlay.body_turn_vector.active ? '#ff4560' : '#ffb547', 3);
    }
}

function axisColor(axis) {
    if (axis === 'x') return '#ff4560';
    if (axis === 'y') return '#22c55e';
    return '#4d9fff';
}

function drawBox(bbox, t, color, width = 2, translucent = false) {
    const [x1, y1] = mapPoint([bbox[0], bbox[1]], t);
    const [x2, y2] = mapPoint([bbox[2], bbox[3]], t);
    overlayCtx.save();
    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth = width;
    if (translucent) {
        overlayCtx.fillStyle = color + '26';
        overlayCtx.fillRect(x1, y1, x2 - x1, y2 - y1);
    }
    overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const corner = Math.max(10, Math.min(28, (x2 - x1) * 0.18));
    overlayCtx.lineWidth = width + 1;
    [[x1, y1, x1 + corner, y1], [x1, y1, x1, y1 + corner],
     [x2, y1, x2 - corner, y1], [x2, y1, x2, y1 + corner],
     [x1, y2, x1 + corner, y2], [x1, y2, x1, y2 - corner],
     [x2, y2, x2 - corner, y2], [x2, y2, x2, y2 - corner]]
        .forEach(line => drawRawLine(line[0], line[1], line[2], line[3]));
    overlayCtx.restore();
}

function drawDashedBox(bbox, t, color) {
    const [x1, y1] = mapPoint([bbox[0], bbox[1]], t);
    const [x2, y2] = mapPoint([bbox[2], bbox[3]], t);
    overlayCtx.save();
    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth = 2;
    overlayCtx.setLineDash([10, 8]);
    overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    overlayCtx.restore();
}

function drawPolyline(points, t, color, closed = false, width = 2) {
    if (!points || points.length < 2) return;
    overlayCtx.save();
    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth = width;
    overlayCtx.beginPath();
    const first = mapPoint(points[0], t);
    overlayCtx.moveTo(first[0], first[1]);
    points.slice(1).forEach(pt => {
        const [x, y] = mapPoint(pt, t);
        overlayCtx.lineTo(x, y);
    });
    if (closed) overlayCtx.closePath();
    overlayCtx.stroke();
    overlayCtx.restore();
}

function drawPose(overlay, t) {
    const keypoints = overlay.pose_keypoints || [];
    (overlay.pose_skeleton || []).forEach(pair => {
        drawLine(keypoints[pair[0]], keypoints[pair[1]], t, '#7dd3fc', 2);
    });
    keypoints.forEach(kp => {
        if (kp[2] >= 0.3) drawPoint(kp, t, '#7dd3fc', 3);
    });
    Object.values(overlay.wrists || {}).forEach(kp => drawPoint(kp, t, '#ff4560', 5));
    Object.values(overlay.shoulders || {}).forEach(kp => drawPoint(kp, t, '#ffb547', 5));
}

function drawPoint(pt, t, color, radius) {
    const [x, y] = mapPoint(pt, t);
    overlayCtx.save();
    overlayCtx.fillStyle = color;
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, radius, 0, Math.PI * 2);
    overlayCtx.fill();
    overlayCtx.restore();
}

function drawLine(start, end, t, color, width = 2) {
    if (!start || !end) return;
    const [x1, y1] = mapPoint(start, t);
    const [x2, y2] = mapPoint(end, t);
    overlayCtx.save();
    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth = width;
    drawRawLine(x1, y1, x2, y2);
    overlayCtx.restore();
}

function drawRawLine(x1, y1, x2, y2) {
    overlayCtx.beginPath();
    overlayCtx.moveTo(x1, y1);
    overlayCtx.lineTo(x2, y2);
    overlayCtx.stroke();
}

function drawArrow(start, end, t, color, width = 2) {
    const [x1, y1] = mapPoint(start, t);
    const [x2, y2] = mapPoint(end, t);
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const head = 12;
    overlayCtx.save();
    overlayCtx.strokeStyle = color;
    overlayCtx.fillStyle = color;
    overlayCtx.lineWidth = width;
    drawRawLine(x1, y1, x2, y2);
    overlayCtx.beginPath();
    overlayCtx.moveTo(x2, y2);
    overlayCtx.lineTo(x2 - head * Math.cos(angle - Math.PI / 6), y2 - head * Math.sin(angle - Math.PI / 6));
    overlayCtx.lineTo(x2 - head * Math.cos(angle + Math.PI / 6), y2 - head * Math.sin(angle + Math.PI / 6));
    overlayCtx.closePath();
    overlayCtx.fill();
    overlayCtx.restore();
}

// ===== 告警列表更新 =====
function updateAlertList(alerts) {
    if (!dom.alertList) return;

    if (alerts.length === 0) {
        dom.alertList.innerHTML = '<div style="padding:12px;color:var(--text-muted);text-align:center;font-size:14px;">无活跃告警</div>';
        dom.statusDot.className = 'live-dot active';
        dom.statusText.textContent = '检测中';
        return;
    }

    // 更新状态指示灯
    const hasDanger = alerts.some(a => a.severity === 'danger');
    if (hasDanger) {
        dom.statusDot.className = 'live-dot';  // 红色闪烁
        dom.statusText.textContent = '⚠ 告警';
    }

    dom.alertList.innerHTML = alerts.slice(0, 3).map(a => {
        const bgColor = a.severity === 'danger' ? 'rgba(255,69,96,0.15)' :
                        a.severity === 'warning' ? 'rgba(255,181,71,0.15)' :
                        'rgba(77,159,255,0.1)';
        const borderColor = a.severity === 'danger' ? 'rgba(255,69,96,0.3)' :
                            a.severity === 'warning' ? 'rgba(255,181,71,0.3)' :
                            'rgba(77,159,255,0.2)';
        return `<div style="padding:8px 12px;margin-bottom:6px;background:${bgColor};border-left:3px solid ${borderColor};border-radius:6px;font-size:13px;">
            <span class="severity-dot ${a.severity || 'info'}"></span>
            <span style="color:var(--text-primary);">${a.message || a.type}</span>
        </div>`;
    }).join('');

    // 触发综合告警(音效、语音播报、桌面通知、Toast)
    alerts.filter(a => a.severity === 'danger' || a.severity === 'warning').forEach(a => {
        if (typeof triggerAlert !== 'undefined') {
            triggerAlert(a);
        }
    });
}

// ===== 错误处理 =====
function showCameraError(message) {
    const placeholder = document.getElementById('camera-placeholder');
    if (placeholder) {
        placeholder.innerHTML = `
            <div style="text-align:center;padding:40px;">
                <i class="bi bi-exclamation-triangle-fill" style="font-size:3rem;color:var(--danger);margin-bottom:16px;display:block;"></i>
                <p style="color:var(--danger);font-weight:600;margin-bottom:8px;">摄像头启动失败</p>
                <p style="color:var(--text-secondary);font-size:14px;">${message}</p>
                <p style="color:var(--text-muted);font-size:12px;margin-top:12px;">请确保:<br>1. 已连接摄像头设备<br>2. 浏览器已授权摄像头权限<br>3. 使用 HTTPS 或 localhost 访问</p>
            </div>`;
    }
}

// ===== 页面初始化 =====
document.addEventListener('DOMContentLoaded', async () => {
    cacheDomElements();

    // 枚举摄像头设备
    await enumerateDevices();

    // 绑定按钮事件
    if (dom.startBtn) {
        dom.startBtn.addEventListener('click', startCamera);
    }
    if (dom.stopBtn) {
        dom.stopBtn.addEventListener('click', stopCamera);
    }

    // 设备切换时自动重启
    if (dom.deviceSelect) {
        dom.deviceSelect.addEventListener('change', () => {
            if (isRunning) {
                stopCamera();
                setTimeout(startCamera, 500);
            }
        });
    }

    // 页面卸载时清理
    window.addEventListener('beforeunload', () => {
        if (isRunning) stopCamera();
    });
    window.addEventListener('resize', () => {
        if (isRunning) resizeOverlayCanvas();
    });
});
