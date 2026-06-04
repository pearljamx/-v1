/**
 * upload.js - 文件上传交互逻辑
 */

let selectedFile = null;
let currentTaskId = null;
let pollInterval = null;

// DOM元素
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const startBtn = document.getElementById('start-btn');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress-bar');
const progressPercent = document.getElementById('progress-percent');
const progressLabel = document.getElementById('progress-label');

// ===== 拖拽上传事件 =====
if (dropZone && fileInput) {
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
}

// ===== 文件处理 =====
function handleFile(file) {
    // 检查文件类型
    const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp',
                          'video/mp4', 'video/avi', 'video/mov', 'video/webm'];
    if (!allowedTypes.includes(file.type)) {
        showToast('不支持的文件格式，请上传图像或视频文件', 'warning');
        return;
    }

    // 检查文件大小 (500MB)
    if (file.size > 500 * 1024 * 1024) {
        showToast('文件大小超过500MB限制', 'warning');
        return;
    }

    selectedFile = file;
    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = formatFileSize(file.size);
    if (fileInfo) fileInfo.classList.remove('d-none');
    if (startBtn) startBtn.disabled = false;

    // 自动检测模式
    const modeSelect = document.getElementById('detect-mode');
    if (modeSelect && modeSelect.value === 'auto') {
        modeSelect.value = file.type.startsWith('image/') ? 'image' : 'video';
    }
}

function clearFile() {
    selectedFile = null;
    if (fileInput) fileInput.value = '';
    if (fileInfo) fileInfo.classList.add('d-none');
    if (startBtn) startBtn.disabled = true;
    resetUI();
}

// ===== 文件大小格式化 =====
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
}

// ===== 开始检测 =====
async function startDetection() {
    if (!selectedFile || !startBtn || !progressContainer) return;

    startBtn.disabled = true;
    progressContainer.classList.remove('d-none');
    updateProgress(0, '上传文件中...');

    // Step 1: 上传文件
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('mode', document.getElementById('detect-mode').value);
    formData.append('enable_fatigue', document.getElementById('enable-fatigue').checked);
    formData.append('enable_pose', document.getElementById('enable-pose').checked);
    formData.append('enable_gaze', document.getElementById('enable-gaze').checked);
    formData.append('enable_distraction', document.getElementById('enable-distraction').checked);
    formData.append('enable_physio', document.getElementById('enable-physio').checked);

    try {
        updateProgress(10, '正在上传...');
        const uploadResp = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const uploadData = await uploadResp.json();

        if (!uploadResp.ok) {
            throw new Error(uploadData.error || '上传失败');
        }

        currentTaskId = uploadData.task_id;
        updateProgress(30, '上传完成，开始处理...');

        // Step 2: 触发检测
        const detectType = uploadData.detect_type || 'auto';
        const detectResp = await fetch(`/detect/${detectType}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: currentTaskId })
        });
        const detectData = await detectResp.json();

        if (!detectResp.ok) {
            throw new Error(detectData.error || '检测启动失败');
        }

        // Step 3: 开始轮询进度
        startPolling(currentTaskId);

    } catch (error) {
        console.error('检测失败:', error);
        showToast('检测失败: ' + error.message, 'danger');
        resetUI();
    }
}

// ===== 进度更新 =====
function updateProgress(percent, label) {
    if (progressBar) progressBar.style.width = percent + '%';
    if (progressPercent) progressPercent.textContent = percent + '%';
    if (label && progressLabel) progressLabel.textContent = label;
}

// ===== 轮询任务状态 =====
function startPolling(taskId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        try {
            const resp = await fetch(`/api/status/${taskId}`);
            const data = await resp.json();

            if (data.status === 'processing') {
                const pct = data.progress || 0;
                updateProgress(30 + pct * 0.7, `处理中... (${data.current_frame || 0} 帧)`);

                // 更新实时数据
                if (data.current_alerts) {
                    updateLiveAlerts(data.current_alerts);
                }
                if (data.preview_frame) {
                    updatePreview(data.preview_frame);
                }
            } else if (data.status === 'done') {
                clearInterval(pollInterval);
                pollInterval = null;
                updateProgress(100, '处理完成!');
                // 跳转到结果页面
                setTimeout(() => {
                    window.location.href = `/result/${taskId}`;
                }, 500);
            } else if (data.status === 'error') {
                clearInterval(pollInterval);
                pollInterval = null;
                showToast('处理出错: ' + (data.error || '未知错误'), 'danger');
                resetUI();
            }
        } catch (error) {
            console.error('轮询失败:', error);
        }
    }, 1000); // 每秒轮询一次
}

// ===== 实时预览更新 =====
function updatePreview(frameBase64) {
    const previewPlaceholder = document.getElementById('preview-placeholder');
    const previewContent = document.getElementById('preview-content');
    const previewImage = document.getElementById('preview-image');

    if (previewPlaceholder) previewPlaceholder.classList.add('d-none');
    if (previewContent) previewContent.classList.remove('d-none');
    if (previewImage) previewImage.src = 'data:image/jpeg;base64,' + frameBase64;
}

// ===== 实时告警更新 =====
function updateLiveAlerts(alerts) {
    if (window.updateDashboardAlerts) {
        window.updateDashboardAlerts(alerts);
    }
}

// ===== 重置UI =====
function resetUI() {
    if (startBtn) startBtn.disabled = !selectedFile;
    if (progressContainer) progressContainer.classList.add('d-none');
    updateProgress(0, '');
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// ===== Toast通知 (纯CSS，无Bootstrap依赖) =====
function showToast(message, type = 'info') {
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }

    const icons = { info: 'info-circle', warning: 'exclamation-triangle', danger: 'x-circle-fill' };
    const cssType = type === 'danger' ? 'toast-danger' : type === 'warning' ? 'toast-warning' : 'toast-info';

    const toastEl = document.createElement('div');
    toastEl.className = 'toast-custom ' + cssType;
    toastEl.innerHTML = '<i class="bi bi-' + (icons[type] || 'info-circle') + '" style="margin-right:8px;"></i>' + message;

    toastContainer.appendChild(toastEl);

    // 5秒后自动移除
    setTimeout(function() {
        toastEl.style.opacity = '0';
        toastEl.style.transform = 'translateX(100px)';
        toastEl.style.transition = 'all 0.3s ease';
        setTimeout(function() { toastEl.remove(); }, 300);
    }, 5000);
}
