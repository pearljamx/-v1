/**
 * alert.js - 告警通知组件
 */

// 告警音效（可选，浏览器需用户交互后才能播放）
let alertAudioCtx = null;

// ===== 初始化告警音频 =====
function initAlertAudio() {
    try {
        alertAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    } catch (e) {
        console.log('音频不可用');
    }
}

// ===== 播放告警音效 =====
function playAlertSound(severity) {
    if (!alertAudioCtx) return;

    const osc = alertAudioCtx.createOscillator();
    const gain = alertAudioCtx.createGain();
    osc.connect(gain);
    gain.connect(alertAudioCtx.destination);

    if (severity === 'danger') {
        // 高频急促音
        osc.frequency.value = 880;
        osc.type = 'square';
        gain.gain.value = 0.3;
        osc.start();
        osc.stop(alertAudioCtx.currentTime + 0.3);
    } else if (severity === 'warning') {
        // 中频提示音
        osc.frequency.value = 660;
        osc.type = 'sine';
        gain.gain.value = 0.2;
        osc.start();
        osc.stop(alertAudioCtx.currentTime + 0.15);
    }
}

// ===== 桌面通知 =====
function sendDesktopNotification(title, body, severity) {
    if (!('Notification' in window)) return;

    if (Notification.permission === 'granted') {
        new Notification(title, {
            body: body,
            icon: '/static/favicon.ico',
            tag: 'driver-alert'
        });
    } else if (Notification.permission !== 'denied') {
        Notification.requestPermission().then(perm => {
            if (perm === 'granted') {
                new Notification(title, {
                    body: body,
                    icon: '/static/favicon.ico',
                    tag: 'driver-alert'
                });
            }
        });
    }
}

// ===== Toast通知 (纯CSS，无Bootstrap依赖) =====
function showToast(message, type) {
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }

    const icons = { danger: 'x-circle-fill', warning: 'exclamation-triangle-fill', info: 'info-circle-fill' };
    const cssType = type === 'danger' ? 'toast-danger' :
                    type === 'warning' ? 'toast-warning' : 'toast-info';

    const toastEl = document.createElement('div');
    toastEl.className = 'toast-custom ' + cssType;
    toastEl.innerHTML = '<i class="bi bi-' + (icons[type] || 'info-circle-fill') + '" style="margin-right:8px;"></i>' + message;

    toastContainer.appendChild(toastEl);

    // 限制最多5个Toast
    while (toastContainer.children.length > 5) {
        toastContainer.firstElementChild.remove();
    }

    // 5秒后自动fadeOut并remove
    setTimeout(function() {
        toastEl.style.opacity = '0';
        toastEl.style.transform = 'translateX(100px)';
        toastEl.style.transition = 'all 0.3s ease';
        setTimeout(function() { toastEl.remove(); }, 300);
    }, 5000);
}

// ===== 综合告警触发函数 =====
function triggerAlert(alertData) {
    // 1. 播放音效
    playAlertSound(alertData.severity);

    // 2. 语音播报 (TTS)
    speakAlert(alertData.message, alertData.severity);

    // 3. 桌面通知（仅danger级别）
    if (alertData.severity === 'danger') {
        sendDesktopNotification(
            '⚠️ 驾驶告警',
            alertData.message || '检测到危险驾驶行为',
            alertData.severity
        );
    }

    // 4. 页面Toast
    showToast(alertData.message, alertData.severity);

    // 5. 记录到告警日志面板
    logAlertToPanel(alertData);
}

// ===== 告警面板日志 =====
function logAlertToPanel(alertData) {
    const tbody = document.getElementById('alerts-tbody');
    if (!tbody) return;

    const noDataRow = tbody.querySelector('tr td[colspan]');
    if (noDataRow) noDataRow.parentElement.remove();

    const row = document.createElement('tr');
    const time = new Date(alertData.timestamp * 1000 || Date.now()).toLocaleTimeString('zh-CN');
    const badgeClass = alertData.severity === 'danger' ?
        'alert-badge danger' :
        alertData.severity === 'warning' ?
        'alert-badge warning' : '';

    const severityLabel = alertData.severity === 'danger' ? '🔴 危险' :
                          alertData.severity === 'warning' ? '🟡 警告' : '🔵 信息';

    row.innerHTML = `
        <td><small>${time}</small></td>
        <td><span class="${badgeClass}">${alertData.source || alertData.type || ''}</span></td>
        <td>${severityLabel}</td>
        <td><small>${alertData.message || ''}</small></td>
    `;
    tbody.insertBefore(row, tbody.firstChild);

    // 限制最多显示50行
    while (tbody.children.length > 50) {
        tbody.lastElementChild.remove();
    }
}

// ===== 请求通知权限 =====
function requestNotificationPermission() {
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
}

// ===== TTS语音播报 (Web Speech API) =====
let _ttsMuted = false;
let _ttsLastSpoken = {};  // {message: timestamp}

// ===== 语音播报 =====
function speakAlert(message, severity) {
    if (_ttsMuted) return;
    if (!message || typeof message !== 'string') return;

    // 节流: 相同消息5秒内不重复播报
    const now = Date.now();
    if (_ttsLastSpoken[message] && (now - _ttsLastSpoken[message]) < 5000) {
        return;
    }

    // 检查浏览器是否支持 speechSynthesis
    if (!('speechSynthesis' in window)) {
        console.log('浏览器不支持语音合成');
        return;
    }

    try {
        const utterance = new SpeechSynthesisUtterance(message);

        // 使用中文语音
        utterance.lang = 'zh-CN';

        // 根据严重程度设置语速
        if (severity === 'danger') {
            utterance.rate = 1.2;   // 危险告警: 快速播报
        } else {
            utterance.rate = 1.0;   // 警告/信息: 正常语速
        }

        // 尝试匹配最佳中文语音
        const voices = window.speechSynthesis.getVoices();
        if (voices.length > 0) {
            // 优先选择中文女声 (zh-CN)
            let bestVoice = voices.find(v => v.lang === 'zh-CN' && v.name.includes('Female'));
            if (!bestVoice) {
                bestVoice = voices.find(v => v.lang === 'zh-CN');
            }
            if (!bestVoice) {
                bestVoice = voices.find(v => v.lang.startsWith('zh'));
            }
            if (bestVoice) {
                utterance.voice = bestVoice;
            }
        }

        // 记录已播报
        _ttsLastSpoken[message] = now;

        // 清理过期记录 (超过10秒的)
        for (const key of Object.keys(_ttsLastSpoken)) {
            if (now - _ttsLastSpoken[key] > 10000) {
                delete _ttsLastSpoken[key];
            }
        }

        // 播报
        window.speechSynthesis.speak(utterance);
    } catch (e) {
        console.log('语音播报失败:', e);
    }
}

// ===== 静音/取消静音切换 =====
function toggleTtsMute() {
    _ttsMuted = !_ttsMuted;

    // 如果切换到静音，取消所有待播报的语音
    if (_ttsMuted && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
    }

    // 更新UI按钮状态
    const btn = document.getElementById('btn-tts-mute');
    if (btn) {
        if (_ttsMuted) {
            btn.innerHTML = '<i class="bi bi-volume-mute-fill"></i> 取消静音';
            btn.className = btn.className.replace(/btn-secondary|btn-outline-secondary/g, 'btn-outline-danger') || '';
            if (!btn.className.includes('btn-outline-danger')) {
                btn.className = 'btn btn-outline-danger btn-sm';
            }
        } else {
            btn.innerHTML = '<i class="bi bi-volume-up-fill"></i> 静音';
            btn.className = 'btn btn-outline-secondary btn-sm';
        }
    }

    return _ttsMuted;
}

// ===== 获取当前静音状态 =====
function isTtsMuted() {
    return _ttsMuted;
}

// ===== 预加载中文语音列表 (兼容 Chrome 异步加载) =====
function preloadVoices() {
    if ('speechSynthesis' in window) {
        // 触发语音列表加载
        window.speechSynthesis.getVoices();
        // Chrome 异步加载语音，监听 voiceschanged 事件
        window.speechSynthesis.onvoiceschanged = function() {
            window.speechSynthesis.getVoices();
        };
    }
}

// 页面加载后初始化
document.addEventListener('DOMContentLoaded', () => {
    initAlertAudio();
    requestNotificationPermission();
    preloadVoices();
});
