/**
 * dashboard.js - 仪表盘实时更新逻辑
 */

// 存储时序数据
const dashboardData = {
    earValues: [],
    marValues: [],
    pitchValues: [],
    yawValues: [],
    rollValues: [],
    ppgSignal: [],
    timestamps: [],
    frameCount: 0
};

// ===== 更新仪表盘数据 =====
function updateDashboard(resultData) {
    if (!resultData) return;

    dashboardData.frameCount++;

    // 更新疲劳相关数据
    if (resultData.fatigue) {
        const f = resultData.fatigue;
        if (f.ear !== undefined) {
            dashboardData.earValues.push(f.ear);
            dashboardData.marValues.push(f.mar || 0);
            dashboardData.timestamps.push(dashboardData.frameCount);
        }
    }

    // 更新头部姿态数据
    if (resultData.head_pose) {
        const hp = resultData.head_pose;
        dashboardData.pitchValues.push(hp.pitch || 0);
        dashboardData.yawValues.push(hp.yaw || 0);
        dashboardData.rollValues.push(hp.roll || 0);
    }

    // 更新PPG数据
    if (resultData.ppg_value !== undefined) {
        dashboardData.ppgSignal.push(resultData.ppg_value);
    }

    // 限制数据长度
    const maxPoints = 300;
    if (dashboardData.earValues.length > maxPoints) {
        dashboardData.earValues = dashboardData.earValues.slice(-maxPoints);
        dashboardData.marValues = dashboardData.marValues.slice(-maxPoints);
        dashboardData.timestamps = dashboardData.timestamps.slice(-maxPoints);
    }
}

// ===== 更新仪表盘告警 =====
function updateDashboardAlerts(alerts) {
    if (!alerts || alerts.length === 0) return;

    const statusBar = document.getElementById('overall-status');
    const statusText = document.getElementById('status-text');
    const badgeContainer = document.getElementById('active-alert-badges');

    // 检查是否有danger级别告警
    const hasDanger = alerts.some(a => a.severity === 'danger');
    const hasWarning = alerts.some(a => a.severity === 'warning');

    // 更新状态栏样式（使用颜色指示，无图标/emoji）
    statusBar.classList.remove('alert-secondary', 'alert-warning', 'alert-danger');
    if (hasDanger) {
        statusBar.classList.add('alert-danger');
        statusText.innerHTML = '<strong>危险告警! 请立即注意!</strong>';
    } else if (hasWarning) {
        statusBar.classList.add('alert-warning');
        statusText.innerHTML = '<strong>检测到异常行为</strong>';
    }

    // 更新告警徽章（使用新的 alert-badge danger / alert-badge warning 类名）
    badgeContainer.innerHTML = alerts.map(a => {
        const cls = a.severity === 'danger' ? 'alert-badge danger' :
                    a.severity === 'warning' ? 'alert-badge warning' : 'alert-badge info';
        return `<span class="badge ${cls} me-1">${a.type || a.alert_type}</span>`;
    }).join('');

    // 更新告警计数
    const countBadge = document.getElementById('alert-count-badge');
    if (countBadge) {
        countBadge.textContent = alerts.length;
        countBadge.classList.remove('d-none');
    }

    // 添加到告警表格
    addAlertsToTable(alerts);
}

// ===== 添加告警到表格 =====
function addAlertsToTable(alerts) {
    const tbody = document.getElementById('alerts-tbody');
    if (!tbody) return;

    // 移除"暂无数据"行
    const noDataRow = tbody.querySelector('tr td[colspan]');
    if (noDataRow) noDataRow.parentElement.remove();

    alerts.forEach(alert => {
        const row = document.createElement('tr');
        const time = new Date(alert.timestamp * 1000).toLocaleTimeString('zh-CN');
        const severityClass = alert.severity === 'danger' ? 'text-danger fw-bold' :
                              alert.severity === 'warning' ? 'text-warning' : 'text-info';
        row.innerHTML = `
            <td><small>${time}</small></td>
            <td><span class="badge bg-secondary">${alert.source || alert.type || ''}</span></td>
            <td class="${severityClass}">${alert.severity === 'danger' ? '🔴 危险' :
                                          alert.severity === 'warning' ? '🟡 警告' : '🔵 信息'}</td>
            <td><small>${alert.message || ''}</small></td>
        `;
        tbody.insertBefore(row, tbody.firstChild);

        // 弹出Toast通知
        if (alert.severity === 'danger') {
            showToast(`⚠️ ${alert.message}`, 'danger');
        }
    });

    // 限制行数
    while (tbody.children.length > 50) {
        tbody.lastElementChild.remove();
    }
}

// ===== 更新生理信号显示 =====
function updatePhysioDisplay(data) {
    if (data.heart_rate) {
        document.getElementById('heart-rate-value').textContent = Math.round(data.heart_rate);
        document.getElementById('hr-card-value').textContent = Math.round(data.heart_rate);
    }
    if (data.hrv) {
        document.getElementById('hrv-value').textContent = Math.round(data.hrv);
    }
    if (data.bp_systolic) {
        document.getElementById('bp-systolic-value').textContent = Math.round(data.bp_systolic);
    }
    if (data.bp_diastolic) {
        document.getElementById('bp-diastolic-value').textContent = Math.round(data.bp_diastolic);
    }
}

// ===== 更新底部统计卡片 =====
function updateSummaryCards(summary) {
    if (!summary) return;

    const fatigueScore = document.getElementById('fatigue-score');
    if (fatigueScore && summary.fatigue_score !== undefined) {
        fatigueScore.textContent = Math.round(summary.fatigue_score);
        fatigueScore.className = 'fs-3 fw-bold ' + getScoreClass(summary.fatigue_score);
    }

    const distractionScore = document.getElementById('distraction-score');
    if (distractionScore && summary.distraction_score !== undefined) {
        distractionScore.textContent = Math.round(summary.distraction_score);
        distractionScore.className = 'fs-3 fw-bold ' + getScoreClass(summary.distraction_score);
    }

    if (summary.total_alerts !== undefined) {
        document.getElementById('alert-total').textContent = summary.total_alerts;
    }
}

// ===== 评分颜色类 =====
function getScoreClass(score) {
    if (score >= 70) return 'score-good';
    if (score >= 40) return 'score-warning';
    return 'score-danger';
}

// ===== 重置仪表盘 =====
function resetDashboard() {
    dashboardData.earValues = [];
    dashboardData.marValues = [];
    dashboardData.pitchValues = [];
    dashboardData.yawValues = [];
    dashboardData.rollValues = [];
    dashboardData.ppgSignal = [];
    dashboardData.timestamps = [];
    dashboardData.frameCount = 0;

    document.getElementById('heart-rate-value').textContent = '--';
    document.getElementById('hrv-value').textContent = '--';
    document.getElementById('bp-systolic-value').textContent = '--';
    document.getElementById('bp-diastolic-value').textContent = '--';
    document.getElementById('blink-rate-value').textContent = '--';
    document.getElementById('eye-closure-count').textContent = '0';
    document.getElementById('yawn-count').textContent = '0';
    document.getElementById('gaze-deviation-count').textContent = '0';
    document.getElementById('hands-off-count').textContent = '0';
    document.getElementById('body-turn-count').textContent = '0';
    document.getElementById('fatigue-score').textContent = '--';
    document.getElementById('distraction-score').textContent = '--';
    document.getElementById('alert-total').textContent = '0';
    document.getElementById('hr-card-value').textContent = '--';

    const alertsTbody = document.getElementById('alerts-tbody');
    if (alertsTbody) {
        alertsTbody.innerHTML = '<tr><td colspan="4" class="text-center text-secondary">暂无告警记录</td></tr>';
    }

    const countBadge = document.getElementById('alert-count-badge');
    if (countBadge) countBadge.classList.add('d-none');

    const statusBar = document.getElementById('overall-status');
    statusBar.classList.remove('alert-warning', 'alert-danger');
    statusBar.classList.add('alert-secondary');
    document.getElementById('status-text').innerHTML = '<i class="bi bi-info-circle-fill me-2"></i>请上传图像或视频文件开始检测';
    document.getElementById('active-alert-badges').innerHTML = '';
}
