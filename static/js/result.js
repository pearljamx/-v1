/**
 * result.js - 结果页面数据加载和渲染
 */

// ===== 加载检测结果 =====
async function loadResults(taskId) {
    try {
        // 获取完整结果
        const resp = await fetch(`/api/results/${taskId}`);
        if (!resp.ok) {
            document.getElementById('result-alerts-tbody').innerHTML =
                '<tr><td colspan="5" class="text-center text-danger">加载结果失败</td></tr>';
            return;
        }

        const data = await resp.json();
        renderResults(data);

    } catch (error) {
        console.error('加载结果失败:', error);
        document.getElementById('result-alerts-tbody').innerHTML =
            '<tr><td colspan="5" class="text-center text-danger">网络错误</td></tr>';
    }
}

// ===== 渲染结果数据 =====
function renderResults(data) {
    // 1. 综合评分
    renderOverallRisk(data.summary);
    renderScores(data.summary);

    // 2. 疲劳检测详情
    if (data.fatigue) {
        renderFatigueDetails(data.fatigue);
    }

    // 3. 分心检测详情
    if (data.head_pose || data.distraction) {
        renderDistractionDetails(data);
        renderDriverState(data.distraction || {});
    }

    // 4. 生理信号
    if (data.physiological) {
        renderPhysioDetails(data.physiological);
    }

    // 5. 告警列表
    renderAlerts(data.alerts || []);
}

// ===== 渲染综合风险等级 =====
function renderOverallRisk(summary) {
    if (!summary) return;

    const el = document.getElementById('overall-risk');
    if (!el) return;

    const risk = summary.overall_risk || 'low';
    const riskText = risk === 'high' ? '高风险' : risk === 'medium' ? '中等风险' : '低风险';
    const riskClass = risk === 'high' ? 'text-danger' : risk === 'medium' ? 'text-warning' : 'text-success';

    el.textContent = riskText;
    el.className = `display-4 fw-bold mt-2 ${riskClass}`;
}

// ===== 数字计数动画 =====
function animateCountUp(el, target, duration) {
    duration = duration || 1500;
    const start = performance.now();
    const from = 0;

    function step(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1.0);
        // ease-out 缓动
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = from + (target - from) * eased;

        // 根据目标值是否为浮点数来决定显示精度
        if (Number.isInteger(target)) {
            el.textContent = Math.round(current);
        } else {
            el.textContent = current.toFixed(1);
        }

        if (progress < 1.0) {
            requestAnimationFrame(step);
        } else {
            // 确保最终值精确
            el.textContent = Number.isInteger(target) ? target : target.toFixed(1);
        }
    }

    requestAnimationFrame(step);
}

// ===== 渲染评分 =====
function renderScores(summary) {
    if (!summary) return;

    const fatigueEl = document.getElementById('r-fatigue-score');
    const distEl = document.getElementById('r-distraction-score');

    if (fatigueEl && summary.fatigue_score !== undefined) {
        fatigueEl.className = 'display-4 fw-bold mt-2 ' + getScoreClass(summary.fatigue_score);
        animateCountUp(fatigueEl, summary.fatigue_score);
    }

    if (distEl && summary.distraction_score !== undefined) {
        distEl.className = 'display-4 fw-bold mt-2 ' + getScoreClass(summary.distraction_score);
        animateCountUp(distEl, summary.distraction_score);
    }
}

// ===== 渲染疲劳检测详情 =====
function renderFatigueDetails(fatigue) {
    // 平均值和最小值
    const avgEar = document.getElementById('avg-ear');
    const minEar = document.getElementById('min-ear');
    const blinkRate = document.getElementById('r-blink-rate');
    const noddingCount = document.getElementById('r-nodding-count');

    if (avgEar && fatigue.avg_ear !== undefined) {
        avgEar.textContent = fatigue.avg_ear.toFixed(3);
    }
    if (minEar && fatigue.min_ear !== undefined) {
        minEar.textContent = fatigue.min_ear.toFixed(3);
    }
    if (blinkRate && fatigue.blink_rate !== undefined) {
        blinkRate.textContent = fatigue.blink_rate.toFixed(1) + ' 次/分钟';
    }
    if (noddingCount && fatigue.nodding_count !== undefined) {
        noddingCount.textContent = fatigue.nodding_count;
    }
}

// ===== 渲染分心检测详情 =====
function renderDistractionDetails(data) {
    const gazeDev = document.getElementById('r-gaze-dev');
    const objects = document.getElementById('r-objects');

    if (gazeDev && data.distraction) {
        const events = data.distraction.gaze_deviation_events || [];
        gazeDev.textContent = events.length;
    }

    if (objects && data.distraction) {
        const detected = data.distraction.objects_detected || [];
        if (detected.length > 0) {
            objects.innerHTML = detected.map(o => {
                // 映射 class 到具体的 CSS 类名
                const cls = (o.class || '').toLowerCase();
                let tagClass = 'object-tag';
                if (cls === 'phone' || cls === 'cell phone') {
                    tagClass += ' phone';
                } else if (cls === 'smoking' || cls === 'cigarette') {
                    tagClass += ' smoking';
                } else if (cls === 'drinking' || cls === 'bottle' || cls === 'cup') {
                    tagClass += ' drinking';
                }
                return `<span class="${tagClass}">${o.class} (${o.count || 1}次)</span>`;
            }).join(' ');
        } else {
            objects.textContent = '未检测到异常物体';
        }
    }
}

function renderDriverState(distraction) {
    const classEl = document.getElementById('r-driver-class');
    const confEl = document.getElementById('r-driver-confidence');
    const state = distraction.driver_state || {};
    const labelMap = {
        normal: '正常驾驶',
        phone: '手机分心',
        drinking: '饮水/进食',
        turning: '转身/交谈',
        drowsy: '疲劳状态',
        secondary_task: '分心操作',
    };
    if (classEl) {
        const name = state.class_name || state.class;
        classEl.textContent = name ? (labelMap[name] || name) : '未输出';
    }
    if (confEl) {
        confEl.textContent = state.confidence !== undefined ? (state.confidence * 100).toFixed(1) + '%' : '--';
    }
}

// ===== 渲染生理信号 =====
function renderPhysioDetails(physio) {
    const section = document.getElementById('physio-section');
    if (section) section.style.display = 'block';

    const hr = document.getElementById('r-heart-rate');
    const bpSys = document.getElementById('r-bp-sys');
    const bpDia = document.getElementById('r-bp-dia');

    if (hr && physio.heart_rate) {
        hr.textContent = Math.round(physio.heart_rate);
    }
    if (bpSys && physio.bp_systolic) {
        bpSys.textContent = Math.round(physio.bp_systolic);
    }
    if (bpDia && physio.bp_diastolic) {
        bpDia.textContent = Math.round(physio.bp_diastolic);
    }
}

// ===== 渲染告警列表 =====
function renderAlerts(alerts) {
    const tbody = document.getElementById('result-alerts-tbody');
    const totalBadge = document.getElementById('result-alert-total');

    if (!tbody) return;

    if (totalBadge) totalBadge.textContent = alerts.length;

    if (alerts.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center text-success">&#10004; 未检测到告警</td></tr>';
        return;
    }

    // 按时间倒序排列
    alerts.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));

    tbody.innerHTML = alerts.map(a => {
        const time = new Date((a.timestamp || 0) * 1000).toLocaleTimeString('zh-CN');
        const severity = a.severity || 'info';
        // 使用 .severity-dot + 颜色名，替换原来的 emoji
        const sevHTML = `<span class="severity-dot ${severity}"></span> ${severity === 'danger' ? '危险' : severity === 'warning' ? '警告' : '信息'}`;
        return `<tr>
            <td><small>${time}</small></td>
            <td><span class="badge bg-secondary">${a.source || ''}</span></td>
            <td><span class="badge bg-secondary">${a.type || ''}</span></td>
            <td>${sevHTML}</td>
            <td><small>${a.message || ''}</small></td>
        </tr>`;
    }).join('');
}

// ===== 评分颜色类 =====
function getScoreClass(score) {
    if (score >= 70) return 'score-good';
    if (score >= 40) return 'score-warning';
    return 'score-danger';
}
