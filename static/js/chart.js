/**
 * chart.js - Chart.js图表绘制
 */

let earChart = null;
let marChart = null;
let headPoseChart = null;
let ppgChart = null;

// ===== 初始化所有图表 =====
function initCharts() {
    initEARChart();
    initMARChart();
    initHeadPoseChart();
    initPPGChart();
}

// ===== EAR时序图 =====
function initEARChart() {
    const ctx = document.getElementById('ear-chart');
    if (!ctx) return;

    if (earChart) earChart.destroy();

    earChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'EAR值',
                data: [],
                borderColor: '#6C47FF',
                backgroundColor: 'rgba(108, 71, 255, 0.08)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    display: true,
                    title: { display: true, text: '帧', color: '#8888AA' },
                    ticks: { color: '#5A5A7A' },
                    grid: { color: 'rgba(108, 71, 255, 0.08)' }
                },
                y: {
                    min: 0,
                    max: 0.5,
                    title: { display: true, text: 'EAR', color: '#8888AA' },
                    ticks: { color: '#5A5A7A' },
                    grid: { color: 'rgba(108, 71, 255, 0.08)' }
                }
            },
            plugins: {
                legend: { labels: { color: '#8888AA' } },
                annotation: {
                    annotations: {
                        thresholdLine: {
                            type: 'line',
                            yMin: 0.2,
                            yMax: 0.2,
                            borderColor: 'rgba(255, 69, 96, 0.5)',
                            borderWidth: 1,
                            borderDash: [5, 5],
                            label: {
                                content: '闭眼阈值(0.2)',
                                display: true,
                                color: '#FF4560',
                                position: 'end'
                            }
                        }
                    }
                }
            },
            animation: { duration: 200 }
        }
    });
}

// ===== MAR时序图 =====
function initMARChart() {
    const ctx = document.getElementById('mar-chart');
    if (!ctx) return;

    if (marChart) marChart.destroy();

    marChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'MAR值',
                data: [],
                borderColor: '#FFB547',
                backgroundColor: 'rgba(255, 181, 71, 0.08)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    display: true,
                    title: { display: true, text: '帧', color: '#8888AA' },
                    ticks: { color: '#5A5A7A' },
                    grid: { color: 'rgba(108, 71, 255, 0.08)' }
                },
                y: {
                    min: 0,
                    max: 1.0,
                    title: { display: true, text: 'MAR', color: '#8888AA' },
                    ticks: { color: '#5A5A7A' },
                    grid: { color: 'rgba(108, 71, 255, 0.08)' }
                }
            },
            plugins: {
                legend: { labels: { color: '#8888AA' } },
                annotation: {
                    annotations: {
                        thresholdLine: {
                            type: 'line',
                            yMin: 0.5,
                            yMax: 0.5,
                            borderColor: 'rgba(255, 69, 96, 0.5)',
                            borderWidth: 1,
                            borderDash: [5, 5],
                            label: {
                                content: '哈欠阈值(0.5)',
                                display: true,
                                color: '#FF4560',
                                position: 'end'
                            }
                        }
                    }
                }
            },
            animation: { duration: 200 }
        }
    });
}

// ===== 头部姿态图 =====
function initHeadPoseChart() {
    const ctx = document.getElementById('head-pose-chart');
    if (!ctx) return;

    if (headPoseChart) headPoseChart.destroy();

    headPoseChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Pitch (俯仰)',
                    data: [],
                    borderColor: '#6C47FF',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3
                },
                {
                    label: 'Yaw (偏航)',
                    data: [],
                    borderColor: '#00D4AA',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3
                },
                {
                    label: 'Roll (翻滚)',
                    data: [],
                    borderColor: '#FFB547',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: '帧', color: '#8888AA' },
                    ticks: { color: '#5A5A7A' },
                    grid: { color: 'rgba(108, 71, 255, 0.08)' }
                },
                y: {
                    title: { display: true, text: '角度(度)', color: '#8888AA' },
                    ticks: { color: '#5A5A7A' },
                    grid: { color: 'rgba(108, 71, 255, 0.08)' }
                }
            },
            plugins: {
                legend: { labels: { color: '#8888AA' } }
            },
            animation: { duration: 200 }
        }
    });
}

// ===== PPG波形图 =====
function initPPGChart() {
    const ctx = document.getElementById('ppg-chart');
    if (!ctx) return;

    if (ppgChart) ppgChart.destroy();

    ppgChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'PPG信号',
                data: [],
                borderColor: '#FF4560',
                backgroundColor: 'rgba(255, 69, 96, 0.08)',
                borderWidth: 1.5,
                pointRadius: 0,
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: '帧', color: '#8888AA' },
                    ticks: { color: '#5A5A7A' },
                    grid: { color: 'rgba(108, 71, 255, 0.08)' }
                },
                y: {
                    title: { display: true, text: '幅值', color: '#8888AA' },
                    ticks: { color: '#5A5A7A' },
                    grid: { color: 'rgba(108, 71, 255, 0.08)' }
                }
            },
            plugins: {
                legend: { labels: { color: '#8888AA' } }
            },
            animation: { duration: 200 }
        }
    });
}

// ===== 更新图表数据 =====
function updateCharts(data) {
    const labels = data.timestamps || [];

    if (earChart && data.ear_values) {
        earChart.data.labels = labels;
        earChart.data.datasets[0].data = data.ear_values;
        earChart.update('none');
    }

    if (marChart && data.mar_values) {
        marChart.data.labels = labels;
        marChart.data.datasets[0].data = data.mar_values;
        marChart.update('none');
    }

    if (headPoseChart && data.pitch_values) {
        headPoseChart.data.labels = labels;
        headPoseChart.data.datasets[0].data = data.pitch_values;
        headPoseChart.data.datasets[1].data = data.yaw_values || [];
        headPoseChart.data.datasets[2].data = data.roll_values || [];
        headPoseChart.update('none');
    }

    if (ppgChart && data.ppg_signal) {
        ppgChart.data.labels = labels;
        ppgChart.data.datasets[0].data = data.ppg_signal;
        ppgChart.update('none');
    }
}

// ===== 绘制结果页面图表 (完整时序数据) =====
function renderResultCharts(resultData) {
    // EAR/MAR组合图
    const earMarCtx = document.getElementById('ear-mar-chart');
    if (earMarCtx && resultData.fatigue) {
        const f = resultData.fatigue;
        new Chart(earMarCtx, {
            type: 'line',
            data: {
                labels: f.ear_values ? f.ear_values.map((_, i) => i) : [],
                datasets: [
                    {
                        label: 'EAR',
                        data: f.ear_values || [],
                        borderColor: '#6C47FF',
                        backgroundColor: 'rgba(108, 71, 255, 0.08)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.3,
                        fill: true,
                        yAxisID: 'y'
                    },
                    {
                        label: 'MAR',
                        data: f.mar_values || [],
                        borderColor: '#FFB547',
                        backgroundColor: 'rgba(255, 181, 71, 0.08)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.3,
                        fill: true,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: '帧', color: '#8888AA' },
                        ticks: { color: '#5A5A7A' },
                        grid: { color: 'rgba(108, 71, 255, 0.08)' }
                    },
                    y: {
                        type: 'linear',
                        position: 'left',
                        min: 0, max: 0.5,
                        title: { display: true, text: 'EAR', color: '#6C47FF' },
                        ticks: { color: '#6C47FF' },
                        grid: { color: 'rgba(108, 71, 255, 0.08)' }
                    },
                    y1: {
                        type: 'linear',
                        position: 'right',
                        min: 0, max: 1.0,
                        title: { display: true, text: 'MAR', color: '#FFB547' },
                        ticks: { color: '#FFB547' },
                        grid: { drawOnChartArea: false }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#8888AA' } }
                },
                animation: { duration: 200 }
            }
        });
    }

    // 头部姿态折线图
    const hpCtx = document.getElementById('head-pose-line-chart');
    if (hpCtx && resultData.head_pose) {
        const hp = resultData.head_pose;
        const labels = hp.pitch_values ? hp.pitch_values.map((_, i) => i) : [];
        new Chart(hpCtx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Pitch', data: hp.pitch_values || [], borderColor: '#6C47FF', borderWidth: 2, pointRadius: 0, tension: 0.3 },
                    { label: 'Yaw',   data: hp.yaw_values   || [], borderColor: '#00D4AA', borderWidth: 2, pointRadius: 0, tension: 0.3 },
                    { label: 'Roll',  data: hp.roll_values  || [], borderColor: '#FFB547', borderWidth: 2, pointRadius: 0, tension: 0.3 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: '帧', color: '#8888AA' },
                        ticks: { color: '#5A5A7A' },
                        grid: { color: 'rgba(108, 71, 255, 0.08)' }
                    },
                    y: {
                        title: { display: true, text: '角度 (度)', color: '#8888AA' },
                        ticks: { color: '#5A5A7A' },
                        grid: { color: 'rgba(108, 71, 255, 0.08)' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#8888AA' } }
                },
                animation: { duration: 200 }
            }
        });
    }

    // PPG波形图
    const ppgCtx = document.getElementById('ppg-result-chart');
    if (ppgCtx && resultData.physiological && resultData.physiological.ppg_signal) {
        new Chart(ppgCtx, {
            type: 'line',
            data: {
                labels: resultData.physiological.ppg_signal.map((_, i) => i),
                datasets: [{
                    label: 'PPG',
                    data: resultData.physiological.ppg_signal,
                    borderColor: '#FF4560',
                    backgroundColor: 'rgba(255, 69, 96, 0.08)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.3,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: '帧', color: '#8888AA' },
                        ticks: { color: '#5A5A7A' },
                        grid: { color: 'rgba(108, 71, 255, 0.08)' }
                    },
                    y: {
                        title: { display: true, text: '幅值', color: '#8888AA' },
                        ticks: { color: '#5A5A7A' },
                        grid: { color: 'rgba(108, 71, 255, 0.08)' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#8888AA' } }
                },
                animation: { duration: 200 }
            }
        });
    }
}
