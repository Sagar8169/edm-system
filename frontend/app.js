/**
 * EDM System — Frontend Application Logic
 * Connects to Flask backend API for real pipeline data.
 * Handles data visualization, student table, SHAP explanations,
 * fairness charts, HITL overrides, and interactive UI.
 */

// ============================================================
// Configuration
// ============================================================
const API_BASE = window.location.origin + '/api';

// ============================================================
// State
// ============================================================
let STATE = {
    overview: null,
    students: [],
    totalStudents: 0,
    totalPages: 1,
    currentPage: 1,
    perPage: 20,
    sortField: 'riskScore',
    sortDir: 'desc',
    selectedStudent: null,
    clusters: null,
    fairness: null,
    report: null,
    feedbackLog: [],
    connected: false,
};

// ============================================================
// API Client
// ============================================================
async function api(endpoint, options = {}) {
    try {
        const res = await fetch(`${API_BASE}${endpoint}`, {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error(`API error (${endpoint}):`, err);
        throw err;
    }
}

// ============================================================
// Initialization
// ============================================================
document.addEventListener('DOMContentLoaded', async () => {
    showLoading(true);
    initNavigation();
    initHeroAnimations();

    try {
        // Check health first
        const health = await api('/health');
        STATE.connected = true;
        updateConnectionStatus(true, health.all_components_loaded);

        // Load all data in parallel
        const [overview, studentsRes, clustersRes, fairnessRes, reportRes, feedbackRes] = await Promise.all([
            api('/overview'),
            api(`/students?page=1&per_page=${STATE.perPage}&sort=${STATE.sortField}&dir=${STATE.sortDir}`),
            api('/clusters').catch(() => null),
            api('/fairness').catch(() => null),
            api('/report').catch(() => null),
            api('/feedback').catch(() => []),
        ]);

        STATE.overview = overview;
        STATE.students = studentsRes.students;
        STATE.totalStudents = studentsRes.total;
        STATE.totalPages = studentsRes.totalPages;
        STATE.clusters = clustersRes;
        STATE.fairness = fairnessRes;
        STATE.report = reportRes;
        STATE.feedbackLog = Array.isArray(feedbackRes) ? feedbackRes : [];

        // Render everything
        renderOverview();
        initPerformanceRings();
        initCharts();
        initFeatureBars();
        renderStudentTable();
        initStudentTableEvents();
        initXAI();
        renderClusters();
        renderFairness();
        renderReport();
        renderFeedbackLog();
        initScrollObserver();

    } catch (err) {
        console.error('Failed to connect to backend:', err);
        STATE.connected = false;
        updateConnectionStatus(false, false);
        showToast('⚠️', 'Backend not connected. Start with: python backend/app.py');
    } finally {
        showLoading(false);
    }
});

// ============================================================
// Connection Status
// ============================================================
function updateConnectionStatus(connected, allLoaded) {
    const indicator = document.getElementById('connection-status');
    if (!indicator) return;

    if (connected && allLoaded) {
        indicator.className = 'connection-indicator connected';
        indicator.title = 'Connected — All components loaded';
        indicator.innerHTML = '<span class="conn-dot"></span> Live';
    } else if (connected) {
        indicator.className = 'connection-indicator partial';
        indicator.title = 'Connected — Some components missing';
        indicator.innerHTML = '<span class="conn-dot"></span> Partial';
    } else {
        indicator.className = 'connection-indicator disconnected';
        indicator.title = 'Disconnected — Backend not running';
        indicator.innerHTML = '<span class="conn-dot"></span> Offline';
    }
}

function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
    }
}

// ============================================================
// Overview Rendering
// ============================================================
function renderOverview() {
    const o = STATE.overview;
    if (!o) return;

    // Hero stats
    animateCounter('stat-students', o.totalStudents);
    const xgb = o.metrics?.xgboost || {};
    animateCounter('stat-accuracy', (xgb.accuracy || 0) * 100, 1, '%');
    animateCounter('stat-features', o.nFeatures || 0);
    animateCounter('stat-fairness', o.afi || 0, 2);

    // KPI cards
    setText('kpi-total', (o.totalStudents || 0).toLocaleString());
    setText('kpi-atrisk', (o.atRisk || 0).toLocaleString());
    setText('kpi-safe', (o.onTrack || 0).toLocaleString());

    // Update trend text
    const atRiskPct = o.totalStudents ? ((o.atRisk / o.totalStudents) * 100).toFixed(1) : '0';
    const onTrackPct = o.totalStudents ? ((o.onTrack / o.totalStudents) * 100).toFixed(1) : '0';
    const trendElements = document.querySelectorAll('.kpi-card');
    if (trendElements[1]) trendElements[1].querySelector('.kpi-trend').textContent = `${atRiskPct}% of total`;
    if (trendElements[2]) trendElements[2].querySelector('.kpi-trend').textContent = `${onTrackPct}% of total`;

    // Update performance ring labels
    const metricMap = {
        'ring-accuracy': xgb.accuracy,
        'ring-precision': xgb.precision,
        'ring-recall': xgb.recall,
        'ring-f1': xgb.f1,
        'ring-roc': xgb.roc_auc,
    };
    for (const [id, val] of Object.entries(metricMap)) {
        const card = document.getElementById(id)?.closest('.perf-card');
        if (card && val != null) {
            card.querySelector('.ring-label').textContent = (val * 100).toFixed(1) + '%';
        }
    }
}

function animateCounter(containerId, target, decimals = 0, suffix = '') {
    const container = document.getElementById(containerId);
    if (!container) return;
    const el = container.querySelector('.stat-number') || container;
    const duration = 2000;
    const start = performance.now();

    function update(now) {
        const progress = Math.min((now - start) / duration, 1);
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = target * easeOut;
        if (decimals > 0) {
            el.textContent = current.toFixed(decimals) + suffix;
        } else {
            el.textContent = Math.floor(current).toLocaleString() + suffix;
        }
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

// ============================================================
// Multi-Page Routing & Navigation (Matching Streamlit Pages)
// ============================================================
const PAGE_CONFIG = {
    'home': { title: 'Student Development', icon: '🎓' },
    'overview': { title: 'Dashboard', icon: '📊' },
    'student-list': { title: 'Student List', icon: '📋' },
    'student-detail': { title: 'Student Detail', icon: '🔍' },
    'personalization': { title: 'Personalization', icon: '🎯' },
    'fairness': { title: 'Fairness', icon: '⚖️' },
    'report': { title: 'Report', icon: '📝' },
};

const PAGE_ALIASES = {
    '': 'home',
    'home': 'home',
    'student-development': 'home',
    'overview': 'overview',
    'dashboard': 'overview',
    'students': 'student-list',
    'student-list': 'student-list',
    'explainability': 'student-detail',
    'student-detail': 'student-detail',
    'clusters': 'personalization',
    'personalization': 'personalization',
    'fairness': 'fairness',
    'report': 'report',
};

function resolvePageId(raw) {
    if (!raw) return 'home';
    const clean = raw.replace(/^#\/?/, '').trim().toLowerCase();
    return PAGE_ALIASES[clean] || (PAGE_CONFIG[clean] ? clean : 'home');
}

function navigateTo(pageId) {
    const resolved = resolvePageId(pageId);
    const targetHash = `#/${resolved}`;
    if (window.location.hash !== targetHash) {
        window.location.hash = targetHash;
    } else {
        renderPage(resolved);
    }
}

function renderPage(pageId) {
    const resolved = resolvePageId(pageId);
    const config = PAGE_CONFIG[resolved] || PAGE_CONFIG['home'];

    // 1. Hide all page views and show the active one
    document.querySelectorAll('.page-view').forEach(view => {
        const isCurrent = view.getAttribute('data-page') === resolved || view.id === `page-${resolved}`;
        view.classList.toggle('active', isCurrent);
    });

    // 2. Update sidebar navigation items
    document.querySelectorAll('.sidebar-nav-item').forEach(item => {
        const itemPage = item.getAttribute('data-page');
        item.classList.toggle('active', itemPage === resolved);
    });

    // 3. Update topbar heading
    const topbarTitle = document.getElementById('topbar-title');
    if (topbarTitle) {
        topbarTitle.innerHTML = `<span class="topbar-icon">${config.icon}</span><span class="topbar-heading">${config.title}</span>`;
    }

    // 4. Scroll to top
    window.scrollTo({ top: 0, behavior: 'instant' });

    // 5. Close mobile sidebar
    closeMobileSidebar();

    // 6. Ensure cards in active view are visible immediately
    const target = document.getElementById(`page-${resolved}`);
    if (target) {
        target.querySelectorAll('.kpi-card, .perf-card, .cluster-card, .chart-card, .report-card, .hypothesis-card').forEach(el => {
            el.classList.add('visible');
            el.style.opacity = '1';
            el.style.transform = 'none';
        });
    }

    // 7. Handle page-specific updates (resize charts, animate rings)
    if (resolved === 'overview') {
        setTimeout(initPerformanceRings, 100);
    }
    if (resolved === 'personalization' && STATE.clusters) {
        setTimeout(renderClusters, 50);
    }
    if (resolved === 'fairness' && STATE.fairness) {
        setTimeout(renderFairness, 50);
    }

    // Trigger Chart.js recalculation on visible container
    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 60);
}

function viewStudentDetail(studentId) {
    selectStudentForXAI(studentId);
    navigateTo('student-detail');
}

function closeMobileSidebar() {
    const sidebar = document.getElementById('app-sidebar');
    const backdrop = document.getElementById('sidebar-backdrop');
    if (sidebar) sidebar.classList.remove('open');
    if (backdrop) backdrop.classList.remove('active');
}

function openMobileSidebar() {
    const sidebar = document.getElementById('app-sidebar');
    const backdrop = document.getElementById('sidebar-backdrop');
    if (sidebar) sidebar.classList.add('open');
    if (backdrop) backdrop.classList.add('active');
}

function initNavigation() {
    // Hash change event listener
    window.addEventListener('hashchange', () => {
        renderPage(window.location.hash);
    });

    // Sidebar navigation link clicks
    document.querySelectorAll('.sidebar-nav-item').forEach(item => {
        item.addEventListener('click', e => {
            e.preventDefault();
            const targetPage = item.getAttribute('data-page');
            navigateTo(targetPage);
        });
    });

    // Mobile menu toggle & close
    const mobileBtn = document.getElementById('mobile-menu-btn');
    if (mobileBtn) {
        mobileBtn.addEventListener('click', openMobileSidebar);
    }

    const closeBtn = document.getElementById('sidebar-close-btn');
    if (closeBtn) {
        closeBtn.addEventListener('click', closeMobileSidebar);
    }

    const backdrop = document.getElementById('sidebar-backdrop');
    if (backdrop) {
        backdrop.addEventListener('click', closeMobileSidebar);
    }

    // Theme toggle
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'light' ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', next);
            showToast(next === 'light' ? '☀️' : '🌙', `${next.charAt(0).toUpperCase() + next.slice(1)} mode active`);
        });
    }

    // Initial page load
    const initialPage = resolvePageId(window.location.hash);
    renderPage(initialPage);
}

// ============================================================
// Hero Animations
// ============================================================
function initHeroAnimations() {
    // Neural network visualization
    const container = document.getElementById('neural-net');
    if (container) {
        const canvas = document.createElement('canvas');
        canvas.width = container.offsetWidth || 500;
        canvas.height = 400;
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        container.appendChild(canvas);

        const ctx = canvas.getContext('2d');
        const nodes = [];
        const connections = [];

        const layers = [4, 6, 8, 6, 3];
        layers.forEach((count, layerIdx) => {
            for (let i = 0; i < count; i++) {
                const x = (layerIdx + 1) * (canvas.width / (layers.length + 1));
                const y = (i + 1) * (canvas.height / (count + 1));
                nodes.push({ x, y, layer: layerIdx, radius: 4 + Math.random() * 3 });
            }
        });

        let idx = 0;
        for (let l = 0; l < layers.length - 1; l++) {
            const currentStart = idx;
            const currentEnd = idx + layers[l];
            const nextStart = currentEnd;
            const nextEnd = nextStart + layers[l + 1];

            for (let i = currentStart; i < currentEnd; i++) {
                for (let j = nextStart; j < nextEnd; j++) {
                    if (Math.random() > 0.3) {
                        connections.push({ from: i, to: j, opacity: 0.1 + Math.random() * 0.2 });
                    }
                }
            }
            idx = currentEnd;
        }

        function animateNetwork() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            connections.forEach(conn => {
                const from = nodes[conn.from];
                const to = nodes[conn.to];
                ctx.beginPath();
                ctx.moveTo(from.x, from.y);
                ctx.lineTo(to.x, to.y);
                ctx.strokeStyle = `rgba(102, 126, 234, ${conn.opacity})`;
                ctx.lineWidth = 0.5;
                ctx.stroke();
                conn.opacity = 0.05 + Math.sin(Date.now() * 0.001 + conn.from * 0.5) * 0.15;
            });

            nodes.forEach(node => {
                const glow = 0.3 + Math.sin(Date.now() * 0.002 + node.layer * 1.5) * 0.2;
                ctx.beginPath();
                ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(102, 126, 234, ${glow + 0.3})`;
                ctx.fill();

                ctx.beginPath();
                ctx.arc(node.x, node.y, node.radius + 4, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(102, 126, 234, ${glow * 0.2})`;
                ctx.fill();
            });

            requestAnimationFrame(animateNetwork);
        }
        animateNetwork();
    }
}

// ============================================================
// Performance Rings
// ============================================================
function initPerformanceRings() {
    const xgb = STATE.overview?.metrics?.xgboost || {};
    const circumference = 2 * Math.PI * 52;

    const rings = [
        { id: 'ring-accuracy', value: (xgb.accuracy || 0) * 100, color: '#667eea' },
        { id: 'ring-precision', value: (xgb.precision || 0) * 100, color: '#2ecc71' },
        { id: 'ring-recall', value: (xgb.recall || 0) * 100, color: '#e74c3c' },
        { id: 'ring-f1', value: (xgb.f1 || 0) * 100, color: '#f39c12' },
        { id: 'ring-roc', value: (xgb.roc_auc || 0) * 100, color: '#764ba2' },
    ];

    rings.forEach(ring => {
        const el = document.getElementById(ring.id);
        if (el) {
            el.style.strokeDasharray = circumference;
            el.style.strokeDashoffset = circumference;
            el.style.stroke = ring.color;

            setTimeout(() => {
                const offset = circumference - (ring.value / 100) * circumference;
                el.style.strokeDashoffset = offset;
            }, 500);
        }
    });
}

// ============================================================
// Charts (Chart.js)
// ============================================================
function initCharts() {
    Chart.defaults.color = '#8888a8';
    Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
    Chart.defaults.font.family = "'Inter', sans-serif";

    const o = STATE.overview;
    if (!o) return;

    // Distribution Donut
    const ctxDist = document.getElementById('chart-distribution');
    if (ctxDist && o.resultDistribution) {
        new Chart(ctxDist, {
            type: 'doughnut',
            data: {
                labels: o.resultDistribution.labels,
                datasets: [{
                    data: o.resultDistribution.values,
                    backgroundColor: o.resultDistribution.colors,
                    borderWidth: 0,
                    hoverOffset: 8,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '65%',
                plugins: {
                    legend: { position: 'right', labels: { padding: 16, usePointStyle: true, pointStyleWidth: 12 } }
                }
            }
        });
    }

    // Comparison Bar Chart
    const ctxComp = document.getElementById('chart-comparison');
    if (ctxComp && o.metrics) {
        const xgb = o.metrics.xgboost || {};
        const lr = o.metrics.baseline || {};
        const metricNames = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'];
        const xgbValues = [xgb.accuracy, xgb.precision, xgb.recall, xgb.f1, xgb.roc_auc];
        const lrValues = [lr.accuracy, lr.precision, lr.recall, lr.f1, lr.roc_auc];

        new Chart(ctxComp, {
            type: 'bar',
            data: {
                labels: metricNames,
                datasets: [
                    {
                        label: 'XGBoost',
                        data: xgbValues,
                        backgroundColor: 'rgba(102, 126, 234, 0.7)',
                        borderColor: '#667eea',
                        borderWidth: 1,
                        borderRadius: 6,
                    },
                    {
                        label: 'Logistic Regression',
                        data: lrValues,
                        backgroundColor: 'rgba(231, 76, 60, 0.5)',
                        borderColor: '#e74c3c',
                        borderWidth: 1,
                        borderRadius: 6,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 1, grid: { color: 'rgba(255,255,255,0.04)' } },
                    x: { grid: { display: false } }
                },
                plugins: {
                    legend: { labels: { usePointStyle: true, pointStyleWidth: 12 } }
                }
            }
        });
    }
}

// ============================================================
// Feature Importance Bars
// ============================================================
function initFeatureBars() {
    const container = document.getElementById('feature-bars');
    if (!container || !STATE.overview?.featureImportance) return;

    const features = STATE.overview.featureImportance;
    if (!features.length) return;

    const maxVal = Math.max(...features.map(f => f.mean_abs_shap || 0));

    features.forEach((feat, i) => {
        const pct = maxVal > 0 ? ((feat.mean_abs_shap || 0) / maxVal) * 100 : 0;
        const hue = 260 - (pct * 1.5);

        const row = document.createElement('div');
        row.className = 'feature-bar';
        row.style.animationDelay = `${i * 0.05}s`;

        row.innerHTML = `
            <span class="feature-bar-name">${feat.feature || feat.name}</span>
            <div class="feature-bar-track">
                <div class="feature-bar-fill" style="width: 0%; background: linear-gradient(90deg, hsl(${hue}, 70%, 55%), hsl(${hue}, 70%, 45%));">
                    ${(feat.mean_abs_shap || 0).toFixed(3)}
                </div>
            </div>
        `;
        container.appendChild(row);

        setTimeout(() => {
            row.querySelector('.feature-bar-fill').style.width = `${pct}%`;
        }, 300 + i * 80);
    });
}

// ============================================================
// Student Table
// ============================================================
function initStudentTableEvents() {
    // Search
    document.getElementById('search-input').addEventListener('input', () => {
        STATE.currentPage = 1;
        loadStudents();
    });

    // Filters
    document.getElementById('filter-risk').addEventListener('change', () => {
        STATE.currentPage = 1;
        loadStudents();
    });
    document.getElementById('filter-cluster').addEventListener('change', () => {
        STATE.currentPage = 1;
        loadStudents();
    });

    // Sort headers
    document.querySelectorAll('.data-table th[data-sort]').forEach(th => {
        th.addEventListener('click', () => {
            const field = th.dataset.sort;
            const fieldMap = {
                id: 'id', module: 'module', risk: 'riskScore', status: 'predictedAtRisk',
                clicks: 'totalClicks', score: 'avgScore', days: 'daysActive', cluster: 'cluster'
            };
            const apiField = fieldMap[field] || 'riskScore';
            if (STATE.sortField === apiField) {
                STATE.sortDir = STATE.sortDir === 'asc' ? 'desc' : 'asc';
            } else {
                STATE.sortField = apiField;
                STATE.sortDir = 'desc';
            }
            STATE.currentPage = 1;
            loadStudents();
        });
    });

    // Export
    document.getElementById('export-btn').addEventListener('click', exportStudentCSV);
}

async function loadStudents() {
    const search = document.getElementById('search-input').value.trim();
    const riskFilter = document.getElementById('filter-risk').value;
    const clusterFilter = document.getElementById('filter-cluster').value;

    let url = `/students?page=${STATE.currentPage}&per_page=${STATE.perPage}&sort=${STATE.sortField}&dir=${STATE.sortDir}`;
    if (search) url += `&search=${encodeURIComponent(search)}`;
    if (riskFilter !== 'all') url += `&risk=${riskFilter}`;
    if (clusterFilter !== 'all') url += `&cluster=${clusterFilter}`;

    try {
        const res = await api(url);
        STATE.students = res.students;
        STATE.totalStudents = res.total;
        STATE.totalPages = res.totalPages;
        STATE.currentPage = res.page;
        renderStudentTable();
    } catch (err) {
        showToast('⚠️', 'Failed to load students');
    }
}

function renderStudentTable() {
    const tbody = document.getElementById('student-tbody');
    const students = STATE.students;

    if (!students || !students.length) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;padding:40px;color:var(--text-muted);">No students found</td></tr>';
        renderPagination();
        return;
    }

    const clusterNames = ['High Eng.', 'Low Eng.', 'Inconsist.', 'Hi Effort'];

    tbody.innerHTML = students.map(s => {
        const level = getRiskLevel(s.riskScore);
        const levelLabel = level.charAt(0).toUpperCase() + level.slice(1);
        const riskColor = s.riskScore >= 0.7 ? '#e74c3c' : s.riskScore >= 0.5 ? '#f39c12' : s.riskScore >= 0.3 ? '#f1c40f' : '#2ecc71';
        const clusterLabel = s.cluster >= 0 && s.cluster < clusterNames.length ? `C${s.cluster}: ${clusterNames[s.cluster]}` : '—';

        return `<tr data-student-id="${s.id}" onclick="viewStudentDetail(${s.id})" style="cursor:pointer;" title="Click to view SHAP explanation">
            <td><strong>${s.id}</strong></td>
            <td>${s.module || '—'}</td>
            <td>
                <div style="display:flex;align-items:center;gap:8px;">
                    <div class="risk-score-bar"><div class="risk-score-fill" style="width:${s.riskScore*100}%;background:${riskColor};"></div></div>
                    <span style="font-family:var(--font-mono);font-weight:600;">${s.riskScore.toFixed(3)}</span>
                </div>
            </td>
            <td><span class="risk-badge risk-${level}">${levelLabel}</span></td>
            <td>${Math.round(s.totalClicks).toLocaleString()}</td>
            <td>${(s.avgScore || 0).toFixed(1)}</td>
            <td>${Math.round(s.daysActive)}</td>
            <td>${clusterLabel}</td>
            <td><button class="btn-secondary" style="padding:5px 12px;font-size:0.75rem;" onclick="event.stopPropagation();viewStudentDetail(${s.id})">🔍 View</button></td>
        </tr>`;
    }).join('');

    renderPagination();
}

function renderPagination() {
    const container = document.getElementById('pagination');
    const totalPages = STATE.totalPages;
    const currentPage = STATE.currentPage;
    let html = '';

    const start = Math.max(1, currentPage - 2);
    const end = Math.min(totalPages, currentPage + 2);

    if (currentPage > 1) html += `<button class="page-btn" onclick="goToPage(${currentPage-1})">‹</button>`;

    for (let i = start; i <= end; i++) {
        html += `<button class="page-btn ${i === currentPage ? 'active' : ''}" onclick="goToPage(${i})">${i}</button>`;
    }

    if (currentPage < totalPages) html += `<button class="page-btn" onclick="goToPage(${currentPage+1})">›</button>`;

    container.innerHTML = html;
}

function goToPage(page) {
    STATE.currentPage = page;
    loadStudents();
}

function getRiskLevel(score) {
    if (score >= 0.7) return 'critical';
    if (score >= 0.5) return 'high';
    if (score >= 0.3) return 'medium';
    return 'low';
}

function exportStudentCSV() {
    const headers = ['Student ID', 'Module', 'Risk Score', 'At Risk', 'Total Clicks', 'Avg Score', 'Days Active', 'Cluster'];
    const rows = STATE.students.map(s => [s.id, s.module, s.riskScore.toFixed(4), s.atRisk, Math.round(s.totalClicks), (s.avgScore||0).toFixed(1), Math.round(s.daysActive), s.cluster]);
    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'student_list_export.csv'; a.click();
    URL.revokeObjectURL(url);
    showToast('✅', 'Student list exported as CSV');
}

// ============================================================
// XAI / Explainability
// ============================================================
function initXAI() {
    const select = document.getElementById('xai-student-select');

    // Populate with first 100 students from the loaded list
    STATE.students.slice(0, 100).forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.id;
        opt.textContent = `${s.id} — ${s.module} (Risk: ${s.riskScore.toFixed(3)})`;
        select.appendChild(opt);
    });

    select.addEventListener('change', () => selectStudentForXAI(parseInt(select.value)));

    // HITL buttons
    document.getElementById('hitl-confirm').addEventListener('click', () => hitlAction('confirmed_at_risk'));
    document.getElementById('hitl-override').addEventListener('click', () => hitlAction('overridden_not_at_risk'));

    // Select first student
    if (STATE.students.length) selectStudentForXAI(STATE.students[0].id);
}

async function selectStudentForXAI(studentId) {
    const select = document.getElementById('xai-student-select');
    select.value = studentId;

    // Find student in current list for basic info
    const basicInfo = STATE.students.find(s => s.id === studentId);

    // Get detailed SHAP explanation from backend
    try {
        const explanation = await api(`/students/${studentId}/explain`);

        // Profile card
        document.getElementById('xai-avatar').textContent = studentId.toString().slice(-2);
        document.getElementById('xai-student-id').textContent = `Student #${studentId}`;
        document.getElementById('xai-module').textContent = `Module ${explanation.module || '?'}`;

        const riskColor = explanation.riskScore >= 0.7 ? '#e74c3c' : explanation.riskScore >= 0.5 ? '#f39c12' : '#2ecc71';
        document.getElementById('xai-risk-score').textContent = explanation.riskScore.toFixed(3);
        document.getElementById('xai-risk-score').style.color = riskColor;
        document.getElementById('xai-prediction').textContent = explanation.predictedAtRisk ? '⚠️ At Risk' : '✅ Not At Risk';
        document.getElementById('xai-actual').textContent = explanation.actualResult || '—';

        const clusterNames = ['High Engagement', 'Low Engagement', 'Inconsistent', 'High Effort'];
        if (basicInfo && basicInfo.cluster >= 0 && basicInfo.cluster < clusterNames.length) {
            document.getElementById('xai-cluster').textContent = `C${basicInfo.cluster}: ${clusterNames[basicInfo.cluster]}`;
        } else {
            document.getElementById('xai-cluster').textContent = '—';
        }

        // SHAP Waterfall
        renderShapWaterfall(explanation.features);

        // Risk/Protective factors
        const riskFactors = explanation.features.filter(f => f.shapValue > 0).slice(0, 4);
        const protectiveFactors = explanation.features.filter(f => f.shapValue < 0).slice(0, 4);

        document.getElementById('risk-factors-list').innerHTML = riskFactors.map(f =>
            `<li>🔴 <strong>${f.name}</strong> → +${f.shapValue.toFixed(3)} risk</li>`
        ).join('') || '<li>No significant risk factors</li>';

        document.getElementById('protective-factors-list').innerHTML = protectiveFactors.map(f =>
            `<li>🟢 <strong>${f.name}</strong> → ${f.shapValue.toFixed(3)} risk</li>`
        ).join('') || '<li>No significant protective factors</li>';

        // Intervention
        const interventions = [
            'Continue current path. Offer advanced resources and peer mentoring roles.',
            'URGENT: Schedule immediate 1-on-1 tutorial support. Provide structured study plans.',
            'Targeted study skills workshop. Peer mentoring pairing. Formative assessment practice.',
            'Review assessment preparation strategies. Alternative learning materials. Learning support referral.'
        ];
        const cluster = basicInfo?.cluster ?? 0;
        document.getElementById('intervention-text').textContent = interventions[Math.min(cluster, 3)];

        // Store for HITL
        STATE.selectedStudent = { ...basicInfo, ...explanation };

    } catch (err) {
        console.error('Failed to load explanation:', err);
        showToast('⚠️', `Could not load explanation for student ${studentId}`);
    }

    // Scroll to XAI section if needed
    const xaiSection = document.getElementById('explainability');
    if (!isInViewport(xaiSection)) {
        xaiSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

function renderShapWaterfall(features) {
    const container = document.getElementById('shap-waterfall');
    container.innerHTML = '';

    if (!features || !features.length) {
        container.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:20px;">No SHAP data available</p>';
        return;
    }

    const topValues = features.slice(0, 12);
    const maxAbsVal = Math.max(...topValues.map(f => Math.abs(f.shapValue)), 0.01);

    topValues.forEach((sv, i) => {
        const row = document.createElement('div');
        row.className = 'shap-bar-row';
        row.style.animationDelay = `${i * 0.05}s`;

        const barWidth = (Math.abs(sv.shapValue) / maxAbsVal) * 45;
        const isPositive = sv.shapValue > 0;
        const barStyle = isPositive
            ? `left:50%;width:${barWidth}%;`
            : `right:50%;width:${barWidth}%;`;

        row.innerHTML = `
            <span class="shap-feature-name">${sv.name}</span>
            <div class="shap-bar-container">
                <div class="shap-center-line"></div>
                <div class="shap-bar ${isPositive ? 'positive' : 'negative'}" style="${barStyle}"></div>
            </div>
            <span class="shap-value" style="color:${isPositive ? '#e74c3c' : '#2ecc71'}">${sv.shapValue > 0 ? '+' : ''}${sv.shapValue.toFixed(4)}</span>
        `;

        container.appendChild(row);
    });
}

async function hitlAction(decision) {
    if (!STATE.selectedStudent) return;

    const notes = document.getElementById('hitl-notes').value;

    try {
        await api('/feedback', {
            method: 'POST',
            body: JSON.stringify({
                studentId: STATE.selectedStudent.id || STATE.selectedStudent.studentId,
                module: STATE.selectedStudent.module,
                originalPrediction: STATE.selectedStudent.predictedAtRisk || STATE.selectedStudent.atRisk,
                riskScore: STATE.selectedStudent.riskScore,
                decision,
                notes,
            }),
        });

        const feedback = document.getElementById('hitl-feedback');
        feedback.className = 'hitl-feedback show';
        feedback.style.background = decision === 'confirmed_at_risk' ? 'var(--danger-bg)' : 'var(--success-bg)';
        feedback.style.color = decision === 'confirmed_at_risk' ? 'var(--danger)' : 'var(--success)';
        feedback.textContent = decision === 'confirmed_at_risk'
            ? '✅ Risk assessment confirmed and logged!'
            : '❌ Override logged — Student marked as not at risk.';

        document.getElementById('hitl-notes').value = '';
        showToast('📝', `Feedback logged for Student #${STATE.selectedStudent.id || STATE.selectedStudent.studentId}`);

        // Refresh feedback log
        const feedbackData = await api('/feedback');
        STATE.feedbackLog = Array.isArray(feedbackData) ? feedbackData : [];
        renderFeedbackLog();

        setTimeout(() => { feedback.className = 'hitl-feedback'; }, 4000);

    } catch (err) {
        showToast('⚠️', 'Failed to save feedback');
    }
}

// ============================================================
// Clusters
// ============================================================
function renderClusters() {
    if (!STATE.clusters) return;

    STATE.clusters.clusters.forEach(c => {
        const el = id => document.getElementById(id);
        const countEl = el(`c${c.id}-count`);
        const clicksEl = el(`c${c.id}-clicks`);
        const scoreEl = el(`c${c.id}-score`);

        if (countEl) countEl.textContent = c.count.toLocaleString();
        if (clicksEl) clicksEl.textContent = Math.round(c.avgClicks).toLocaleString();
        if (scoreEl) scoreEl.textContent = (c.avgScore || 0).toFixed(1);
    });

    // PCA Scatter
    const ctxCluster = document.getElementById('chart-clusters');
    if (ctxCluster && STATE.clusters.pcaData?.length) {
        const clusterColors = ['#2ecc71', '#e74c3c', '#f39c12', '#e67e22'];
        const clusterNames = ['C0: High Engagement', 'C1: Low Engagement', 'C2: Inconsistent', 'C3: High Effort'];
        const datasets = [];

        const uniqueClusters = [...new Set(STATE.clusters.pcaData.map(p => p.cluster))].sort();
        uniqueClusters.forEach(c => {
            const points = STATE.clusters.pcaData
                .filter(p => p.cluster === c)
                .map(p => ({ x: p.x, y: p.y }));

            datasets.push({
                label: clusterNames[c] || `Cluster ${c}`,
                data: points,
                backgroundColor: (clusterColors[c] || '#999') + '80',
                borderColor: clusterColors[c] || '#999',
                borderWidth: 1,
                pointRadius: 2,
                pointHoverRadius: 5,
            });
        });

        new Chart(ctxCluster, {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top', labels: { usePointStyle: true, pointStyleWidth: 12 } }
                },
                scales: {
                    x: { title: { display: true, text: 'PCA Component 1' }, grid: { color: 'rgba(255,255,255,0.04)' } },
                    y: { title: { display: true, text: 'PCA Component 2' }, grid: { color: 'rgba(255,255,255,0.04)' } }
                }
            }
        });
    }
}

// ============================================================
// Fairness
// ============================================================
function renderFairness() {
    if (!STATE.fairness) return;

    const afi = STATE.fairness.afi || 0;

    // AFI Gauge Animation
    const arcEl = document.getElementById('afi-arc');
    if (arcEl) {
        const totalDash = 251.33;
        setTimeout(() => {
            arcEl.style.strokeDashoffset = totalDash * (1 - afi);
        }, 500);
    }

    // AFI value display
    setText('afi-value', afi.toFixed(3));

    // AFI Interpretation
    const interpEl = document.getElementById('afi-interpretation');
    if (interpEl) {
        if (afi >= 0.85) interpEl.textContent = `✅ Excellent fairness (AFI = ${afi.toFixed(3)}). Minimal disparity across demographics.`;
        else if (afi >= 0.7) interpEl.textContent = `⚠️ Acceptable fairness (AFI = ${afi.toFixed(3)}). Some disparity within bounds.`;
        else interpEl.textContent = `🚨 Significant disparity (AFI = ${afi.toFixed(3)}). Investigate bias sources.`;
    }

    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
        });
    });

    // Render fairness charts for each attribute
    const attrMap = {};
    if (STATE.fairness.attributes) {
        for (const [key, data] of Object.entries(STATE.fairness.attributes)) {
            if (key.includes('gender')) attrMap['gender'] = data;
            else if (key.includes('disability')) attrMap['disability'] = data;
            else if (key.includes('age')) attrMap['age'] = data;
        }
    }

    ['gender', 'disability', 'age'].forEach(attr => {
        const data = attrMap[attr];
        if (data) {
            renderFairnessChart(attr, data);
            renderFairnessTable(attr, data);
        }
    });
}

function renderFairnessChart(attr, data) {
    const canvas = document.getElementById(`chart-fairness-${attr}`);
    if (!canvas || !data) return;

    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: data.groups,
            datasets: [
                { label: 'Accuracy', data: data.accuracy, backgroundColor: 'rgba(52,152,219,0.7)', borderRadius: 4 },
                { label: 'Precision', data: data.precision, backgroundColor: 'rgba(46,204,113,0.7)', borderRadius: 4 },
                { label: 'Recall', data: data.recall, backgroundColor: 'rgba(231,76,60,0.7)', borderRadius: 4 },
                { label: 'F1', data: data.f1, backgroundColor: 'rgba(243,156,18,0.7)', borderRadius: 4 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, max: 1, grid: { color: 'rgba(255,255,255,0.04)' } },
                x: { grid: { display: false } }
            },
            plugins: { legend: { labels: { usePointStyle: true } } }
        }
    });
}

function renderFairnessTable(attr, data) {
    const container = document.getElementById(`table-${attr}`);
    if (!container || !data) return;

    let html = `<table class="data-table">
        <thead><tr><th>Group</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>`;

    data.groups.forEach((group, i) => {
        html += `<tr>
            <td><strong>${group}</strong></td>
            <td>${data.accuracy[i].toFixed(4)}</td>
            <td>${data.precision[i].toFixed(4)}</td>
            <td>${data.recall[i].toFixed(4)}</td>
            <td>${data.f1[i].toFixed(4)}</td>
        </tr>`;
    });

    // Gap row
    const metrics = ['accuracy', 'precision', 'recall', 'f1'];
    html += '<tr style="background:rgba(255,255,255,0.02);font-weight:600;"><td>Gap (max-min)</td>';
    metrics.forEach(m => {
        const gap = Math.max(...data[m]) - Math.min(...data[m]);
        const color = gap < 0.05 ? '#2ecc71' : gap < 0.1 ? '#f39c12' : '#e74c3c';
        html += `<td style="color:${color}">${gap.toFixed(4)}</td>`;
    });
    html += '</tr>';

    html += '</tbody></table>';
    container.innerHTML = html;
}

// ============================================================
// Report
// ============================================================
function renderReport() {
    if (!STATE.report) return;

    // KPI Table
    const kpiBody = document.getElementById('kpi-tbody');
    if (kpiBody && STATE.report.kpis) {
        kpiBody.innerHTML = STATE.report.kpis.map(kpi => `
            <tr>
                <td><strong>${kpi.name}</strong></td>
                <td style="font-family:var(--font-mono)">${kpi.xgb}</td>
                <td style="font-family:var(--font-mono);color:var(--text-muted)">${kpi.lr}</td>
                <td>${kpi.target}</td>
                <td class="${kpi.met ? 'status-met' : 'status-miss'}">${kpi.met ? '✅ Met' : '❌ Below'}</td>
            </tr>
        `).join('');
    }

    // Hypothesis Test
    const ht = STATE.report.hypothesisTest;
    if (ht) {
        setText('hyp-statistic', (ht.statistic || 0).toFixed(4));
        const pVal = ht.p_value != null ? ht.p_value : ht.pValue;
        setText('hyp-pvalue', pVal != null ? (pVal === 0 ? '< 1e-10' : pVal.toExponential(4)) : '—');

        const significant = ht.significant_at_005 != null ? ht.significant_at_005 : ht.significant;
        const resultEl = document.getElementById('hyp-result');
        if (resultEl) {
            resultEl.textContent = significant ? 'Significant' : 'Not Significant';
            resultEl.style.color = significant ? 'var(--success)' : 'var(--danger)';
        }

        const verdict = document.getElementById('hyp-verdict');
        if (verdict) {
            verdict.style.background = significant ? 'var(--success-bg)' : 'var(--warning-bg)';
            verdict.style.border = `1px solid ${significant ? 'rgba(46,204,113,0.3)' : 'rgba(243,156,18,0.3)'}`;
            verdict.style.color = significant ? 'var(--success)' : 'var(--warning)';
            const xgbCorrect = ht.xgb_only_correct || ht.xgbCorrect || 0;
            verdict.innerHTML = significant
                ? `<strong>✅ REJECT H₀</strong> at α=0.05. XGBoost is significantly better than baseline (p=${pVal != null ? (pVal === 0 ? '< 1e-10' : pVal.toExponential(2)) : '?'}). XGBoost corrected ${xgbCorrect} predictions that baseline missed.`
                : `⚠️ <strong>FAIL TO REJECT H₀.</strong> No significant difference at α=0.05.`;
        }
    }

    // Export buttons
    document.getElementById('export-report-txt')?.addEventListener('click', () => {
        const text = STATE.report.fullReport || 'No report available.';
        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a'); a.href = url; a.download = 'edm_report.txt'; a.click();
        showToast('📄', 'Report downloaded');
    });

    document.getElementById('export-kpi-csv')?.addEventListener('click', () => {
        if (!STATE.report.kpis) return;
        const rows = [['KPI', 'XGBoost', 'Baseline', 'Target', 'Met']];
        STATE.report.kpis.forEach(k => rows.push([k.name, k.xgb, k.lr, k.target, k.met ? 'Yes' : 'No']));
        downloadCSV(rows, 'kpi_summary.csv');
    });

    document.getElementById('export-fairness-csv')?.addEventListener('click', () => {
        if (!STATE.fairness?.attributes) return;
        const rows = [['Attribute', 'Group', 'Accuracy', 'Precision', 'Recall', 'F1']];
        Object.entries(STATE.fairness.attributes).forEach(([attr, data]) => {
            data.groups.forEach((g, i) => {
                rows.push([attr, g, data.accuracy[i], data.precision[i], data.recall[i], data.f1[i]]);
            });
        });
        downloadCSV(rows, 'fairness_report.csv');
    });
}

function downloadCSV(rows, filename) {
    const csv = rows.map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
    showToast('📊', `${filename} exported`);
}

// ============================================================
// Feedback Log
// ============================================================
function renderFeedbackLog() {
    const tbody = document.getElementById('feedback-tbody');
    if (!tbody) return;

    if (!STATE.feedbackLog.length) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;padding:20px;color:var(--text-muted);">No feedback logged yet. Use the Teacher Override panel to add entries.</td></tr>';
        return;
    }

    tbody.innerHTML = STATE.feedbackLog.slice().reverse().map(entry => `
        <tr>
            <td>${entry.timestamp ? new Date(entry.timestamp).toLocaleString() : '—'}</td>
            <td>${entry.student_id || entry.studentId || '—'}</td>
            <td>${entry.original_prediction != null ? (entry.original_prediction ? 'At Risk' : 'Not At Risk') : '—'}</td>
            <td><span class="risk-badge ${(entry.teacher_decision || entry.decision || '').includes('confirmed') ? 'risk-critical' : 'risk-low'}">${(entry.teacher_decision || entry.decision || '').replace(/_/g, ' ')}</span></td>
            <td>${entry.notes || '—'}</td>
        </tr>
    `).join('');
}

// ============================================================
// Scroll Observer (Animate on Scroll)
// ============================================================
function initScrollObserver() {
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.kpi-card, .perf-card, .cluster-card, .chart-card, .report-card, .hypothesis-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });

    const style = document.createElement('style');
    style.textContent = '.visible { opacity: 1 !important; transform: translateY(0) !important; }';
    document.head.appendChild(style);
}

// ============================================================
// Utilities
// ============================================================
function showToast(icon, message) {
    const toast = document.getElementById('toast');
    toast.querySelector('.toast-icon').textContent = icon;
    toast.querySelector('.toast-msg').textContent = message;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3500);
}

function isInViewport(el) {
    const rect = el.getBoundingClientRect();
    return rect.top >= 0 && rect.top < window.innerHeight;
}

// Global functions for inline onclick handlers
window.navigateTo = navigateTo;
window.viewStudentDetail = viewStudentDetail;
window.selectStudentForXAI = selectStudentForXAI;
window.goToPage = goToPage;
