/* Kaala Web UI — Navigation, WebSocket, and global state */

const API_BASE = '/api';

let ws = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_DELAY = 30000;

/* Navigation */
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const panel = link.dataset.panel;
        switchPanel(panel);
    });
});

function switchPanel(name) {
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelector(`.nav-link[data-panel="${name}"]`).classList.add('active');
    document.getElementById(`panel-${name}`).classList.add('active');

    if (name === 'goals') loadGoals();
    if (name === 'schedules') loadSchedules();
    if (name === 'history') loadHistory();
}

/* WebSocket */
function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws`);

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleWSEvent(msg.event, msg.data);
    };

    ws.onclose = () => {
        reconnectAttempts++;
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), MAX_RECONNECT_DELAY);
        setTimeout(connectWS, delay);
    };

    ws.onopen = () => {
        reconnectAttempts = 0;
    };
}

function handleWSEvent(event, data) {
    if (event === 'reminder') {
        showToast('reminder', 'Reminder', data.message || JSON.stringify(data));
    } else if (event === 'scheduled_prompt_fired') {
        showToast('prompt', 'Scheduled Prompt', data.message || JSON.stringify(data));
        const activePanel = document.querySelector('.panel.active')?.id;
        if (activePanel === 'panel-goals') loadGoals();
        if (activePanel === 'panel-schedules') loadSchedules();
    }
}

/* Toast notifications */
function showToast(type, title, message) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<div class="toast-title">${title}</div><div>${escapeHtml(message)}</div>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

/* Utility */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function apiFetch(path, options = {}) {
    const res = await fetch(`${API_BASE}${path}`, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Request failed');
    }
    return res.json();
}

function formatTime(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    return d.toLocaleString();
}

function formatDate(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString();
}

/* Initialize */
connectWS();