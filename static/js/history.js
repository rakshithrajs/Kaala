/* Kaala Web UI — History panel */

document.getElementById('history-refresh').addEventListener('click', loadHistory);
document.getElementById('history-agent').addEventListener('change', loadHistory);
document.getElementById('history-limit').addEventListener('change', loadHistory);

async function loadHistory() {
    const container = document.getElementById('history-list');
    container.innerHTML = '<div class="empty-state"><p>Loading history...</p></div>';

    const agent = document.getElementById('history-agent').value;
    const limit = document.getElementById('history-limit').value;

    try {
        const params = new URLSearchParams({ limit });
        if (agent) params.set('agent', agent);
        const entries = await apiFetch(`/history?${params}`);

        if (entries.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>No history entries found.</p></div>';
            return;
        }

        container.innerHTML = '';
        entries.forEach(entry => {
            container.appendChild(createHistoryEntry(entry));
        });
    } catch (err) {
        container.innerHTML = `<div class="empty-state"><p>Error: ${escapeHtml(err.message)}</p></div>`;
    }
}

function createHistoryEntry(entry) {
    const div = document.createElement('div');
    div.className = 'history-entry';

    const roleLabel = entry.role === 'user' ? 'User' : entry.agent;
    const content = entry.content.length > 500
        ? entry.content.substring(0, 500) + '...'
        : entry.content;

    div.innerHTML = `
        <div class="entry-header">
            <span class="entry-agent ${entry.agent}">${escapeHtml(entry.agent)} (${escapeHtml(entry.role)})</span>
            <span class="entry-time">${formatTime(entry.timestamp)}</span>
        </div>
        <div class="entry-content">${escapeHtml(content)}</div>
    `;
    return div;
}