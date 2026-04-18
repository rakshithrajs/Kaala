/* Kaala Web UI — Schedules panel */

let currentScheduleStatus = 'pending';

document.querySelectorAll('#schedules-tabs .tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('#schedules-tabs .tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        currentScheduleStatus = tab.dataset.status;
        loadSchedules();
    });
});

async function loadSchedules() {
    const container = document.getElementById('schedules-list');
    container.innerHTML = '<div class="empty-state"><p>Loading prompts...</p></div>';

    try {
        const params = new URLSearchParams();
        if (currentScheduleStatus) params.set('status', currentScheduleStatus);
        const schedules = await apiFetch(`/schedules?${params}`);

        if (schedules.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>No scheduled prompts found.</p></div>';
            return;
        }

        container.innerHTML = '';
        schedules.forEach(sched => {
            container.appendChild(createScheduleCard(sched));
        });
    } catch (err) {
        container.innerHTML = `<div class="empty-state"><p>Error: ${escapeHtml(err.message)}</p></div>`;
    }
}

function createScheduleCard(sched) {
    const card = document.createElement('div');
    card.className = 'card';

    const statusBadge = `<span class="badge badge-${sched.status}">${sched.status.replace('_', ' ')}</span>`;
    const typeBadge = `<span class="badge badge-${sched.prompt_type}">${sched.prompt_type.replace('_', ' ')}</span>`;

    let actions = '';
    if (sched.status === 'pending') {
        actions = `<button class="btn-danger" onclick="cancelSchedule(${sched.id})">Cancel</button>`;
    }

    card.innerHTML = `
        <div class="card-header">
            <span class="card-title">${escapeHtml(sched.prompt)}</span>
            <div>${typeBadge} ${statusBadge}</div>
        </div>
        <div class="card-meta">
            <span>Scheduled: ${formatDate(sched.scheduled_for)}</span>
            ${sched.goal_id ? `<span>Goal #${sched.goal_id}</span>` : ''}
            ${sched.executed_at ? `<span>Executed: ${formatDate(sched.executed_at)}</span>` : ''}
        </div>
        ${actions ? `<div class="card-actions" style="margin-top:8px">${actions}</div>` : ''}
    `;
    return card;
}

async function cancelSchedule(id) {
    if (!confirm('Cancel this scheduled prompt?')) return;
    try {
        await apiFetch(`/schedules/${id}/cancel`, { method: 'POST' });
        loadSchedules();
    } catch (err) {
        showToast('prompt', 'Error', err.message);
    }
}