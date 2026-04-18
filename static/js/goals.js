/* Kaala Web UI — Goals panel */

let currentGoalStatus = '';

document.querySelectorAll('#goals-tabs .tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('#goals-tabs .tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        currentGoalStatus = tab.dataset.status;
        loadGoals();
    });
});

async function loadGoals() {
    const container = document.getElementById('goals-list');
    container.innerHTML = '<div class="empty-state"><p>Loading goals...</p></div>';

    try {
        const params = new URLSearchParams();
        if (currentGoalStatus) params.set('status', currentGoalStatus);
        const goals = await apiFetch(`/goals?${params}`);

        if (goals.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>No goals found.</p></div>';
            return;
        }

        container.innerHTML = '';
        goals.forEach(goal => {
            container.appendChild(createGoalCard(goal));
        });
    } catch (err) {
        container.innerHTML = `<div class="empty-state"><p>Error loading goals: ${escapeHtml(err.message)}</p></div>`;
    }
}

function createGoalCard(goal) {
    const card = document.createElement('div');
    card.className = 'card';

    const statusBadge = `<span class="badge badge-${goal.status}">${goal.status.replace('_', ' ')}</span>`;

    let actions = '';
    if (goal.status === 'pending' || goal.status === 'in_progress') {
        actions = `
            <button class="btn-success" onclick="completeGoal(${goal.id})">Complete</button>
            <button class="btn-danger" onclick="cancelGoal(${goal.id})">Cancel</button>
        `;
    }
    if (goal.status === 'completed' || goal.status === 'cancelled') {
        actions = `<button class="btn-danger" onclick="deleteGoal(${goal.id})">Delete</button>`;
    }

    card.innerHTML = `
        <div class="card-header">
            <span class="card-title">${escapeHtml(goal.goal)}</span>
            ${statusBadge}
        </div>
        ${goal.details ? `<div style="margin-bottom:8px;color:var(--text-muted);font-size:13px">${escapeHtml(goal.details)}</div>` : ''}
        <div class="card-meta">
            <span>Created: ${formatDate(goal.created_at)}</span>
            ${goal.completed_at ? `<span>Completed: ${formatDate(goal.completed_at)}</span>` : ''}
        </div>
        <div class="card-actions" style="margin-top:8px">
            ${actions}
        </div>
    `;
    return card;
}

async function completeGoal(id) {
    try {
        await apiFetch(`/goals/${id}`, {
            method: 'PATCH',
            body: JSON.stringify({ status: 'completed' }),
        });
        loadGoals();
    } catch (err) {
        showToast('prompt', 'Error', err.message);
    }
}

async function cancelGoal(id) {
    try {
        await apiFetch(`/goals/${id}`, {
            method: 'PATCH',
            body: JSON.stringify({ status: 'cancelled' }),
        });
        loadGoals();
    } catch (err) {
        showToast('prompt', 'Error', err.message);
    }
}

async function deleteGoal(id) {
    if (!confirm('Delete this goal?')) return;
    try {
        await apiFetch(`/goals/${id}`, { method: 'DELETE' });
        loadGoals();
    } catch (err) {
        showToast('prompt', 'Error', err.message);
    }
}