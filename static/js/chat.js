/* Kaala Web UI — Chat panel */

const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');
let sending = false;

chatSend.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || sending) return;

    sending = true;
    chatSend.disabled = true;

    appendMessage('user', text);
    chatInput.value = '';

    const thinking = appendThinking();

    try {
        const result = await apiFetch('/chat', {
            method: 'POST',
            body: JSON.stringify({ message: text }),
        });
        thinking.remove();
        renderResult(result);
    } catch (err) {
        thinking.remove();
        appendMessage('system', `Error: ${err.message}`);
    } finally {
        sending = false;
        chatSend.disabled = false;
        chatInput.focus();
    }
}

function appendMessage(role, content, agent) {
    const div = document.createElement('div');
    div.className = `chat-msg ${role}`;

    if (agent) {
        const agentSpan = document.createElement('div');
        agentSpan.className = `msg-agent ${agent.toLowerCase()}`;
        agentSpan.textContent = agent;
        div.appendChild(agentSpan);
    }

    const contentDiv = document.createElement('div');
    contentDiv.textContent = content;
    div.appendChild(contentDiv);

    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}

function appendThinking() {
    const div = document.createElement('div');
    div.className = 'chat-msg thinking';
    div.textContent = 'Thinking...';
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}

function renderResult(result) {
    const type = result.type;

    if (type === 'conversation') {
        appendMessage('assistant', result.response || '', result.signature || 'Assistant');
    } else if (type === 'clarification') {
        appendMessage('assistant', result.response || '', result.signature || 'Iccha');
        if (result.goals?.length) {
            const goalsDiv = document.createElement('div');
            goalsDiv.className = 'chat-msg system';
            goalsDiv.innerHTML = `<strong>Clarification needed for:</strong><ul class="goal-list">${result.goals.map(g => `<li>${escapeHtml(g)}</li>`).join('')}</ul>`;
            chatMessages.appendChild(goalsDiv);
        }
    } else if (type === 'immediate') {
        appendMessage('system', `Immediate action taken for: ${(result.goals || []).join(', ')}`);
        if (result.clarification) {
            appendMessage('assistant', result.clarification, 'Iccha');
        }
    } else if (type === 'goals_scheduled') {
        let html = `<strong>${escapeHtml(result.message || 'Goals scheduled')}</strong>`;
        if (result.goals?.length) {
            html += `<ul class="goal-list">${result.goals.map(g => `<li>${escapeHtml(g)}</li>`).join('')}</ul>`;
        }
        const div = document.createElement('div');
        div.className = 'chat-msg system';
        div.innerHTML = html;
        chatMessages.appendChild(div);
        if (result.clarification) {
            appendMessage('assistant', result.clarification, 'Iccha');
        }
    } else if (type === 'executed') {
        appendMessage('assistant', result.result || '', 'Karma');
        const div = document.createElement('div');
        div.className = 'chat-msg system';
        div.innerHTML = `<strong>Action:</strong> ${escapeHtml(result.action || 'N/A')} &middot; <strong>Tool:</strong> ${escapeHtml(result.tool || 'N/A')}`;
        chatMessages.appendChild(div);
    } else if (type === 'reminder') {
        appendMessage('system', `Reminder: ${result.message || ''}`);
    } else if (type === 'error') {
        appendMessage('system', `Error: ${result.error || 'Unknown error'}`);
    } else {
        appendMessage('system', JSON.stringify(result, null, 2));
    }

    chatMessages.scrollTop = chatMessages.scrollHeight;
}