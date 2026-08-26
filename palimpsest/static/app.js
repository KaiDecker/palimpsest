let conversationId = null;

const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));

function addMessage(role, content, experienceId = null) {
  const empty = $('.empty'); if (empty) empty.remove();
  const template = $('#message-template').content.cloneNode(true); const article = template.querySelector('.message');
  article.classList.add(role); article.querySelector('.role').textContent = role === 'user' ? 'You' : 'Palimpsest'; article.querySelector('.content').textContent = content;
  if (role === 'assistant' && experienceId) {
    article.querySelectorAll('[data-rating]').forEach((button) => button.addEventListener('click', () => sendFeedback(experienceId, {rating: Number(button.dataset.rating)}, button)));
    article.querySelector('[data-edit]').addEventListener('click', () => { const edited = window.prompt('Edit this response:', content); if (edited && edited !== content) sendFeedback(experienceId, {edited_response: edited}, article.querySelector('[data-edit]')); });
    article.querySelector('[data-ab]').addEventListener('click', async () => {
      const button = article.querySelector('[data-ab]'); button.disabled = true; button.textContent = 'Loading…';
      try {
        const response = await fetch(`/api/experiences/${experienceId}/ab`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({model: $('#model-select').value || null})});
        const result = await response.json(); renderCandidates(article, experienceId, result.variants);
      } finally { button.disabled = false; button.textContent = 'A/B'; }
    });
  } else article.querySelector('.feedback').remove();
  $('#messages').appendChild(template); $('#messages').scrollTop = $('#messages').scrollHeight;
}

function renderCandidates(article, experienceId, variants) {
  const container = article.querySelector('.ab-candidates'); container.hidden = false;
  container.innerHTML = variants.map((variant) => `<button class="ab-candidate" data-choice="${variant.label}"><strong>Candidate ${variant.label}</strong>${escapeHtml(variant.content)}</button>`).join('');
  container.querySelectorAll('[data-choice]').forEach((button) => button.addEventListener('click', async () => {
    const chosen = button.dataset.choice; const rejected = chosen === 'A' ? 'B' : 'A';
    const selected = variants.find((variant) => variant.label === chosen).content; const other = variants.find((variant) => variant.label === rejected).content;
    await sendFeedback(experienceId, {chosen_response: selected, rejected_response: other}, button); container.querySelectorAll('button').forEach((item) => { item.disabled = true; }); button.textContent = `Selected ${chosen}`;
  }));
}

async function sendFeedback(experienceId, payload, button) { button.disabled = true; await fetch(`/api/experiences/${experienceId}/feedback`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)}); button.textContent = 'Saved'; loadSidebar(); }

async function loadSidebar() { const [memoryResponse, profileResponse] = await Promise.all([fetch('/api/memories'), fetch('/api/profile')]); const memories = await memoryResponse.json(); const profile = await profileResponse.json(); $('#memories').innerHTML = memories.length ? memories.map((m) => `<div class="list-item">${escapeHtml(m.content)}<small>${escapeHtml(m.type)} · confidence ${Math.round(m.confidence * 100)}%</small></div>`).join('') : '<span class="muted">No memories yet.</span>'; $('#profile').innerHTML = profile.length ? profile.map((p) => `<div class="list-item"><strong>${escapeHtml(p.key)}</strong><small>${escapeHtml(p.value)}</small></div>`).join('') : '<span class="muted">No preferences yet.</span>'; }

async function loadConversation() { if (!conversationId) return; const response = await fetch(`/api/conversations/${conversationId}`); if (!response.ok) return; const data = await response.json(); $('#messages').innerHTML = ''; data.messages.forEach((message) => addMessage(message.role, message.content, message.metadata?.experience_id)); }

async function loadModels() { try { const response = await fetch('/api/models'); const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Unable to load models'); $('#model-select').innerHTML = data.models.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)}</option>`).join(''); $('#model-select').value = data.default; } catch (error) { $('#model-status').textContent = `Model list error: ${error.message}`; $('#model-status').className = 'model-status error'; } }
async function loadDiagnostics() { try { const response = await fetch('/api/model/diagnostics'); const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Diagnostic request failed'); const status = $('#model-status'); status.className = `model-status ${data.status}`; status.textContent = data.status === 'error' ? `Disconnected: ${data.error}` : (data.backend === 'mock' ? 'Offline mock ready' : `Connected · ${data.model}`); } catch (error) { $('#model-status').textContent = `Connection error: ${error.message}`; $('#model-status').className = 'model-status error'; } }

$('#chat-form').addEventListener('submit', async (event) => { event.preventDefault(); const input = $('#message'); const message = input.value.trim(); if (!message) return; input.value = ''; addMessage('user', message); $('#send').disabled = true; $('#status').textContent = 'Thinking…'; try { const response = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({conversation_id:conversationId, message, model: $('#model-select').value || null})}); const result = await response.json(); if (!response.ok) throw new Error(result.detail || 'Request failed'); conversationId = result.conversation_id; addMessage('assistant', result.response, result.experience_id); $('#status').textContent = 'Saved locally.'; loadSidebar(); loadDiagnostics(); } catch (error) { addMessage('assistant', `Model request failed: ${error.message}`); $('#status').textContent = 'Connection error'; loadDiagnostics(); } finally { $('#send').disabled = false; input.focus(); } });
$('#new-chat').addEventListener('click', () => { conversationId = null; $('#messages').innerHTML = '<div class="empty">Start a conversation. Explicit facts and preferences are remembered locally.</div>'; $('#status').textContent = 'Ready to remember.'; }); $('#refresh').addEventListener('click', loadSidebar); $('#model-select').addEventListener('change', loadDiagnostics); loadModels(); loadDiagnostics(); loadSidebar();
