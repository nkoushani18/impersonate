/**
 * Penta-PersonaAI - Chat Application JavaScript
 */

// State management
const state = {
    personas: [],
    activePersona: null,
    debugMode: false,
    chatHistory: []
};

// DOM Elements
const elements = {
    personaList: document.getElementById('persona-list'),
    activePersonaSection: document.getElementById('active-persona-section'),
    activePersonaCard: document.getElementById('active-persona-card'),
    chatMessages: document.getElementById('chat-messages'),
    chatInput: document.getElementById('chat-input'),
    sendBtn: document.getElementById('send-btn'),
    chatSubtitle: document.getElementById('chat-subtitle'),
    debugToggle: document.getElementById('debug-toggle'),
    clearChatBtn: document.getElementById('clear-chat-btn'),
    systemInfoBtn: document.getElementById('system-info-btn'),
    modalOverlay: document.getElementById('modal-overlay'),
    modalBody: document.getElementById('modal-body'),
    modalClose: document.getElementById('modal-close')
};

// Initialize application
async function init() {
    await loadPersonas();
    setupEventListeners();
}

// Load personas from API
async function loadPersonas() {
    try {
        const response = await fetch('/api/personas');
        const data = await response.json();
        state.personas = data.personas;
        renderPersonaList();
    } catch (error) {
        console.error('Failed to load personas:', error);
        elements.personaList.innerHTML = `
            <div class="error-state">
                <p>Failed to load personas. Please refresh.</p>
            </div>
        `;
    }
}

// Render persona list in sidebar
function renderPersonaList() {
    if (state.personas.length === 0) {
        elements.personaList.innerHTML = '<p class="no-data">No personas available</p>';
        return;
    }

    elements.personaList.innerHTML = state.personas.map(persona => `
        <div class="persona-card" data-id="${persona.id}" onclick="selectPersona('${persona.id}')">
            <div class="persona-name">
                ${persona.name || 'Unknown'}
                <span class="persona-id">${persona.id}</span>
            </div>
            <div class="persona-traits">
                ${Object.entries(persona.traits).slice(0, 4).map(([key, value]) => 
                    `<span class="trait-badge">${key.replace(/_/g, ' ')}: ${value}</span>`
                ).join('')}
            </div>
        </div>
    `).join('');
}

// Select a persona
async function selectPersona(personaId) {
    try {
        const response = await fetch('/api/persona/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ persona_id: personaId })
        });

        const data = await response.json();
        
        if (data.success) {
            state.activePersona = data.persona;
            updateActivePersonaUI();
            enableChat();
            addSystemMessage(`Now chatting as **${data.persona.name}** (${personaId})`);
        } else {
            console.error('Failed to select persona:', data.error);
        }
    } catch (error) {
        console.error('Error selecting persona:', error);
    }
}

// Update UI for active persona
function updateActivePersonaUI() {
    // Update sidebar cards
    document.querySelectorAll('.persona-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.id === state.activePersona.persona_id);
    });

    // Show active persona section
    elements.activePersonaSection.style.display = 'block';
    elements.activePersonaCard.innerHTML = `
        <div class="persona-name">${state.activePersona.name}</div>
        <div class="persona-traits">
            ${Object.entries(state.activePersona.traits).map(([key, value]) => 
                `<span class="trait-badge">${key.replace(/_/g, ' ')}: ${value}</span>`
            ).join('')}
        </div>
    `;

    // Update chat subtitle
    elements.chatSubtitle.textContent = `Chatting with ${state.activePersona.name}`;
}

// Enable chat input
function enableChat() {
    elements.chatInput.disabled = false;
    elements.sendBtn.disabled = false;
    elements.chatInput.placeholder = "Ask about career, relationships, values, or life decisions...";
    elements.chatInput.focus();

    // Clear welcome message
    const welcomeMsg = elements.chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.style.display = 'none';
    }
}

// Send message
async function sendMessage() {
    const message = elements.chatInput.value.trim();
    if (!message || !state.activePersona) return;

    // Clear input
    elements.chatInput.value = '';

    // Add user message to chat
    addMessage('user', message);

    // Show typing indicator
    const typingEl = addTypingIndicator();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                debug: state.debugMode
            })
        });

        const data = await response.json();

        // Remove typing indicator
        typingEl.remove();

        // Add persona response
        addMessage('persona', data.response, data.debug);

    } catch (error) {
        console.error('Chat error:', error);
        typingEl.remove();
        addMessage('persona', 'Sorry, something went wrong. Please try again.');
    }
}

// Add message to chat
function addMessage(type, text, debugInfo = null) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${type}`;

    const avatar = type === 'user' ? '👤' : '🧠';
    
    let debugHtml = '';
    if (debugInfo && state.debugMode) {
        debugHtml = `
            <div class="message-debug">
                <div class="debug-title">🔍 Trait Reasoning</div>
                <div class="debug-item"><strong>Intent:</strong> ${debugInfo.intent}</div>
                <div class="debug-item"><strong>Confidence:</strong> ${(debugInfo.confidence * 100).toFixed(0)}%</div>
                <div class="debug-item"><strong>Traits Used:</strong></div>
                ${Object.entries(debugInfo.traits_used || {}).map(([key, value]) => 
                    `<div class="debug-item" style="margin-left: 1rem;">• ${key}: ${value}</div>`
                ).join('')}
            </div>
        `;
    }

    messageEl.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-bubble">${text}</div>
            ${debugHtml}
        </div>
    `;

    elements.chatMessages.appendChild(messageEl);
    scrollToBottom();

    // Save to history
    state.chatHistory.push({ type, text, debugInfo });
}

// Add system message
function addSystemMessage(text) {
    const messageEl = document.createElement('div');
    messageEl.className = 'message system';
    messageEl.style.cssText = 'justify-content: center; margin: 1rem 0;';
    messageEl.innerHTML = `
        <div style="
            background: var(--bg-card);
            border: 1px solid var(--border-accent);
            border-radius: 20px;
            padding: 0.5rem 1rem;
            font-size: 0.85rem;
            color: var(--accent-primary);
        ">${text}</div>
    `;
    elements.chatMessages.appendChild(messageEl);
    scrollToBottom();
}

// Add typing indicator
function addTypingIndicator() {
    const typingEl = document.createElement('div');
    typingEl.className = 'message persona';
    typingEl.innerHTML = `
        <div class="message-avatar">🧠</div>
        <div class="message-content">
            <div class="message-bubble">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
    `;
    elements.chatMessages.appendChild(typingEl);
    scrollToBottom();
    return typingEl;
}

// Scroll chat to bottom
function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

// Clear chat
function clearChat() {
    state.chatHistory = [];
    elements.chatMessages.innerHTML = `
        <div class="welcome-message" style="${state.activePersona ? 'display: none;' : ''}">
            <div class="welcome-icon">👋</div>
            <h2>Welcome to Penta-PersonaAI</h2>
            <p>This is a responsible, agentic persona AI that simulates how specific personas think and respond about career and relationships.</p>
            <div class="welcome-features">
                <div class="feature">
                    <span class="feature-icon">🎭</span>
                    <span>5 Unique Personas</span>
                </div>
                <div class="feature">
                    <span class="feature-icon">🧬</span>
                    <span>Trait-Based Reasoning</span>
                </div>
                <div class="feature">
                    <span class="feature-icon">🔒</span>
                    <span>No Training Required</span>
                </div>
            </div>
            <p class="welcome-hint">← Select a persona from the sidebar to begin</p>
        </div>
    `;

    if (state.activePersona) {
        addSystemMessage(`Now chatting as **${state.activePersona.name}**`);
    }
}

// Show system info modal
async function showSystemInfo() {
    try {
        const response = await fetch('/api/system/identity');
        const data = await response.json();

        elements.modalBody.innerHTML = `
            <p><strong>Identity:</strong></p>
            <p>${data.identity}</p>
            <p><strong>Core Principle:</strong></p>
            <p>${data.principle}</p>
            <p><strong>Key Features:</strong></p>
            <ul style="color: var(--text-secondary); margin-left: 1.5rem;">
                <li>No training or fine-tuning of model weights</li>
                <li>Personalization at inference time only</li>
                <li>Each persona is isolated - no data mixing</li>
                <li>Responses traceable to survey traits</li>
            </ul>
        `;

        elements.modalOverlay.classList.add('active');
    } catch (error) {
        console.error('Error fetching system info:', error);
    }
}

// Close modal
function closeModal() {
    elements.modalOverlay.classList.remove('active');
}

// Setup event listeners
function setupEventListeners() {
    // Send button
    elements.sendBtn.addEventListener('click', sendMessage);

    // Enter key in input
    elements.chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Debug toggle
    elements.debugToggle.addEventListener('change', (e) => {
        state.debugMode = e.target.checked;
    });

    // Clear chat
    elements.clearChatBtn.addEventListener('click', clearChat);

    // System info
    elements.systemInfoBtn.addEventListener('click', showSystemInfo);

    // Modal close
    elements.modalClose.addEventListener('click', closeModal);
    elements.modalOverlay.addEventListener('click', (e) => {
        if (e.target === elements.modalOverlay) {
            closeModal();
        }
    });

    // Escape key closes modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeModal();
        }
    });
}

// Make selectPersona available globally
window.selectPersona = selectPersona;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);
