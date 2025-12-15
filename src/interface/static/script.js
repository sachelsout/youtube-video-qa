// State management
let appState = {
    currentVideoId: null,
    currentMode: 'baseline',
    conversations: {
        baseline: [],  // Separate conversation history for baseline mode
        llm: []        // Separate conversation history for LLM mode
    }
};

const elements = {
    setupView: document.getElementById('setup-view'),
    chatView: document.getElementById('chat-view'),
    videoUrl: document.getElementById('videoUrl'),
    transcribeBtn: document.getElementById('transcribeBtn'),
    setupMessage: document.getElementById('setupMessage'),
    videoTitle: document.getElementById('videoTitle'),
    videoThumbnail: document.getElementById('videoThumbnail'),
    chatMessages: document.getElementById('chatMessages'),
    chunksSlider: document.getElementById('chunksSlider'),
    chunksValue: document.getElementById('chunksValue'),
    questionInput: document.getElementById('questionInput'),
    askBtn: document.getElementById('askBtn'),
    charCount: document.getElementById('charCount'),
    clearBtn: document.getElementById('clearBtn'),
    changeVideoBtn: document.getElementById('changeVideoBtn'),
    modeBaseline: document.getElementById('modeBaseline'),
    modeLLM: document.getElementById('modeLLM')
};

// View management
function switchToChat() {
    elements.setupView.classList.remove('active');
    elements.chatView.classList.add('active');
    renderConversation();  // Render conversation for current mode
}

function switchToSetup() {
    elements.chatView.classList.remove('active');
    elements.setupView.classList.add('active');
    appState.conversations = { baseline: [], llm: [] };
    elements.chatMessages.innerHTML = '';
}

// Render conversation for current mode
function renderConversation() {
    elements.chatMessages.innerHTML = '';
    const currentConversation = appState.conversations[appState.currentMode] || [];
    
    currentConversation.forEach(msg => {
        const type = msg.role === 'user' ? 'user' : 'assistant';
        addMessage(msg.content, type, null, false);  // false = don't scroll yet
    });
    
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

// Message display
function addMessage(text, type = 'assistant', metadata = null, scroll = true) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${type}`;
    
    if (type === 'assistant' && metadata?.relevantChunks?.length > 0) {
        // For baseline mode with chunks, format as bullet points
        let html = `<div style="white-space: pre-wrap; word-wrap: break-word;">${escapeHtml(text)}</div>`;
        messageEl.innerHTML = html;
    } else {
        messageEl.textContent = text;
    }
    
    elements.chatMessages.appendChild(messageEl);
    
    if (scroll) {
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// API calls
async function transcribeVideo() {
    const url = elements.videoUrl.value.trim();
    if (!url) {
        showMessage(elements.setupMessage, 'Please enter a YouTube URL', 'error');
        return;
    }
    
    elements.transcribeBtn.disabled = true;
    elements.transcribeBtn.textContent = 'Loading...';
    showMessage(elements.setupMessage, 'Processing video...', 'info');
    
    try {
        console.log('[DEBUG] Starting transcribe request with URL:', url);
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minute timeout
        
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_url: url }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        console.log('[DEBUG] Response status:', response.status);
        const data = await response.json();
        console.log('[DEBUG] Response data:', data);
        
        if (!response.ok) {
            const errorMsg = data.detail || 'Failed to transcribe video';
            console.error('[ERROR]', errorMsg);
            showMessage(elements.setupMessage, errorMsg, 'error');
            return;
        }
        
        appState.currentVideoId = data.video_id;
        elements.videoTitle.textContent = data.title || data.video_id;
        
        // Set thumbnail image URL (YouTube thumbnail format)
        const thumbnailUrl = `https://img.youtube.com/vi/${data.video_id}/maxresdefault.jpg`;
        elements.videoThumbnail.src = thumbnailUrl;
        elements.videoThumbnail.alt = data.title || data.video_id;
        
        // Make thumbnail clickable to open video
        elements.videoThumbnail.style.cursor = 'pointer';
        elements.videoThumbnail.onclick = () => {
            window.open(`https://www.youtube.com/watch?v=${data.video_id}`, '_blank');
        };
        
        console.log('[DEBUG] Video loaded successfully, switching to chat view');
        switchToChat();
        
        // Show message after switching to chat view
        addMessage(`✅ Video loaded! Found ${data.chunk_count} transcript chunks.`, 'info');
        
    } catch (error) {
        console.error('[ERROR] Transcribe error:', error);
        if (error.name === 'AbortError') {
            showMessage(elements.setupMessage, 'Request timed out (video processing took too long)', 'error');
        } else {
            showMessage(elements.setupMessage, `Error: ${error.message}`, 'error');
        }
    } finally {
        elements.transcribeBtn.disabled = false;
        elements.transcribeBtn.textContent = 'Load Video';
    }
}

async function askQuestion() {
    const question = elements.questionInput.value.trim();
    
    if (!question) return;
    if (!appState.currentVideoId) {
        addMessage('Error: No video loaded', 'error');
        return;
    }
    
    // Add user message to current mode's conversation
    addMessage(question, 'user');
    appState.conversations[appState.currentMode].push({ role: 'user', content: question });
    
    // Clear input
    elements.questionInput.value = '';
    updateCharCount();
    
    // Disable input
    elements.askBtn.disabled = true;
    elements.questionInput.disabled = true;
    
    // Show loading
    addMessage('Thinking...', 'info');
    
    try {
        console.log('[DEBUG] Sending question to', appState.currentMode, 'mode:', question);
        
        const chunksToRetrieve = parseInt(elements.chunksSlider.value) || 5;
        console.log('[DEBUG] Chunks to retrieve:', chunksToRetrieve);
        
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_id: appState.currentVideoId,
                question: question,
                mode: appState.currentMode,
                chunks_k: chunksToRetrieve,
                conversation_history: appState.conversations[appState.currentMode]
            })
        });
        
        const data = await response.json();
        console.log('[DEBUG] Answer response:', data);
        
        // Remove loading message
        const loadingMsg = elements.chatMessages.lastChild;
        if (loadingMsg && loadingMsg.textContent === 'Thinking...') {
            loadingMsg.remove();
        }
        
        if (!response.ok) {
            const errorMsg = data.detail || 'Failed to get answer';
            console.error('[ERROR]', errorMsg);
            addMessage(`Error: ${errorMsg}`, 'error');
            return;
        }
        
        addMessage(data.answer, 'assistant', { relevantChunks: data.relevant_chunks || [] });
        appState.conversations[appState.currentMode].push({ role: 'assistant', content: data.answer });
        
    } catch (error) {
        console.error('[ERROR] Ask question error:', error);
        addMessage(`Error: ${error.message}`, 'error');
    } finally {
        elements.askBtn.disabled = false;
        elements.questionInput.disabled = false;
        elements.questionInput.focus();
    }
}

// UI Handlers
elements.transcribeBtn.addEventListener('click', transcribeVideo);
elements.videoUrl.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') transcribeVideo();
});

elements.askBtn.addEventListener('click', askQuestion);
elements.questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
    }
});

elements.questionInput.addEventListener('input', updateCharCount);

elements.chunksSlider.addEventListener('input', (e) => {
    elements.chunksValue.textContent = e.target.value;
});

elements.clearBtn.addEventListener('click', () => {
    if (confirm(`Clear ${appState.currentMode} conversation history?`)) {
        appState.conversations[appState.currentMode] = [];
        renderConversation();
        elements.questionInput.value = '';
        updateCharCount();
    }
});

elements.changeVideoBtn.addEventListener('click', switchToSetup);

// Mode switching
elements.modeBaseline.addEventListener('click', () => {
    appState.currentMode = 'baseline';
    updateModeButtons();
    renderConversation();  // Switch to baseline conversation
});

elements.modeLLM.addEventListener('click', () => {
    appState.currentMode = 'llm';
    updateModeButtons();
    renderConversation();  // Switch to LLM conversation
});

function updateModeButtons() {
    elements.modeBaseline.classList.toggle('active', appState.currentMode === 'baseline');
    elements.modeLLM.classList.toggle('active', appState.currentMode === 'llm');
}

function updateCharCount() {
    const count = elements.questionInput.value.length;
    elements.charCount.textContent = `${count} / 500`;
}

function showMessage(element, text, type) {
    element.textContent = text;
    element.className = `message ${type}`;
    element.classList.remove('hidden');
    setTimeout(() => element.classList.add('hidden'), 5000);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('[INIT] DOM loaded, initializing app...');
    
    // Verify all elements exist
    const missingElements = [];
    for (const [key, el] of Object.entries(elements)) {
        if (!el) {
            console.error(`[INIT] Missing element: ${key}`);
            missingElements.push(key);
        }
    }
    
    if (missingElements.length > 0) {
        console.error('[INIT] Missing elements:', missingElements);
    } else {
        console.log('[INIT] All elements found!');
    }
    
    updateCharCount();
    if (elements.questionInput) {
        elements.questionInput.focus();
    }
    
    console.log('[INIT] Initialization complete');
    
    // Dark mode initialization
    initializeDarkMode();
});

// Dark Mode Management
function initializeDarkMode() {
    const darkModeToggle = document.getElementById('darkModeToggle');
    const isDarkMode = localStorage.getItem('darkMode') === 'true';
    
    if (isDarkMode) {
        document.body.classList.add('dark-mode');
        darkModeToggle.textContent = '☀️';
    }
    
    darkModeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        const isNowDark = document.body.classList.contains('dark-mode');
        localStorage.setItem('darkMode', isNowDark);
        darkModeToggle.textContent = isNowDark ? '☀️' : '🌙';
    });
}