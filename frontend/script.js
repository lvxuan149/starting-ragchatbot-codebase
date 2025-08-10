// API base URL - use relative path to work from any host
const API_URL = '/api';

// Global state
let currentSessionId = null;
let abortController = null;

// Validation constants
const MAX_INPUT_LENGTH = 1000;
const MIN_INPUT_LENGTH = 1;

// DOM elements
let chatMessages, chatInput, sendButton, totalCourses, courseTitles;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements after page loads
    chatMessages = document.getElementById('chatMessages');
    chatInput = document.getElementById('chatInput');
    sendButton = document.getElementById('sendButton');
    totalCourses = document.getElementById('totalCourses');
    courseTitles = document.getElementById('courseTitles');
    
    setupEventListeners();
    createNewSession();
    loadCourseStats();
});

// Event Listeners
function setupEventListeners() {
    // Chat functionality
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    
    // Setup input validation
    setupInputValidation();
    
    // Suggested questions
    document.querySelectorAll('.suggested-item').forEach(button => {
        button.addEventListener('click', (e) => {
            const question = e.target.getAttribute('data-question');
            chatInput.value = question;
            sendMessage();
        });
    });
}


// Chat Functions
async function sendMessage() {
    const query = chatInput.value.trim();
    
    // Validation
    if (!query) {
        showError('Please enter a message');
        return;
    }
    
    if (query.length < MIN_INPUT_LENGTH) {
        showError('Message is too short');
        return;
    }
    
    if (query.length > MAX_INPUT_LENGTH) {
        showError(`Message is too long (max ${MAX_INPUT_LENGTH} characters)`);
        return;
    }

    // Cancel previous request if exists
    if (abortController) {
        abortController.abort();
    }

    // Create new abort controller for this request
    abortController = new AbortController();
    const timeoutId = setTimeout(() => {
        abortController.abort();
    }, 10000); // 10 second timeout

    // Disable input
    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;

    // Add user message
    addMessage(query, 'user');

    // Add loading message - create a unique container for it
    const loadingMessage = createLoadingMessage();
    chatMessages.appendChild(loadingMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: currentSessionId
            }),
            signal: abortController.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            if (response.status === 0) {
                throw new Error('Request timed out. Please try again.');
            }
            throw new Error('Query failed');
        }

        const data = await response.json();
        
        // Update session ID if new
        if (!currentSessionId) {
            currentSessionId = data.session_id;
        }

        // Replace loading message with response
        loadingMessage.remove();
        addMessage(data.answer, 'assistant', data.sources);

    } catch (error) {
        clearTimeout(timeoutId);
        
        // Replace loading message with error
        loadingMessage.remove();
        
        if (error.name === 'AbortError') {
            addMessage('Request was cancelled. Please try again.', 'assistant');
        } else {
            addMessage(`Error: ${error.message}`, 'assistant');
        }
    } finally {
        abortController = null;
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }
}

function createLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="loading">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    return messageDiv;
}

function addMessage(content, type, sources = null, isWelcome = false) {
    const messageId = Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}${isWelcome ? ' welcome-message' : ''}`;
    messageDiv.id = `message-${messageId}`;
    
    // Convert markdown to HTML for assistant messages with sanitization
    let displayContent;
    if (type === 'assistant') {
        // Parse markdown then sanitize
        const rawHtml = marked.parse(content);
        displayContent = DOMPurify.sanitize(rawHtml, {
            ALLOWED_TAGS: ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'br', 'strong', 'em', 
                         'u', 'ol', 'ul', 'li', 'code', 'pre', 'blockquote', 'hr'],
            ALLOWED_ATTR: ['class']
        });
    } else {
        displayContent = escapeHtml(content);
    }
    
    let html = `<div class="message-content">${displayContent}</div>`;
    
    if (sources && sources.length > 0) {
        html += `
            <details class="sources-collapsible">
                <summary class="sources-header">Sources</summary>
                <div class="sources-content">${sources.join(', ')}</div>
            </details>
        `;
    }
    
    messageDiv.innerHTML = html;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

// Helper function to escape HTML for user messages
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Removed removeMessage function - no longer needed since we handle loading differently

async function createNewSession() {
    currentSessionId = null;
    chatMessages.innerHTML = '';
    addMessage('Welcome to the Course Materials Assistant! I can help you with questions about courses, lessons and specific content. What would you like to know?', 'assistant', null, true);
}

// Helper functions
function showError(message) {
    // Remove existing error messages
    const existingErrors = document.querySelectorAll('.error-message');
    existingErrors.forEach(err => err.remove());
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    
    // Insert after input container
    const inputContainer = document.querySelector('.chat-input-container');
    inputContainer.parentNode.insertBefore(errorDiv, inputContainer.nextSibling);
    
    // Remove after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

function setupInputValidation() {
    const charCounter = document.createElement('div');
    charCounter.className = 'char-counter';
    charCounter.textContent = `0 / ${MAX_INPUT_LENGTH}`;
    charCounter.style.cssText = `
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-align: right;
        margin-top: 0.25rem;
    `;
    
    chatInput.parentNode.appendChild(charCounter);
    
    chatInput.addEventListener('input', (e) => {
        const length = e.target.value.length;
        charCounter.textContent = `${length} / ${MAX_INPUT_LENGTH}`;
        
        if (length > MAX_INPUT_LENGTH * 0.9) {
            charCounter.style.color = '#f87171';
        } else {
            charCounter.style.color = 'var(--text-secondary)';
        }
    });
}

// Load course statistics
async function loadCourseStats() {
    try {
        const response = await fetch(`${API_URL}/courses`);
        if (!response.ok) throw new Error('Failed to load course stats');
        
        const data = await response.json();
        
        // Update stats in UI
        if (totalCourses) {
            totalCourses.textContent = data.total_courses;
        }
        
        // Update course titles
        if (courseTitles) {
            if (data.course_titles && data.course_titles.length > 0) {
                courseTitles.innerHTML = data.course_titles
                    .map(title => `<div class="course-title-item">${title}</div>`)
                    .join('');
            } else {
                courseTitles.innerHTML = '<span class="no-courses">No courses available</span>';
            }
        }
        
    } catch (error) {
        // Set default values on error
        if (totalCourses) {
            totalCourses.textContent = '0';
        }
        if (courseTitles) {
            courseTitles.innerHTML = '<span class="error">Failed to load courses</span>';
        }
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    // Cancel any pending requests
    if (abortController) {
        abortController.abort();
    }
});