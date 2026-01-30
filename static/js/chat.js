function toggleChat() {
    const chatWindow = document.getElementById('chat-window');
    chatWindow.classList.toggle('hidden');
    if (!chatWindow.classList.contains('hidden')) {
        document.getElementById('user-input').focus();
    }
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

function appendMessage(text, sender) {
    const messagesDiv = document.getElementById('chat-messages');
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', sender);
    msgDiv.textContent = text;
    messagesDiv.appendChild(msgDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function sendMessage() {
    const input = document.getElementById('user-input');
    const text = input.value.trim();

    if (!text) return;

    // 1. Show User Message
    appendMessage(text, 'user');
    input.value = '';

    // 2. Show Loading
    const messagesDiv = document.getElementById('chat-messages');
    const loadingDiv = document.createElement('div');
    loadingDiv.classList.add('message', 'loading');
    loadingDiv.textContent = 'Dr. Ycry is thinking...';
    loadingDiv.id = 'loading-msg';
    messagesDiv.appendChild(loadingDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    try {
        // 3. Call API
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: text })
        });

        const data = await response.json();

        // Remove loading
        document.getElementById('loading-msg').remove();

        if (data.error) {
            appendMessage("⚠️ " + data.error, 'bot');
        } else {
            appendMessage(data.response, 'bot');
        }

    } catch (error) {
        document.getElementById('loading-msg').remove();
        appendMessage("❌ Network error. Please try again.", 'bot');
        console.error('Chat Error:', error);
    }
}
