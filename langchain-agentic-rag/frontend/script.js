// LangChain Agentic RAG Frontend JavaScript

class RAGClient {
    constructor() {
        this.baseUrl = window.location.origin;
        this.sessionId = this.generateSessionId();
        this.isConnected = false;
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkConnection();
        this.loadStats();
        this.loadAgentTools();
    }

    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9);
    }

    bindEvents() {
        // Chat form submission
        const chatForm = document.getElementById('chatForm');
        const chatInput = document.getElementById('chatInput');
        
        chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.submitQuery();
        });

        // Auto-resize textarea
        chatInput.addEventListener('input', () => {
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
        });

        // Upload form
        const uploadForm = document.getElementById('uploadForm');
        uploadForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.uploadDocument();
        });

        // Control buttons
        document.getElementById('clearBtn').addEventListener('click', () => this.clearChat());
        document.getElementById('exportBtn').addEventListener('click', () => this.exportChat());
        document.getElementById('refreshStatsBtn').addEventListener('click', () => this.loadStats());

        // Agent steps toggle
        document.addEventListener('click', (e) => {
            if (e.target.closest('.agent-steps-header')) {
                this.toggleAgentSteps(e.target.closest('.agent-steps'));
            }
        });
    }

    async checkConnection() {
        try {
            const response = await fetch(`${this.baseUrl}/api/health`);
            const data = await response.json();
            
            this.updateConnectionStatus(data.status === 'healthy');
        } catch (error) {
            console.error('Connection check failed:', error);
            this.updateConnectionStatus(false);
        }
    }

    updateConnectionStatus(isConnected) {
        this.isConnected = isConnected;
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');

        if (isConnected) {
            statusDot.classList.add('connected');
            statusText.textContent = 'Connected';
        } else {
            statusDot.classList.remove('connected');
            statusText.textContent = 'Disconnected';
        }
    }

    async submitQuery() {
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        const useAgent = document.getElementById('useAgent').checked;
        const maxResults = parseInt(document.getElementById('maxResults').value);

        const query = chatInput.value.trim();
        if (!query) return;

        // Disable input
        chatInput.disabled = true;
        sendBtn.disabled = true;

        // Add user message to chat
        this.addMessage('user', query);

        // Clear input
        chatInput.value = '';
        chatInput.style.height = 'auto';

        try {
            this.showLoading(true);

            const response = await fetch(`${this.baseUrl}/api/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query,
                    session_id: this.sessionId,
                    max_results: maxResults,
                    use_agent: useAgent
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            // Add assistant response
            this.addMessage('assistant', data.response, {
                sources: data.sources,
                agentSteps: data.agent_steps
            });

            this.showToast('Query processed successfully', 'success');

        } catch (error) {
            console.error('Query failed:', error);
            this.addMessage('assistant', `I apologize, but I encountered an error: ${error.message}`, {
                isError: true
            });
            this.showToast('Query failed: ' + error.message, 'error');
        } finally {
            // Re-enable input
            chatInput.disabled = false;
            sendBtn.disabled = false;
            chatInput.focus();
            this.showLoading(false);
        }
    }

    addMessage(role, content, options = {}) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;

        const timestamp = new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
        });

        let roleIcon = role === 'user' ? '👤' : '🤖';
        let roleName = role === 'user' ? 'You' : 'Assistant';

        let messageHtml = `
            <div class="message-content">
                <div class="message-header">
                    <span class="message-role">${roleIcon} ${roleName}</span>
                    <span class="message-time">${timestamp}</span>
                </div>
                <div class="message-text">${this.formatMessage(content)}</div>
        `;

        // Add sources if available
        if (options.sources && options.sources.length > 0) {
            messageHtml += `
                <div class="message-sources">
                    <h4>📚 Sources</h4>
                    ${options.sources.map(source => 
                        `<div class="source-item">• ${source.title} (${source.type})</div>`
                    ).join('')}
                </div>
            `;
        }

        // Add agent steps if available
        if (options.agentSteps && options.agentSteps.length > 0) {
            messageHtml += `
                <div class="agent-steps">
                    <div class="agent-steps-header">
                        🔍 Agent Reasoning (${options.agentSteps.length} steps)
                        <span>▼</span>
                    </div>
                    <div class="agent-steps-content">
                        ${options.agentSteps.map((step, index) => `
                            <div class="agent-step">
                                <div class="step-tool">Step ${index + 1}: ${step.tool}</div>
                                ${step.thought ? `<div class="step-thought">Thought: ${step.thought}</div>` : ''}
                                <div class="step-observation">Result: ${this.truncateText(step.observation, 200)}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        messageHtml += '</div>';
        messageDiv.innerHTML = messageHtml;

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    formatMessage(content) {
        // Use marked.js for markdown rendering if available
        if (typeof marked !== 'undefined') {
            return marked.parse(content);
        }
        
        // Basic formatting fallback
        return content
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>');
    }

    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substr(0, maxLength) + '...';
    }

    toggleAgentSteps(agentStepsDiv) {
        const content = agentStepsDiv.querySelector('.agent-steps-content');
        const header = agentStepsDiv.querySelector('.agent-steps-header span');
        
        if (content.classList.contains('expanded')) {
            content.classList.remove('expanded');
            header.textContent = '▼';
        } else {
            content.classList.add('expanded');
            header.textContent = '▲';
        }
    }

    async uploadDocument() {
        const title = document.getElementById('docTitle').value.trim();
        const content = document.getElementById('docContent').value.trim();

        if (!title || !content) {
            this.showToast('Please fill in both title and content', 'warning');
            return;
        }

        try {
            this.showLoading(true);

            const response = await fetch(`${this.baseUrl}/api/upload`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    title,
                    content,
                    metadata: {
                        uploaded_at: new Date().toISOString(),
                        source: 'web_interface'
                    }
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            this.showToast(`Document "${title}" uploaded successfully (${data.chunks_created} chunks)`, 'success');
            
            // Clear form
            document.getElementById('docTitle').value = '';
            document.getElementById('docContent').value = '';
            
            // Refresh stats
            this.loadStats();

        } catch (error) {
            console.error('Upload failed:', error);
            this.showToast('Upload failed: ' + error.message, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async loadStats() {
        try {
            const response = await fetch(`${this.baseUrl}/api/stats`);
            if (!response.ok) throw new Error('Failed to load stats');
            
            const stats = await response.json();
            
            document.getElementById('totalDocs').textContent = stats.total_documents || 0;
            document.getElementById('totalChunks').textContent = stats.total_chunks || 0;
            document.getElementById('totalEmbeddings').textContent = stats.total_embeddings || 0;

        } catch (error) {
            console.error('Failed to load stats:', error);
            document.getElementById('totalDocs').textContent = '?';
            document.getElementById('totalChunks').textContent = '?';
            document.getElementById('totalEmbeddings').textContent = '?';
        }
    }

    async loadAgentTools() {
        try {
            const response = await fetch(`${this.baseUrl}/api/tools`);
            if (!response.ok) throw new Error('Failed to load tools');
            
            const data = await response.json();
            const toolsContainer = document.getElementById('agentTools');
            
            if (data.tools && data.tools.length > 0) {
                toolsContainer.innerHTML = data.tools.map(tool => `
                    <div class="tool-item">
                        <div class="tool-name">${tool.name}</div>
                        <div class="tool-description">${tool.description}</div>
                    </div>
                `).join('');
            } else {
                toolsContainer.innerHTML = '<div class="loading">No tools available</div>';
            }

        } catch (error) {
            console.error('Failed to load agent tools:', error);
            document.getElementById('agentTools').innerHTML = '<div class="loading">Failed to load tools</div>';
        }
    }

    clearChat() {
        const chatMessages = document.getElementById('chatMessages');
        
        // Keep the initial assistant message
        const messages = chatMessages.querySelectorAll('.message');
        for (let i = 1; i < messages.length; i++) {
            messages[i].remove();
        }
        
        // Reset session
        this.sessionId = this.generateSessionId();
        
        this.showToast('Chat cleared', 'success');
    }

    exportChat() {
        const messages = document.querySelectorAll('.message');
        let exportText = 'LangChain Agentic RAG Chat Export\n';
        exportText += '=====================================\n\n';

        messages.forEach(message => {
            const role = message.querySelector('.message-role').textContent;
            const time = message.querySelector('.message-time').textContent;
            const text = message.querySelector('.message-text').textContent;
            
            exportText += `${role} (${time}):\n${text}\n\n`;
        });

        // Download as text file
        const blob = new Blob([exportText], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat_export_${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        this.showToast('Chat exported successfully', 'success');
    }

    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;

        toastContainer.appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(() => {
            toast.remove();
        }, 5000);
    }

    showLoading(show) {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (show) {
            loadingOverlay.classList.remove('hidden');
        } else {
            loadingOverlay.classList.add('hidden');
        }
    }
}

// Initialize the RAG client when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.ragClient = new RAGClient();
    
    // Set up periodic connection checks
    setInterval(() => {
        window.ragClient.checkConnection();
    }, 30000); // Check every 30 seconds
    
    // Set up periodic stats refresh
    setInterval(() => {
        window.ragClient.loadStats();
    }, 60000); // Refresh every minute
});

// Handle keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to send message
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        document.getElementById('chatForm').dispatchEvent(new Event('submit'));
    }
    
    // Escape to clear current input
    if (e.key === 'Escape') {
        const chatInput = document.getElementById('chatInput');
        if (chatInput === document.activeElement) {
            chatInput.value = '';
            chatInput.style.height = 'auto';
        }
    }
});

// Handle page visibility for connection monitoring
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.ragClient) {
        // Page became visible, check connection and refresh stats
        window.ragClient.checkConnection();
        window.ragClient.loadStats();
    }
});

// Service worker registration for offline support (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then((registration) => {
                console.log('ServiceWorker registration successful');
            })
            .catch((error) => {
                console.log('ServiceWorker registration failed');
            });
    });
}