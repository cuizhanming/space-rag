# 🤖 LangChain Agentic RAG System

A powerful Retrieval-Augmented Generation (RAG) system built with LangChain, Gemini API, and agentic capabilities. This system provides intelligent knowledge retrieval and question answering with advanced reasoning capabilities.

## ✨ Features

- **🧠 Agentic RAG**: Uses LangChain ReAct agents for intelligent reasoning and tool usage
- **🚀 Gemini API Integration**: Powered by Google's advanced Gemini language model
- **🔍 Vector Search**: ChromaDB for efficient semantic search and retrieval
- **💬 Session Management**: Conversation history and context management
- **🌐 Web Interface**: Modern, responsive frontend with real-time chat
- **🔌 OpenWebUI Compatible**: Works with OpenWebUI for advanced chat management
- **📚 Knowledge Base**: Easy document upload and management
- **🛠️ RESTful API**: Comprehensive API for integration with other systems

## 🏗️ Architecture

```
langchain-agentic-rag/
├── backend/                 # FastAPI backend
│   ├── app.py              # Main application
│   ├── agent.py            # LangChain ReAct agent
│   ├── agent_tools.py      # Custom agent tools
│   ├── config.py           # Configuration management
│   ├── knowledge_base.py   # Document management
│   ├── llm_client.py       # Gemini API client
│   ├── models.py           # Pydantic data models
│   ├── session_manager.py  # Session management
│   └── vector_store.py     # ChromaDB integration
├── frontend/               # Web interface
│   ├── index.html          # Main page
│   ├── script.js           # Frontend JavaScript
│   └── style.css           # Styling
├── docs/                   # Sample documents
├── tests/                  # Test files
├── pyproject.toml          # Dependencies
├── .env.example            # Environment template
├── run.sh                  # Startup script
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [UV package manager](https://github.com/astral-sh/uv) (recommended)
- Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd langchain-agentic-rag
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env file with your Gemini API key
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

4. **Start the system:**
   ```bash
   ./run.sh
   ```

The system will be available at:
- **Web Interface**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/api/health

## 🔧 Configuration

Edit the `.env` file to customize the system:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional configurations
PORT=8001
GEMINI_MODEL_NAME=gemini-1.5-pro
MODEL_TEMPERATURE=0.7
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_SEARCH_RESULTS=5
MAX_AGENT_ITERATIONS=10
```

## 📚 Usage

### Web Interface

1. Open http://localhost:8001 in your browser
2. Upload documents using the sidebar
3. Start asking questions in the chat interface
4. Enable "Agentic Reasoning" for advanced responses

### API Usage

#### Query the System

```python
import requests

response = requests.post("http://localhost:8001/api/query", json={
    "query": "What is machine learning?",
    "use_agent": True,
    "max_results": 5
})

print(response.json()["response"])
```

#### Upload Documents

```python
response = requests.post("http://localhost:8001/api/upload", json={
    "title": "My Document",
    "content": "Document content here...",
    "metadata": {"author": "User"}
})

print(response.json())
```

#### Get Knowledge Base Stats

```python
response = requests.get("http://localhost:8001/api/stats")
print(response.json())
```

### OpenWebUI Integration

The system is compatible with OpenWebUI. Configure OpenWebUI to use:
- **Base URL**: `http://localhost:8001/api`
- **Model**: `langchain-agentic-rag`

## 🤖 Agent Capabilities

The system uses a LangChain ReAct agent with the following tools:

- **Knowledge Search**: Semantic search through the knowledge base
- **Document Stats**: Get statistics about stored documents
- **List Documents**: Browse available documents

The agent can:
- Reason about queries and plan search strategies
- Use multiple tools to gather comprehensive information
- Provide source references and reasoning steps
- Handle complex multi-step questions

## 🧪 Development

### Running Tests

```bash
uv run pytest tests/
```

### Code Quality

```bash
# Format code
uv run black backend/

# Check types
uv run mypy backend/

# Lint code
uv run flake8 backend/
```

### Adding New Agent Tools

1. Create a new tool in `backend/agent_tools.py`
2. Register it in the `get_agent_tools()` function
3. Update the agent prompt if needed

Example:

```python
class CustomTool(BaseTool):
    name = "custom_tool"
    description = "Description of what the tool does"
    
    def _run(self, query: str) -> str:
        # Tool implementation
        return "Tool result"
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/query` | POST | Submit queries |
| `/api/upload` | POST | Upload documents |
| `/api/stats` | GET | Knowledge base statistics |
| `/api/health` | GET | System health check |
| `/api/tools` | GET | Available agent tools |
| `/api/sessions/{id}` | GET | Session information |
| `/api/models` | GET | OpenWebUI compatibility |
| `/api/chat/completions` | POST | OpenWebUI chat completions |

## 🔍 Troubleshooting

### Common Issues

1. **Missing API Key**: Ensure `GEMINI_API_KEY` is set in `.env`
2. **Port Conflicts**: Change `PORT` in `.env` if 8001 is occupied
3. **Memory Issues**: Reduce `CHUNK_SIZE` and `MAX_SEARCH_RESULTS`
4. **Import Errors**: Run `uv sync` to install all dependencies

### Logs and Debugging

- Enable debug mode: Set `DEBUG=true` in `.env`
- Check logs in the console output
- Use `/api/health` to verify system components

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Ensure code quality checks pass
6. Submit a pull request

## 📄 License

This project is open source. Please check the repository for license details.

## 🙏 Acknowledgments

- **LangChain** for the agent framework
- **Google** for the Gemini API
- **ChromaDB** for vector storage
- **FastAPI** for the web framework
- **Sentence Transformers** for embeddings

---

**Happy RAGging! 🚀**

For questions or support, please open an issue in the repository.