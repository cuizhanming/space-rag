---
title: Technology Stack
description: "Defines the technology stack, frameworks, dependencies, and development practices."
inclusion: always
---

# Technology Stack

## Core Technologies

### Backend
- **Python 3.13+**: Primary language
- **FastAPI**: Web framework for REST API endpoints
- **Uvicorn**: ASGI server for FastAPI
- **Anthropic API**: Claude Sonnet 4 for AI text generation
- **ChromaDB 1.0.15**: Vector database for semantic search
- **Sentence Transformers 5.0.0**: Embedding model (all-MiniLM-L6-v2)

### Frontend
- **HTML5/CSS3/JavaScript**: Vanilla web technologies
- **Marked.js**: Markdown parsing for response formatting
- **Responsive Design**: Mobile-friendly interface

### Dependencies
- **python-dotenv**: Environment variable management
- **python-multipart**: File upload support
- **pydantic**: Data validation and serialization

## Architecture

### System Components
1. **FastAPI Application** (`backend/app.py`): Main web server with API endpoints
2. **RAG System** (`backend/rag_system.py`): Core orchestrator for retrieval and generation
3. **Vector Store** (`backend/vector_store.py`): ChromaDB interface for semantic search
4. **AI Generator** (`backend/ai_generator.py`): Anthropic Claude API integration
5. **Document Processor** (`backend/document_processor.py`): Text chunking and course parsing
6. **Search Tools** (`backend/search_tools.py`): Tool-based search architecture
7. **Session Manager** (`backend/session_manager.py`): Conversation context management

### Data Models (`backend/models.py`)
- **Course**: Represents a complete course with lessons
- **Lesson**: Individual lessons within courses
- **CourseChunk**: Text chunks for vector storage

## Development Workflow

### Package Management
- **uv**: Modern Python package manager for dependency management
- **pyproject.toml**: Project configuration and dependencies

### Running the Application
```bash
# Install dependencies
uv sync

# Start development server
./run.sh
# OR manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Environment Configuration
- **`.env` file**: Required for Anthropic API key
- **Environment Variables**: `ANTHROPIC_API_KEY`

### API Endpoints
- `POST /api/query`: Process user queries and return AI responses
- `GET /api/courses`: Get course statistics and metadata
- `GET /docs`: FastAPI automatic API documentation

## Configuration Settings (`backend/config.py`)

### AI Settings
- **Model**: Claude Sonnet 4 (claude-sonnet-4-20250514)
- **Temperature**: 0 (deterministic responses)
- **Max Tokens**: 800

### Document Processing
- **Chunk Size**: 800 characters
- **Chunk Overlap**: 100 characters
- **Max Results**: 5 search results
- **Max History**: 2 conversation exchanges

### Storage
- **ChromaDB Path**: `./chroma_db`
- **Embedding Model**: all-MiniLM-L6-v2

## Development Practices

### Code Organization
- Clear separation between backend logic and frontend
- Modular component architecture
- Type hints with Pydantic models
- Error handling and logging

### CORS Configuration
- Allows cross-origin requests for development
- Trusted host middleware for proxy support
- Static file serving for frontend

### Security Considerations
- API key management through environment variables
- Input validation through Pydantic models
- Error message sanitization