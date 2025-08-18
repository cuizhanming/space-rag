# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Course Materials RAG (Retrieval-Augmented Generation) system located in the `starting-ragchatbot-codebase/` subdirectory. The system combines ChromaDB vector storage, Anthropic's Claude AI, and a web interface to provide intelligent, context-aware responses about educational content.

## Development Commands

**Navigate to project directory:**
```bash
cd starting-ragchatbot-codebase
```

**Install dependencies:**
```bash
uv sync
```

**Run the application:**
```bash
./run.sh
# OR manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Environment setup:**
Create `.env` file in `starting-ragchatbot-codebase/` with:
```
ANTHROPIC_API_KEY=your_key_here
```

## Architecture

The system uses a tool-based architecture where Claude AI uses search tools to find relevant content before generating responses.

**Backend Structure (`starting-ragchatbot-codebase/backend/`):**
- `app.py` - FastAPI application with CORS, serves static files from `../frontend/`
- `rag_system.py` - Main orchestrator coordinating all components
- `vector_store.py` - ChromaDB interface for embeddings and semantic search
- `ai_generator.py` - Anthropic Claude API wrapper with tool support
- `document_processor.py` - Course document parsing and chunking (800 chars, 100 overlap)
- `session_manager.py` - Conversation history management (2 previous messages)
- `search_tools.py` - Tool-based search system for AI agent
- `models.py` - Pydantic data models for Course, Lesson, CourseChunk
- `config.py` - Configuration management with environment variables

**Frontend (`starting-ragchatbot-codebase/frontend/`):**
- Vanilla HTML/CSS/JavaScript with Marked.js for markdown rendering
- Communicates via `/api/query` and `/api/courses` endpoints

**Data Flow:**
1. Documents in `docs/` are processed into chunks during startup (`app.py:94-104`)
2. Chunks stored in ChromaDB with embeddings (all-MiniLM-L6-v2)
3. User queries trigger tool-based search via `CourseSearchTool` in `search_tools.py`
4. Claude uses search tools to find relevant content and generate responses
5. Sources are tracked through the `ToolManager` for transparency

## Key Configuration

- Model: claude-sonnet-4-20250514
- Embedding model: all-MiniLM-L6-v2
- Chunk size: 800 characters with 100 character overlap
- Max search results: 5 per query
- Storage: ChromaDB in `./chroma_db`
- Max conversation history: 2 previous messages

## API Endpoints

- `POST /api/query` - Process user queries with optional session_id
- `GET /api/courses` - Get course statistics and analytics
- `/` - Serves static frontend files from `../frontend/`

## Key Implementation Notes

- The system automatically loads documents from `../docs` on startup if the directory exists
- Uses `DevStaticFiles` class to serve frontend with no-cache headers for development
- Session management creates unique session IDs for conversation tracking
- Tool-based architecture allows Claude to search before responding rather than using pre-retrieved context
- ChromaDB stores both course metadata and content chunks for comprehensive search

## Dependencies

Built with Python 3.13+, managed by UV package manager. Core dependencies include FastAPI, ChromaDB, Anthropic SDK, and Sentence Transformers.