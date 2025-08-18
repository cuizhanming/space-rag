# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system for querying course materials. It combines ChromaDB vector storage, Anthropic's Claude AI, and a web interface to provide intelligent, context-aware responses about educational content.

## Development Commands

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
Create `.env` file with:
```
ANTHROPIC_API_KEY=your_key_here
```

## Architecture

**Backend Structure (`backend/`):**
- `app.py` - FastAPI application with CORS and static file serving
- `rag_system.py` - Main orchestrator coordinating all components
- `vector_store.py` - ChromaDB interface for embeddings and search
- `ai_generator.py` - Anthropic Claude API wrapper
- `document_processor.py` - Course document parsing and chunking
- `session_manager.py` - Conversation history management
- `search_tools.py` - Tool-based search system for AI agent
- `models.py` - Pydantic data models
- `config.py` - Configuration with environment variables

**Frontend (`frontend/`):**
- Vanilla HTML/CSS/JavaScript with Marked.js for markdown rendering
- Communicates with backend via `/api/query` and `/api/courses` endpoints

**Data Flow:**
1. Documents in `docs/` are processed into chunks during startup
2. Chunks stored in ChromaDB with embeddings (all-MiniLM-L6-v2)
3. User queries trigger semantic search + AI generation via tool-based system
4. Claude uses search tools to find relevant content and generate responses

## Key Configuration

- Chunk size: 800 characters with 100 character overlap
- Max search results: 5 per query
- Conversation history: 2 previous messages
- Model: claude-sonnet-4-20250514
- Storage: ChromaDB in `./chroma_db`

## Course Document Format

Course files are expected to be text files with structured content including titles, descriptions, and lessons. The document processor automatically extracts metadata and creates searchable chunks.

## API Endpoints

- `POST /api/query` - Process user queries with optional session_id
- `GET /api/courses` - Get course statistics and analytics
- `/` - Serves static frontend files

## Dependencies

Built with Python 3.13+, using FastAPI, ChromaDB, Anthropic SDK, Sentence Transformers, and UV for package management.