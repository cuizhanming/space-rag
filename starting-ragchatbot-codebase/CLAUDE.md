# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Course Materials RAG (Retrieval-Augmented Generation) system that implements a tool-based architecture where Claude AI autonomously decides when to search course content. Unlike traditional RAG systems that always retrieve context, this system uses search tools only when needed based on query analysis.

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

Create `.env` file in project root with:

```bash
ANTHROPIC_API_KEY=your_key_here
```

**Testing and code quality:**

```bash
# Run tests
cd backend && uv run python tests/run_tests.py

# Format code with Black
./scripts/format.sh

# Check code format
./scripts/check-format.sh

# Run comprehensive quality checks (format + tests)
./scripts/quality-check.sh
```

## Architecture

The system uses a **tool-based RAG architecture** where Claude autonomously decides when to search, rather than always retrieving context. This is implemented through a sophisticated workflow:

**Tool-Based Search Flow:**

1. User query → Claude receives query + tool definitions
2. Claude decides autonomously whether to search based on query type
3. If needed, calls `CourseSearchTool` with optional course/lesson filtering  
4. Search results formatted with course/lesson context headers
5. Claude synthesizes search results into coherent responses
6. Sources tracked automatically for UI display

**Dual Vector Storage Pattern:**

- `course_catalog` collection: Course metadata (titles, instructors, links) for semantic course name matching
- `course_content` collection: Chunked course content with lesson-level metadata

**Backend Structure (`backend/`):**

- `app.py` - FastAPI with custom `DevStaticFiles` (no-cache headers for development)
- `rag_system.py` - Main orchestrator coordinating all components
- `vector_store.py` - ChromaDB interface with dual collection management
- `ai_generator.py` - Claude API wrapper with tool execution workflow (`_handle_tool_execution`)
- `document_processor.py` - Sentence-based chunking with context prefixing
- `session_manager.py` - Conversation history (2 previous exchanges max)
- `search_tools.py` - `CourseSearchTool` + `ToolManager` for tool registration/execution
- `models.py` - Pydantic data models for Course, Lesson, CourseChunk
- `config.py` - Configuration with claude-sonnet-4-20250514 hardcoded

**Frontend (`frontend/`):**

- Vanilla JavaScript with Marked.js for markdown rendering
- Session state management with loading animations
- Source display in collapsible sections with suggested questions

## Key Configuration

- **Model**: claude-sonnet-4-20250514 (hardcoded in config)
- **Embedding model**: all-MiniLM-L6-v2
- **Chunk size**: 800 characters with 100 character overlap
- **Max search results**: 5 per query
- **Conversation history**: 2 previous message exchanges
- **Storage**: ChromaDB in `./chroma_db` (auto-initializes)
- **Temperature**: 0 (deterministic responses)
- **Max tokens**: 800

## Course Document Format

**Required structured headers:**

```text
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [lesson_url]
[lesson content...]
```

**Processing behavior:**

- Metadata extracted from structured headers
- Content chunked by sentence boundaries with overlap
- Chunks prefixed with lesson context: "Course [title] Lesson [N] content: [chunk]"
- Documents auto-loaded from `../docs/` on startup with duplicate detection

## AI Generation Strategy

**System Prompt Rules** (`ai_generator.py:8-30`):

- "One search per query maximum" - prevents excessive tool usage
- Distinguishes general knowledge vs course-specific questions
- No meta-commentary: "Do not mention 'based on the search results'"
- Response style: Brief, educational, example-supported

**Tool Decision Logic**: Claude autonomously decides when to use search tools based on query analysis - general knowledge questions answered directly, course-specific questions trigger search.

## Session Management

**Session Lifecycle:**

- Auto-created on first query if no session_id provided
- Maintains 2 previous message exchanges (configurable in config)
- History formatted as: "User: [question]\nAssistant: [response]"
- Conversation context passed to Claude with each query

## API Endpoints

- `POST /api/query` - Main chat endpoint with optional session_id
- `GET /api/courses` - Course statistics/analytics  
- `/` - Static file serving with no-cache headers via custom `DevStaticFiles`

## Error Handling Patterns

- Unicode encoding fallbacks in document processing
- ChromaDB connection error recovery
- Tool execution errors returned to Claude as tool results
- Frontend loading state cleanup on errors
- Comprehensive exception handling with traceback logging

## Data Flow Details

1. **Startup**: Documents auto-loaded from `../docs/` if exists (`app.py:94-104`)
2. **Query Processing**: Session creation → RAG orchestration → AI generation with tools
3. **Tool Execution**: Claude calls search → vector search with filtering → formatted results → synthesis
4. **Response**: Sources tracked via `ToolManager.get_last_sources()` → UI display → source reset
5. **Session Update**: Query-response pair added to conversation history

## Dependencies

Built with Python 3.13+, UV package manager. Core dependencies: FastAPI, ChromaDB, Anthropic SDK, Sentence Transformers. Uses pinned versions for stability (ChromaDB 1.0.15, Anthropic 0.58.2).

- use uv to run python files and manage all dependencies
