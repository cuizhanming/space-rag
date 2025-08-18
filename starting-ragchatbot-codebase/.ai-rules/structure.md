---
title: Project Structure
description: "Defines the project organization, file structure, and naming conventions."
inclusion: always
---

# Project Structure

## Root Directory Layout
```
/
├── .ai-rules/              # AI assistant steering files
├── backend/                # Python FastAPI backend
├── docs/                   # Course materials storage
├── frontend/               # Web interface files
├── main.py                 # Simple entry point (not used in production)
├── pyproject.toml          # Python project configuration
├── run.sh                  # Application startup script
├── uv.lock                 # Package dependency lock file
└── README.md               # Project documentation
```

## Backend Structure (`backend/`)
```
backend/
├── app.py                  # FastAPI application and API endpoints
├── ai_generator.py         # Anthropic Claude API integration
├── config.py               # Configuration settings and environment variables
├── document_processor.py   # Text processing and course parsing
├── models.py               # Pydantic data models
├── rag_system.py           # Main RAG orchestrator
├── search_tools.py         # Tool-based search architecture
├── session_manager.py      # Conversation context management
└── vector_store.py         # ChromaDB interface
```

## Frontend Structure (`frontend/`)
```
frontend/
├── index.html              # Main web interface
├── script.js               # JavaScript application logic
└── style.css               # CSS styling
```

## Course Materials (`docs/`)
```
docs/
├── course1_script.txt      # Individual course transcripts
├── course2_script.txt      # Structured course content
├── course3_script.txt      # Lesson-based organization
└── course4_script.txt      # Text files with course metadata
```

## File Naming Conventions

### Python Files
- **snake_case**: All Python modules use lowercase with underscores
- **Descriptive names**: Clear indication of component purpose
- **Single responsibility**: Each file handles one major component

### Course Documents
- **course{n}_script.txt**: Sequential numbering for course materials
- **UTF-8 encoding**: All text files use UTF-8 encoding
- **Structured format**: Standardized course metadata format

### Configuration Files
- **pyproject.toml**: Python project metadata and dependencies
- **uv.lock**: Locked dependency versions
- **.env**: Environment variables (not in repository)

## Component Organization

### Core Components
1. **API Layer** (`app.py`): FastAPI endpoints and middleware
2. **Business Logic** (`rag_system.py`): Core RAG functionality
3. **Data Access** (`vector_store.py`): Database operations
4. **External Services** (`ai_generator.py`): Third-party integrations
5. **Utilities** (`document_processor.py`, `session_manager.py`): Supporting functions

### Data Models
- **Centralized in `models.py`**: All Pydantic models in one file
- **Type safety**: Full type annotations throughout
- **Validation**: Input/output validation with Pydantic

### Tool Architecture
- **Abstract base class**: `Tool` interface in `search_tools.py`
- **Concrete implementations**: `CourseSearchTool` for semantic search
- **Tool management**: `ToolManager` for registration and execution

## Directory Conventions

### Backend Development
- **Single directory**: All backend code in `backend/`
- **Flat structure**: No deep nesting, related files grouped together
- **Clear separation**: Business logic separate from API layer

### Frontend Development
- **Static files**: HTML, CSS, JS served directly
- **No build process**: Vanilla technologies, no compilation needed
- **CDN dependencies**: External libraries loaded from CDN

### Data Storage
- **docs/**: Source course materials
- **chroma_db/**: Vector database storage (created at runtime)
- **Generated files**: Database files not in version control

## Import Patterns

### Relative Imports
- **Within backend**: Direct imports between backend modules
- **Example**: `from config import config`

### External Dependencies
- **Standard library**: Direct imports
- **Third-party**: From installed packages
- **Type annotations**: From `typing` module

## Development Workflow Structure

### Starting Point
- **run.sh**: Primary development entry point
- **Backend focus**: Application starts from backend directory
- **Port 8000**: Standard development port

### Environment Setup
- **uv sync**: Dependency installation
- **Environment variables**: `.env` file in root
- **Database initialization**: Automatic on startup

### File Modification Guidelines
- **Backend changes**: Restart uvicorn for Python changes
- **Frontend changes**: Browser refresh (no-cache headers in dev)
- **Course materials**: Automatic processing on startup