# User Query Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Frontend<br/>(script.js)
    participant API as FastAPI<br/>(app.py)
    participant RAG as RAG System<br/>(rag_system.py)
    participant Session as Session Manager<br/>(session_manager.py)
    participant AI as AI Generator<br/>(ai_generator.py)
    participant Tools as Tool Manager<br/>(search_tools.py)
    participant Vector as Vector Store<br/>(vector_store.py)
    participant Claude as Claude API<br/>(Anthropic)

    User->>Frontend: Types query & clicks send
    Frontend->>Frontend: Disable UI, show loading
    Frontend->>API: POST /api/query<br/>{query, session_id}
    
    API->>Session: Create session if none exists
    Session-->>API: session_id
    
    API->>RAG: query(query, session_id)
    
    RAG->>Session: Get conversation history
    Session-->>RAG: Previous 2 exchanges
    
    RAG->>AI: generate_response()<br/>(query, history, tools, tool_manager)
    
    AI->>Claude: Send query with system prompt<br/>& tool definitions
    
    Note over Claude: Claude decides to use search tool
    
    Claude-->>AI: Tool use request<br/>(search_course_content)
    
    AI->>Tools: execute_tool()<br/>(tool_name, parameters)
    
    Tools->>Vector: search(query, course_name, lesson_number)
    Vector-->>Tools: SearchResults<br/>(documents, metadata)
    
    Tools->>Tools: Format results with context<br/>Store sources in last_sources
    Tools-->>AI: Formatted search results
    
    AI->>Claude: Send tool results back
    Claude-->>AI: Final synthesized response
    
    AI-->>RAG: Generated response text
    
    RAG->>Tools: get_last_sources()
    Tools-->>RAG: Source list for UI
    
    RAG->>Session: add_exchange()<br/>(session_id, query, response)
    
    RAG->>Tools: reset_sources()
    
    RAG-->>API: (response, sources)
    
    API-->>Frontend: JSON response<br/>{answer, sources, session_id}
    
    Frontend->>Frontend: Remove loading animation
    Frontend->>Frontend: Render markdown response<br/>Display sources if present
    Frontend->>Frontend: Re-enable UI controls
    Frontend->>User: Show response & sources
```

## Key Components & Data Flow

### 1. **Frontend Layer** (`frontend/script.js`)
- Captures user input
- Manages UI state (loading, disabled controls)
- Handles API communication
- Renders responses with markdown + sources

### 2. **API Layer** (`backend/app.py`)
- FastAPI endpoint `/api/query`
- Session management
- Error handling & HTTP responses

### 3. **Orchestration Layer** (`backend/rag_system.py`)
- Coordinates all components
- Manages conversation history
- Tool registration & source tracking

### 4. **AI Generation** (`backend/ai_generator.py`)
- Claude API interaction
- Tool execution workflow
- Response synthesis

### 5. **Search Tools** (`backend/search_tools.py`)
- `CourseSearchTool` for semantic search
- Result formatting with context
- Source tracking for UI

### 6. **Data Layer** (`backend/vector_store.py`)
- ChromaDB vector search
- Course/lesson filtering
- Embedding-based retrieval

## Architecture Highlights

- **Tool-Based**: Claude autonomously decides when to search
- **Session-Aware**: Maintains conversation context (2 previous exchanges)
- **Source Tracking**: UI shows which courses/lessons were referenced
- **Async Flow**: Loading states while processing
- **Error Handling**: Graceful failure modes throughout