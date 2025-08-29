"""FastAPI application for LangChain Agentic RAG system."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from .config import settings
from .models import (
    QueryRequest, QueryResponse, DocumentUploadRequest, 
    DocumentUploadResponse, KnowledgeBaseStats, ErrorResponse
)
from .knowledge_base import knowledge_base
from .agent import rag_agent
from .session_manager import session_manager
from .llm_client import llm_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting LangChain Agentic RAG system...")
    
    # Load documents from docs directory if it exists
    docs_path = "../docs"
    if os.path.exists(docs_path):
        logger.info(f"Loading documents from {docs_path}")
        try:
            loaded_count = await knowledge_base.load_documents_from_directory(docs_path)
            logger.info(f"Loaded {loaded_count} documents")
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
    
    # Start periodic cleanup task
    async def cleanup_sessions():
        while True:
            import asyncio
            await asyncio.sleep(3600)  # Run every hour
            session_manager.cleanup_expired_sessions()
    
    import asyncio
    asyncio.create_task(cleanup_sessions())
    
    logger.info("System startup complete")
    yield
    
    logger.info("Shutting down LangChain Agentic RAG system...")


# Create FastAPI app
app = FastAPI(
    title="LangChain Agentic RAG System",
    description="An intelligent RAG system using LangChain, Gemini API, and agentic capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint serving a simple welcome page."""
    return """
    <html>
        <head>
            <title>LangChain Agentic RAG System</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }
                .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                .endpoint { background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; font-family: monospace; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 LangChain Agentic RAG System</h1>
                <p>Intelligent knowledge retrieval with LangChain and Gemini API</p>
            </div>
            
            <div class="section">
                <h2>🚀 Features</h2>
                <ul>
                    <li>Agentic RAG with LangChain ReAct agent</li>
                    <li>Gemini API integration for advanced language understanding</li>
                    <li>Vector-based knowledge search with ChromaDB</li>
                    <li>Session management for conversation history</li>
                    <li>RESTful API for easy integration</li>
                    <li>OpenWebUI compatibility</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>📚 API Endpoints</h2>
                <div class="endpoint">POST /api/query - Submit queries to the RAG system</div>
                <div class="endpoint">POST /api/upload - Upload documents to knowledge base</div>
                <div class="endpoint">GET /api/stats - Get knowledge base statistics</div>
                <div class="endpoint">GET /api/docs - API documentation (Swagger UI)</div>
                <div class="endpoint">GET /api/health - Health check endpoint</div>
            </div>
            
            <div class="section">
                <h2>🔗 Quick Links</h2>
                <p><a href="/docs">API Documentation</a> | <a href="/api/stats">Knowledge Base Stats</a> | <a href="/api/health">Health Check</a></p>
            </div>
        </body>
    </html>
    """


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Process a user query using the agentic RAG system."""
    try:
        # Create session if not provided
        session_id = request.session_id or session_manager.create_session()
        
        # Add user message to session
        session_manager.add_message(session_id, "user", request.query)
        
        # Get conversation history if available
        chat_history = session_manager.get_langchain_messages(session_id, limit=10)
        
        # Process query with agent
        if request.use_agent:
            result = await rag_agent.query(request.query, chat_history[:-1])  # Exclude current query
            response_text = result["answer"]
            sources = result["sources"]
            agent_steps = result["agent_steps"]
        else:
            # Simple LLM response without agent
            response_text = await llm_client.generate_response_from_text(
                request.query,
                system_prompt="You are a helpful AI assistant. Please provide a clear and concise response."
            )
            sources = []
            agent_steps = []
        
        # Add assistant response to session
        session_manager.add_message(session_id, "assistant", response_text)
        
        return QueryResponse(
            response=response_text,
            sources=sources,
            session_id=session_id,
            agent_steps=agent_steps,
            metadata={
                "use_agent": request.use_agent,
                "max_results": request.max_results
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload", response_model=DocumentUploadResponse)
async def upload_document(request: DocumentUploadRequest) -> DocumentUploadResponse:
    """Upload a document to the knowledge base."""
    try:
        result = await knowledge_base.add_document(
            title=request.title,
            content=request.content,
            metadata=request.metadata
        )
        return result
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=KnowledgeBaseStats)
async def get_stats() -> KnowledgeBaseStats:
    """Get knowledge base statistics."""
    try:
        return await knowledge_base.get_stats()
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        # Test key components
        stats = await knowledge_base.get_stats()
        session_stats = session_manager.get_stats()
        
        return {
            "status": "healthy",
            "service": "LangChain Agentic RAG",
            "version": "1.0.0",
            "components": {
                "knowledge_base": {
                    "status": "ok",
                    "documents": stats.total_documents,
                    "chunks": stats.total_chunks
                },
                "session_manager": {
                    "status": "ok",
                    "active_sessions": session_stats["total_sessions"]
                },
                "agent": {
                    "status": "ok",
                    "tools": len(rag_agent.tools)
                }
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/api/sessions/{session_id}")
async def get_session_info(session_id: str) -> Dict[str, Any]:
    """Get session information."""
    try:
        session_info = session_manager.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
        return session_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> Dict[str, str]:
    """Delete a session."""
    try:
        success = session_manager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"message": "Session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tools")
async def get_agent_tools() -> Dict[str, Any]:
    """Get information about available agent tools."""
    try:
        return rag_agent.get_tools_info()
    except Exception as e:
        logger.error(f"Error getting tools info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# OpenWebUI compatibility endpoints
@app.get("/api/models")
async def openwebui_models():
    """OpenWebUI compatible models endpoint."""
    return {
        "data": [
            {
                "id": "langchain-agentic-rag",
                "object": "model",
                "name": "LangChain Agentic RAG",
                "description": "Intelligent RAG system with agentic capabilities"
            }
        ]
    }


@app.post("/api/chat/completions")
async def openwebui_chat_completions(request: dict):
    """OpenWebUI compatible chat completions endpoint."""
    try:
        messages = request.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Get the last user message
        last_message = messages[-1]["content"]
        
        # Process with our RAG system
        query_request = QueryRequest(
            query=last_message,
            use_agent=True
        )
        
        response = await query_endpoint(query_request)
        
        # Format for OpenWebUI
        return {
            "id": f"chatcmpl-{response.session_id}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.response
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(last_message.split()),
                "completion_tokens": len(response.response.split()),
                "total_tokens": len(last_message.split()) + len(response.response.split())
            }
        }
        
    except Exception as e:
        logger.error(f"Error in chat completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An internal server error occurred"
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info" if settings.debug else "warning"
    )