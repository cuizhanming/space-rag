"""Pydantic models for the LangChain Agentic RAG system."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document model for knowledge base."""
    
    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(BaseModel):
    """Document chunk model for vector storage."""
    
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    chunk_index: int = Field(..., description="Chunk position in document")


class QueryRequest(BaseModel):
    """User query request model."""
    
    query: str = Field(..., min_length=1, description="User query text")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    max_results: Optional[int] = Field(5, ge=1, le=20, description="Maximum search results")
    use_agent: bool = Field(True, description="Whether to use agentic capabilities")


class QueryResponse(BaseModel):
    """Query response model."""
    
    response: str = Field(..., description="Generated response")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents")
    session_id: str = Field(..., description="Session identifier")
    agent_steps: Optional[List[Dict[str, Any]]] = Field(None, description="Agent reasoning steps")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")


class ChatMessage(BaseModel):
    """Chat message model."""
    
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Message metadata")


class ChatSession(BaseModel):
    """Chat session model."""
    
    session_id: str = Field(..., description="Session identifier")
    messages: List[ChatMessage] = Field(default_factory=list, description="Session messages")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")


class DocumentUploadRequest(BaseModel):
    """Document upload request model."""
    
    title: str = Field(..., min_length=1, description="Document title")
    content: str = Field(..., min_length=1, description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    
    document_id: str = Field(..., description="Created document ID")
    chunks_created: int = Field(..., description="Number of chunks created")
    message: str = Field(..., description="Success message")


class KnowledgeBaseStats(BaseModel):
    """Knowledge base statistics model."""
    
    total_documents: int = Field(..., description="Total number of documents")
    total_chunks: int = Field(..., description="Total number of chunks")
    total_embeddings: int = Field(..., description="Total number of embeddings")
    collection_name: str = Field(..., description="Vector collection name")
    last_updated: datetime = Field(..., description="Last update timestamp")


class AgentTool(BaseModel):
    """Agent tool model."""
    
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


class AgentStep(BaseModel):
    """Agent reasoning step model."""
    
    tool: str = Field(..., description="Tool used")
    tool_input: Dict[str, Any] = Field(..., description="Tool input parameters")
    observation: str = Field(..., description="Tool output/observation")
    thought: Optional[str] = Field(None, description="Agent reasoning")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)