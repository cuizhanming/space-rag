"""Agent tools for the LangChain-based RAG system."""

import logging
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .knowledge_base import knowledge_base
from .config import settings

logger = logging.getLogger(__name__)


class KnowledgeSearchInput(BaseModel):
    """Input for knowledge search tool."""
    query: str = Field(description="Search query for the knowledge base")
    max_results: int = Field(default=5, description="Maximum number of results to return")


class KnowledgeSearchTool(BaseTool):
    """Tool for searching the knowledge base."""
    
    name: str = "knowledge_search"
    description: str = """
    Search the knowledge base for relevant information. 
    Use this tool to find documents and content related to the user's query.
    """
    args_schema: Type[BaseModel] = KnowledgeSearchInput
    
    async def _arun(
        self, 
        query: str, 
        max_results: int = 5,
        **kwargs: Any
    ) -> str:
        """Search the knowledge base asynchronously."""
        try:
            results = await knowledge_base.search_documents(
                query=query,
                max_results=max_results
            )
            
            if not results:
                return "No relevant documents found in the knowledge base."
            
            # Format results for the agent
            formatted_results = []
            for i, result in enumerate(results, 1):
                content = result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"]
                title = result.get("title", "Unknown")
                score = result.get("similarity_score", 0.0)
                
                formatted_result = f"""
Result {i} (Score: {score:.3f}):
Title: {title}
Content: {content}
---
"""
                formatted_results.append(formatted_result)
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Error in knowledge search: {e}")
            return f"Error searching knowledge base: {str(e)}"
    
    def _run(
        self, 
        query: str, 
        max_results: int = 5,
        **kwargs: Any
    ) -> str:
        """Search the knowledge base synchronously."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._arun(query, max_results, **kwargs))
        except RuntimeError:
            # If no event loop is running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(query, max_results, **kwargs))
            finally:
                loop.close()


class DocumentStatsInput(BaseModel):
    """Input for document stats tool."""
    pass


class DocumentStatsTool(BaseTool):
    """Tool for getting knowledge base statistics."""
    
    name: str = "document_stats"
    description: str = """
    Get statistics about the knowledge base including number of documents, 
    chunks, and other metadata. Use this when the user asks about the knowledge base status.
    """
    args_schema: Type[BaseModel] = DocumentStatsInput
    
    async def _arun(self, **kwargs: Any) -> str:
        """Get document statistics asynchronously."""
        try:
            stats = await knowledge_base.get_stats()
            
            return f"""
Knowledge Base Statistics:
- Total Documents: {stats.total_documents}
- Total Chunks: {stats.total_chunks}
- Total Embeddings: {stats.total_embeddings}
- Collection Name: {stats.collection_name}
- Last Updated: {stats.last_updated.strftime('%Y-%m-%d %H:%M:%S')}
"""
            
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return f"Error getting knowledge base statistics: {str(e)}"
    
    def _run(self, **kwargs: Any) -> str:
        """Get document statistics synchronously."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._arun(**kwargs))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(**kwargs))
            finally:
                loop.close()


class ListDocumentsInput(BaseModel):
    """Input for list documents tool."""
    limit: int = Field(default=10, description="Maximum number of documents to list")


class ListDocumentsTool(BaseTool):
    """Tool for listing documents in the knowledge base."""
    
    name: str = "list_documents"
    description: str = """
    List documents in the knowledge base. Use this to show available documents
    or when the user asks what documents are available.
    """
    args_schema: Type[BaseModel] = ListDocumentsInput
    
    async def _arun(self, limit: int = 10, **kwargs: Any) -> str:
        """List documents asynchronously."""
        try:
            documents = await knowledge_base.list_documents()
            
            if not documents:
                return "No documents found in the knowledge base."
            
            # Limit results
            documents = documents[:limit]
            
            formatted_docs = []
            for i, doc in enumerate(documents, 1):
                content_preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
                formatted_doc = f"""
{i}. Title: {doc.title}
   ID: {doc.id}
   Created: {doc.created_at.strftime('%Y-%m-%d %H:%M:%S')}
   Preview: {content_preview}
---
"""
                formatted_docs.append(formatted_doc)
            
            total_docs = len(await knowledge_base.list_documents())
            result = f"Documents in Knowledge Base (showing {len(documents)} of {total_docs}):\n" + "\n".join(formatted_docs)
            
            if total_docs > limit:
                result += f"\n... and {total_docs - limit} more documents."
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return f"Error listing documents: {str(e)}"
    
    def _run(self, limit: int = 10, **kwargs: Any) -> str:
        """List documents synchronously."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._arun(limit, **kwargs))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(limit, **kwargs))
            finally:
                loop.close()


def get_agent_tools() -> List[BaseTool]:
    """Get all available agent tools."""
    return [
        KnowledgeSearchTool(),
        DocumentStatsTool(),
        ListDocumentsTool(),
    ]


def get_tool_descriptions() -> str:
    """Get descriptions of all available tools."""
    tools = get_agent_tools()
    descriptions = []
    
    for tool in tools:
        descriptions.append(f"- {tool.name}: {tool.description.strip()}")
    
    return "\n".join(descriptions)