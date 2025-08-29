"""Knowledge base management for document storage and retrieval."""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import Document, DocumentUploadResponse, KnowledgeBaseStats
from .vector_store import vector_store

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Knowledge base manager for document operations."""
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        logger.info("KnowledgeBase initialized")
    
    async def add_document(
        self, 
        title: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentUploadResponse:
        """Add a document to the knowledge base."""
        try:
            # Create document
            doc_id = str(uuid.uuid4())
            document = Document(
                id=doc_id,
                title=title,
                content=content,
                metadata=metadata or {}
            )
            
            # Store document
            self.documents[doc_id] = document
            
            # Add to vector store
            chunk_ids = await vector_store.add_document(document)
            
            logger.info(f"Added document '{title}' with {len(chunk_ids)} chunks")
            
            return DocumentUploadResponse(
                document_id=doc_id,
                chunks_created=len(chunk_ids),
                message=f"Successfully added document '{title}'"
            )
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(document_id)
    
    async def list_documents(self) -> List[Document]:
        """List all documents."""
        return list(self.documents.values())
    
    async def update_document(
        self, 
        document_id: str, 
        title: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a document."""
        try:
            document = self.documents.get(document_id)
            if not document:
                return False
            
            # Remove old version from vector store
            await vector_store.delete_document(document_id)
            
            # Update document
            if title is not None:
                document.title = title
            if content is not None:
                document.content = content
            if metadata is not None:
                document.metadata.update(metadata)
            
            document.updated_at = datetime.utcnow()
            
            # Re-add to vector store
            await vector_store.add_document(document)
            
            logger.info(f"Updated document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            raise
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document."""
        try:
            if document_id not in self.documents:
                return False
            
            # Remove from vector store
            await vector_store.delete_document(document_id)
            
            # Remove from memory
            del self.documents[document_id]
            
            logger.info(f"Deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            raise
    
    async def search_documents(
        self, 
        query: str, 
        max_results: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search documents using vector similarity."""
        try:
            results = await vector_store.search_similar(
                query=query,
                k=max_results,
                filter_criteria=filter_criteria
            )
            
            formatted_results = []
            for doc, score in results:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score,
                    "document_id": doc.metadata.get("document_id"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "title": doc.metadata.get("title", "Unknown")
                }
                formatted_results.append(result)
            
            logger.info(f"Search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    async def get_stats(self) -> KnowledgeBaseStats:
        """Get knowledge base statistics."""
        try:
            vector_stats = await vector_store.get_collection_stats()
            
            stats = KnowledgeBaseStats(
                total_documents=len(self.documents),
                total_chunks=vector_stats.get("total_chunks", 0),
                total_embeddings=vector_stats.get("total_chunks", 0),  # Same as chunks
                collection_name=vector_stats.get("collection_name", "unknown"),
                last_updated=datetime.utcnow()
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            # Return default stats on error
            return KnowledgeBaseStats(
                total_documents=0,
                total_chunks=0,
                total_embeddings=0,
                collection_name="unknown",
                last_updated=datetime.utcnow()
            )
    
    async def load_documents_from_directory(self, directory_path: str) -> int:
        """Load documents from a directory."""
        import os
        import asyncio
        
        loaded_count = 0
        
        try:
            if not os.path.exists(directory_path):
                logger.warning(f"Directory {directory_path} does not exist")
                return 0
            
            for filename in os.listdir(directory_path):
                if filename.endswith(('.txt', '.md', '.py', '.json')):
                    filepath = os.path.join(directory_path, filename)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        await self.add_document(
                            title=filename,
                            content=content,
                            metadata={'source_file': filepath, 'filename': filename}
                        )
                        loaded_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error loading file {filepath}: {e}")
                        continue
            
            logger.info(f"Loaded {loaded_count} documents from {directory_path}")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Error loading documents from directory: {e}")
            return loaded_count


# Global knowledge base instance
knowledge_base = KnowledgeBase()