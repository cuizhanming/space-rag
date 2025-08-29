"""Vector store management using ChromaDB and LangChain."""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document as LangChainDocument

from .config import settings
from .models import Document, DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store manager using ChromaDB with LangChain integration."""
    
    def __init__(self, collection_name: str = "langchain_agentic_rag"):
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
        
        # Initialize LangChain Chroma
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        
        logger.info(f"VectorStore initialized with collection: {self.collection_name}")
    
    async def add_document(self, document: Document) -> List[str]:
        """Add a document to the vector store."""
        try:
            chunks = self._create_chunks(document)
            chunk_ids = []
            
            # Convert to LangChain documents
            langchain_docs = []
            for chunk in chunks:
                langchain_doc = LangChainDocument(
                    page_content=chunk.content,
                    metadata={
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.id,
                        "chunk_index": chunk.chunk_index,
                        "title": document.title,
                        **chunk.metadata,
                        **document.metadata,
                    }
                )
                langchain_docs.append(langchain_doc)
                chunk_ids.append(chunk.id)
            
            # Add to vector store
            self.vector_store.add_documents(
                documents=langchain_docs,
                ids=chunk_ids
            )
            
            logger.info(f"Added document {document.id} with {len(chunks)} chunks")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error adding document {document.id}: {e}")
            raise
    
    async def search_similar(
        self, 
        query: str, 
        k: int = 5, 
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[LangChainDocument, float]]:
        """Search for similar documents."""
        try:
            # Use similarity search with score
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_criteria
            )
            
            logger.info(f"Search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    async def search_similar_langchain(
        self, 
        query: str, 
        k: int = 5, 
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[LangChainDocument]:
        """Search for similar documents (LangChain compatible)."""
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_criteria
            )
            
            logger.info(f"LangChain search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in LangChain search: {e}")
            raise
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks from the vector store."""
        try:
            # Get collection directly for metadata filtering
            collection = self.chroma_client.get_collection(self.collection_name)
            
            # Find all chunks for this document
            results = collection.get(
                where={"document_id": document_id}
            )
            
            if results['ids']:
                # Delete from LangChain vector store
                self.vector_store.delete(ids=results['ids'])
                logger.info(f"Deleted document {document_id} and {len(results['ids'])} chunks")
                return True
            else:
                logger.warning(f"Document {document_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            raise
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            count = collection.count()
            
            # Get unique document count
            results = collection.get(include=['metadatas'])
            unique_docs = set()
            for metadata in results['metadatas']:
                if 'document_id' in metadata:
                    unique_docs.add(metadata['document_id'])
            
            return {
                "total_chunks": count,
                "unique_documents": len(unique_docs),
                "collection_name": self.collection_name,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "total_chunks": 0,
                "unique_documents": 0,
                "collection_name": self.collection_name,
                "error": str(e)
            }
    
    def _create_chunks(self, document: Document) -> List[DocumentChunk]:
        """Create chunks from a document."""
        content = document.content
        chunks = []
        chunk_index = 0
        
        # Simple chunking strategy
        chunk_size = settings.chunk_size
        overlap = settings.chunk_overlap
        
        start = 0
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at word boundaries
            if end < len(content):
                # Look for the last space within the chunk
                while end > start and content[end] not in [' ', '\n', '.', '!', '?']:
                    end -= 1
                
                # If no good break point found, use original end
                if end == start:
                    end = start + chunk_size
            
            chunk_content = content[start:end].strip()
            
            if chunk_content:  # Only create non-empty chunks
                chunk = DocumentChunk(
                    id=str(uuid.uuid4()),
                    document_id=document.id,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    metadata={
                        "start_pos": start,
                        "end_pos": end,
                        "chunk_length": len(chunk_content)
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + 1, end - overlap)
            
            # Prevent infinite loop
            if start >= len(content):
                break
        
        return chunks
    
    async def close(self):
        """Close the vector store connections."""
        try:
            # ChromaDB client doesn't need explicit closing
            logger.info("Vector store closed")
        except Exception as e:
            logger.error(f"Error closing vector store: {e}")


# Global vector store instance
vector_store = VectorStore()