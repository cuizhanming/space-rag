"""Session management for chat conversations."""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .models import ChatSession, ChatMessage

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages chat sessions and conversation history."""
    
    def __init__(self, max_session_age_hours: int = 24):
        self.sessions: Dict[str, ChatSession] = {}
        self.max_session_age_hours = max_session_age_hours
        logger.info("SessionManager initialized")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new chat session."""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        session = ChatSession(session_id=session_id)
        self.sessions[session_id] = session
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID."""
        session = self.sessions.get(session_id)
        
        if session:
            # Check if session is expired
            age = datetime.utcnow() - session.created_at
            if age.total_seconds() > (self.max_session_age_hours * 3600):
                logger.info(f"Session {session_id} expired, removing")
                del self.sessions[session_id]
                return None
        
        return session
    
    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str, 
        metadata: Optional[Dict] = None
    ) -> bool:
        """Add a message to a session."""
        session = self.get_session(session_id)
        if not session:
            # Create new session if it doesn't exist
            self.create_session(session_id)
            session = self.sessions[session_id]
        
        message = ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        session.messages.append(message)
        session.updated_at = datetime.utcnow()
        
        logger.debug(f"Added {role} message to session {session_id}")
        return True
    
    def get_messages(
        self, 
        session_id: str, 
        limit: Optional[int] = None
    ) -> List[ChatMessage]:
        """Get messages from a session."""
        session = self.get_session(session_id)
        if not session:
            return []
        
        messages = session.messages
        if limit:
            messages = messages[-limit:]  # Get last N messages
        
        return messages
    
    def get_langchain_messages(
        self, 
        session_id: str, 
        limit: Optional[int] = None
    ) -> List[BaseMessage]:
        """Get messages in LangChain format."""
        messages = self.get_messages(session_id, limit)
        langchain_messages = []
        
        for msg in messages:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))
        
        return langchain_messages
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self.sessions.keys())
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session information."""
        session = self.get_session(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "message_count": len(session.messages),
            "metadata": session.metadata
        }
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        expired_sessions = []
        cutoff_time = datetime.utcnow() - timedelta(hours=self.max_session_age_hours)
        
        for session_id, session in self.sessions.items():
            if session.created_at < cutoff_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def get_stats(self) -> Dict:
        """Get session manager statistics."""
        total_sessions = len(self.sessions)
        total_messages = sum(len(session.messages) for session in self.sessions.values())
        
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "max_session_age_hours": self.max_session_age_hours
        }


# Global session manager instance
session_manager = SessionManager()