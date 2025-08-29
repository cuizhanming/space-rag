"""LLM client for Gemini API integration with LangChain."""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from .config import settings

logger = logging.getLogger(__name__)


class GeminiLLMClient:
    """Gemini API client using LangChain integration."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            google_api_key=settings.gemini_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            convert_system_message_to_human=True
        )
        
        logger.info(f"GeminiLLMClient initialized with model: {settings.model_name}")
    
    async def generate_response(
        self, 
        messages: List[BaseMessage], 
        **kwargs
    ) -> str:
        """Generate a response using the Gemini model."""
        try:
            response = await self.llm.ainvoke(messages, **kwargs)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def generate_response_from_text(
        self, 
        text: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a response from text input."""
        try:
            messages = []
            
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            
            messages.append(HumanMessage(content=text))
            
            return await self.generate_response(messages, **kwargs)
            
        except Exception as e:
            logger.error(f"Error generating text response: {e}")
            raise
    
    def get_llm(self) -> BaseLLM:
        """Get the underlying LangChain LLM instance."""
        return self.llm
    
    async def stream_response(
        self, 
        messages: List[BaseMessage], 
        **kwargs
    ):
        """Stream response from the model."""
        try:
            async for chunk in self.llm.astream(messages, **kwargs):
                yield chunk.content
                
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            raise


# Global LLM client instance
llm_client = GeminiLLMClient()