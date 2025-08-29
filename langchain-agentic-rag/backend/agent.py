"""LangChain agent implementation for agentic RAG system."""

import logging
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .llm_client import llm_client
from .agent_tools import get_agent_tools
from .config import settings

logger = logging.getLogger(__name__)


class AgenticRAGAgent:
    """Agentic RAG system using LangChain ReAct agent."""
    
    def __init__(self):
        self.llm = llm_client.get_llm()
        self.tools = get_agent_tools()
        
        # Create ReAct prompt template
        self.prompt = PromptTemplate.from_template("""
You are a helpful AI assistant with access to a knowledge base. Your job is to help users by searching for relevant information and providing comprehensive, accurate answers.

Available tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Guidelines:
1. Always search the knowledge base first to find relevant information
2. Use multiple searches if needed to gather comprehensive information  
3. Provide detailed, well-structured answers based on the search results
4. If no relevant information is found, say so clearly
5. Include source references when possible
6. Be conversational and helpful in your responses

Question: {input}
{agent_scratchpad}
""")
        
        # Create the ReAct agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=settings.agent_verbose,
            max_iterations=settings.max_agent_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        logger.info("AgenticRAGAgent initialized")
    
    async def query(
        self, 
        question: str, 
        chat_history: Optional[List[BaseMessage]] = None
    ) -> Dict[str, Any]:
        """Process a user query using the agentic RAG system."""
        try:
            # Prepare input with chat history context if available
            input_text = question
            if chat_history:
                context = self._format_chat_history(chat_history)
                if context:
                    input_text = f"Previous conversation:\n{context}\n\nCurrent question: {question}"
            
            # Execute agent
            result = await self.agent_executor.ainvoke({
                "input": input_text
            })
            
            # Format response
            response = {
                "answer": result["output"],
                "intermediate_steps": result.get("intermediate_steps", []),
                "agent_steps": self._format_agent_steps(result.get("intermediate_steps", [])),
                "sources": self._extract_sources(result.get("intermediate_steps", []))
            }
            
            logger.info(f"Agent completed query with {len(response['intermediate_steps'])} steps")
            return response
            
        except Exception as e:
            logger.error(f"Error in agent query: {e}")
            raise
    
    def _format_chat_history(self, chat_history: List[BaseMessage]) -> str:
        """Format chat history for context."""
        formatted = []
        for message in chat_history[-settings.max_conversation_history:]:  # Limit history
            if isinstance(message, HumanMessage):
                formatted.append(f"User: {message.content}")
            elif isinstance(message, AIMessage):
                formatted.append(f"Assistant: {message.content}")
        
        return "\n".join(formatted)
    
    def _format_agent_steps(self, intermediate_steps: List) -> List[Dict[str, Any]]:
        """Format intermediate steps for response."""
        formatted_steps = []
        
        for step in intermediate_steps:
            if len(step) == 2:
                action, observation = step
                formatted_step = {
                    "tool": action.tool,
                    "tool_input": action.tool_input,
                    "observation": str(observation)[:1000] + "..." if len(str(observation)) > 1000 else str(observation),
                    "thought": getattr(action, 'log', '').split('Action:')[0].replace('Thought:', '').strip()
                }
                formatted_steps.append(formatted_step)
        
        return formatted_steps
    
    def _extract_sources(self, intermediate_steps: List) -> List[Dict[str, Any]]:
        """Extract source information from agent steps."""
        sources = []
        
        for step in intermediate_steps:
            if len(step) == 2:
                action, observation = step
                if action.tool == "knowledge_search":
                    # Try to extract source information from search results
                    obs_str = str(observation)
                    if "Title:" in obs_str and "Content:" in obs_str:
                        # Parse search results to extract sources
                        results = obs_str.split("---")
                        for result in results:
                            if "Title:" in result:
                                lines = result.strip().split("\n")
                                title = ""
                                for line in lines:
                                    if line.startswith("Title:"):
                                        title = line.replace("Title:", "").strip()
                                        break
                                
                                if title:
                                    sources.append({
                                        "title": title,
                                        "type": "knowledge_base",
                                        "tool": "knowledge_search"
                                    })
        
        # Remove duplicates
        unique_sources = []
        seen_titles = set()
        for source in sources:
            if source["title"] not in seen_titles:
                unique_sources.append(source)
                seen_titles.add(source["title"])
        
        return unique_sources
    
    async def simple_query(self, question: str) -> str:
        """Simple query interface that returns just the answer."""
        try:
            result = await self.query(question)
            return result["answer"]
        except Exception as e:
            logger.error(f"Error in simple query: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def get_tools_info(self) -> Dict[str, Any]:
        """Get information about available tools."""
        tools_info = []
        for tool in self.tools:
            tools_info.append({
                "name": tool.name,
                "description": tool.description,
                "args_schema": tool.args_schema.schema() if tool.args_schema else {}
            })
        
        return {
            "tools": tools_info,
            "total_tools": len(self.tools)
        }


# Global agent instance
rag_agent = AgenticRAGAgent()