import anthropic
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
- **search_course_content**: Search for specific content within course materials
- **get_course_outline**: Get complete course structure including title, link, instructor, and all lessons with their titles and numbers

Tool Usage Guidelines:
- **Course outline queries**: Use get_course_outline tool for questions about course structure, lesson lists, or general course information
- **Content-specific queries**: Use search_course_content tool for questions about specific topics, concepts, or detailed educational materials
- **Multi-step queries**: You may use up to 2 tools sequentially for complex questions requiring multiple searches or comparisons
- **Tool sequencing**: Use initial tool results to inform subsequent tool calls for comprehensive answers
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use get_course_outline tool, then provide the course title, course link, and complete lesson information (lesson number and title for each lesson)
- **Course content questions**: Use search_course_content tool, then answer based on retrieved content
- **Complex queries**: Use sequential tools as needed, then provide comprehensive synthesis
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
        max_tool_rounds: int = 2,
    ) -> str:
        """
        Generate AI response with sequential tool usage support.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_tool_rounds: Maximum number of sequential tool calling rounds

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize conversation state
        messages = [{"role": "user", "content": query}]
        tool_rounds_used = 0

        # Sequential tool calling loop
        while tool_rounds_used < max_tool_rounds:
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }

            # Include tools only if we haven't exceeded rounds and tools are available
            if tools and tool_rounds_used < max_tool_rounds:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Get response from Claude
            response = self.client.messages.create(**api_params)

            # If no tool use, return the response
            if response.stop_reason != "tool_use" or not tool_manager:
                return response.content[0].text

            # Execute tools and continue conversation
            messages.append({"role": "assistant", "content": response.content})
            tool_results = self._execute_tools(response, tool_manager)
            messages.append({"role": "user", "content": tool_results})

            tool_rounds_used += 1

        # Final response without tools after max rounds
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _execute_tools(self, response, tool_manager) -> List[Dict[str, Any]]:
        """
        Execute tools from Claude's response and return formatted results.

        Args:
            response: The Claude response containing tool use requests
            tool_manager: Manager to execute tools

        Returns:
            List of formatted tool results for Claude's conversation
        """
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    # Handle tool execution errors gracefully
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution error: {str(e)}",
                        }
                    )
        return tool_results
