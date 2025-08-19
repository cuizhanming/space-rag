"""
Tests for AIGenerator tool calling functionality to debug RAG chatbot failures.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class MockAnthropicClient:
    """Mock Anthropic client for testing"""
    
    def __init__(self):
        self.messages = Mock()


class MockResponse:
    """Mock Anthropic API response"""
    
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class MockContentBlock:
    """Mock content block for tool use"""
    
    def __init__(self, block_type, text=None, name=None, input=None, tool_use_id=None):
        self.type = block_type
        self.text = text
        self.name = name
        self.input = input or {}
        self.id = tool_use_id or "test_tool_id"


class TestAIGenerator(unittest.TestCase):
    """Test AIGenerator tool calling functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test_api_key"
        self.model = "claude-3-sonnet-20241022"
        
        # Mock Anthropic client
        with patch('anthropic.Anthropic') as mock_anthropic:
            self.mock_client = MockAnthropicClient()
            mock_anthropic.return_value = self.mock_client
            self.ai_generator = AIGenerator(self.api_key, self.model)
            
        # Mock tool manager
        self.mock_tool_manager = Mock()

    def test_generate_response_without_tools(self):
        """Test basic response generation without tool calls"""
        # Setup mock response
        mock_content = [MockContentBlock("text", text="This is a test response")]
        mock_response = MockResponse(content=mock_content, stop_reason="end_turn")
        self.mock_client.messages.create.return_value = mock_response

        # Call generate_response
        result = self.ai_generator.generate_response("What is Python?")

        # Assertions
        self.assertEqual(result, "This is a test response")
        self.mock_client.messages.create.assert_called_once()
        
        # Check API call parameters
        call_args = self.mock_client.messages.create.call_args
        self.assertIn("messages", call_args.kwargs)
        self.assertIn("system", call_args.kwargs)
        self.assertEqual(call_args.kwargs["model"], self.model)

    def test_generate_response_with_tools_no_tool_use(self):
        """Test response generation with tools available but not used"""
        # Setup mock tools and response
        mock_tools = [{"name": "search_course_content", "description": "Search courses"}]
        mock_content = [MockContentBlock("text", text="General knowledge response")]
        mock_response = MockResponse(content=mock_content, stop_reason="end_turn")
        self.mock_client.messages.create.return_value = mock_response

        # Call generate_response with tools
        result = self.ai_generator.generate_response(
            "What is 2+2?",
            tools=mock_tools,
            tool_manager=self.mock_tool_manager
        )

        # Should return direct response without tool execution
        self.assertEqual(result, "General knowledge response")
        self.mock_tool_manager.execute_tool.assert_not_called()

    def test_generate_response_with_tool_use(self):
        """Test response generation with tool execution"""
        # Setup mock tools
        mock_tools = [{"name": "search_course_content", "description": "Search courses"}]
        
        # Setup initial response with tool use
        tool_use_block = MockContentBlock(
            "tool_use",
            name="search_course_content",
            input={"query": "machine learning"},
            tool_use_id="tool_123"
        )
        initial_response = MockResponse(
            content=[tool_use_block],
            stop_reason="tool_use"
        )
        
        # Setup final response after tool execution
        final_content = [MockContentBlock("text", text="Based on course content, machine learning...")]
        final_response = MockResponse(content=final_content, stop_reason="end_turn")
        
        # Mock client to return responses in sequence
        self.mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Mock tool execution result
        self.mock_tool_manager.execute_tool.return_value = "Course content about ML algorithms"

        # Call generate_response
        result = self.ai_generator.generate_response(
            "Tell me about machine learning",
            tools=mock_tools,
            tool_manager=self.mock_tool_manager
        )

        # Assertions
        self.assertEqual(result, "Based on course content, machine learning...")
        self.mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning"
        )
        
        # Should have made 2 API calls (initial + follow-up)
        self.assertEqual(self.mock_client.messages.create.call_count, 2)

    def test_execute_tools_single_tool(self):
        """Test _execute_tools with single tool call"""
        # Setup response with tool use
        tool_use_block = MockContentBlock(
            "tool_use",
            name="get_course_outline",
            input={"course_name": "Python Basics"},
            tool_use_id="outline_tool_123"
        )
        response = MockResponse(
            content=[tool_use_block],
            stop_reason="tool_use"
        )
        
        # Mock tool execution
        self.mock_tool_manager.execute_tool.return_value = "Course: Python Basics\nLessons: 1-10"

        # Call _execute_tools
        result = self.ai_generator._execute_tools(response, self.mock_tool_manager)

        # Assertions
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "tool_result")
        self.assertEqual(result[0]["tool_use_id"], "outline_tool_123")
        self.assertEqual(result[0]["content"], "Course: Python Basics\nLessons: 1-10")
        
        self.mock_tool_manager.execute_tool.assert_called_once_with(
            "get_course_outline",
            course_name="Python Basics"
        )

    def test_execute_tools_multiple_tools(self):
        """Test _execute_tools with multiple tool calls"""
        # Setup response with multiple tool uses
        tool1 = MockContentBlock(
            "tool_use",
            name="search_course_content",
            input={"query": "variables"},
            tool_use_id="search_123"
        )
        tool2 = MockContentBlock(
            "tool_use", 
            name="get_course_outline",
            input={"course_name": "Python"},
            tool_use_id="outline_123"
        )
        response = MockResponse(
            content=[tool1, tool2],
            stop_reason="tool_use"
        )
        
        # Mock tool executions
        self.mock_tool_manager.execute_tool.side_effect = [
            "Variables are used to store data",
            "Course: Python\nLessons: 1-8"
        ]

        # Call _execute_tools
        result = self.ai_generator._execute_tools(response, self.mock_tool_manager)

        # Assertions
        self.assertEqual(len(result), 2)
        
        # Check first tool result
        self.assertEqual(result[0]["type"], "tool_result")
        self.assertEqual(result[0]["tool_use_id"], "search_123")
        self.assertEqual(result[0]["content"], "Variables are used to store data")
        
        # Check second tool result
        self.assertEqual(result[1]["type"], "tool_result")
        self.assertEqual(result[1]["tool_use_id"], "outline_123")
        self.assertEqual(result[1]["content"], "Course: Python\nLessons: 1-8")
        
        # Check both tool calls were made
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)
        calls = self.mock_tool_manager.execute_tool.call_args_list
        self.assertEqual(calls[0][0], ("search_course_content",))
        self.assertEqual(calls[0][1], {"query": "variables"})
        self.assertEqual(calls[1][0], ("get_course_outline",))
        self.assertEqual(calls[1][1], {"course_name": "Python"})

    def test_conversation_history_inclusion(self):
        """Test that conversation history is properly included in system prompt"""
        mock_content = [MockContentBlock("text", text="Response with history")]
        mock_response = MockResponse(content=mock_content, stop_reason="end_turn")
        self.mock_client.messages.create.return_value = mock_response

        history = "User: Hello\nAssistant: Hi there!"
        
        self.ai_generator.generate_response(
            "Follow up question",
            conversation_history=history
        )

        # Check that history was included in system prompt
        call_args = self.mock_client.messages.create.call_args
        system_content = call_args.kwargs["system"]
        self.assertIn("Previous conversation:", system_content)
        self.assertIn("User: Hello", system_content)
        self.assertIn("Assistant: Hi there!", system_content)

    def test_api_parameters_structure(self):
        """Test that API parameters are structured correctly"""
        mock_tools = [{"name": "test_tool", "description": "Test tool"}]
        mock_content = [MockContentBlock("text", text="Test response")]
        mock_response = MockResponse(content=mock_content, stop_reason="end_turn")
        self.mock_client.messages.create.return_value = mock_response

        self.ai_generator.generate_response(
            "Test query",
            tools=mock_tools
        )

        # Check API call parameters
        call_args = self.mock_client.messages.create.call_args
        params = call_args.kwargs
        
        # Should include all required parameters
        self.assertEqual(params["model"], self.model)
        self.assertEqual(params["temperature"], 0)
        self.assertEqual(params["max_tokens"], 800)
        self.assertIn("messages", params)
        self.assertIn("system", params)
        self.assertIn("tools", params)
        self.assertEqual(params["tool_choice"], {"type": "auto"})

    def test_execute_tools_error_handling(self):
        """Test handling of tool execution errors"""
        # Setup tool use response
        tool_use_block = MockContentBlock(
            "tool_use",
            name="search_course_content",
            input={"query": "test"},
            tool_use_id="error_test"
        )
        response = MockResponse(
            content=[tool_use_block],
            stop_reason="tool_use"
        )
        
        # Mock tool to raise exception
        self.mock_tool_manager.execute_tool.side_effect = Exception("Database unavailable")

        result = self.ai_generator._execute_tools(response, self.mock_tool_manager)

        # Should return error message in tool result format
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "tool_result")
        self.assertEqual(result[0]["tool_use_id"], "error_test")
        self.assertIn("Tool execution error", result[0]["content"])
        self.assertIn("Database unavailable", result[0]["content"])


class TestSequentialToolCalling(unittest.TestCase):
    """Test sequential tool calling functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test_api_key"
        self.model = "claude-3-sonnet-20241022"
        
        # Mock Anthropic client
        with patch('anthropic.Anthropic') as mock_anthropic:
            self.mock_client = MockAnthropicClient()
            mock_anthropic.return_value = self.mock_client
            self.ai_generator = AIGenerator(self.api_key, self.model)
            
        # Mock tool manager
        self.mock_tool_manager = Mock()

    def test_single_tool_call_backwards_compatibility(self):
        """Verify single tool calls work as before"""
        # Setup mock tools
        mock_tools = [{"name": "search_course_content", "description": "Search courses"}]
        
        # Setup single tool use scenario
        tool_use_block = MockContentBlock(
            "tool_use",
            name="search_course_content",
            input={"query": "python basics"},
            tool_use_id="tool_123"
        )
        tool_response = MockResponse(content=[tool_use_block], stop_reason="tool_use")
        
        # Setup final response
        final_content = [MockContentBlock("text", text="Python is a programming language")]
        final_response = MockResponse(content=final_content, stop_reason="end_turn")
        
        # Mock API calls: tool use + final response (2 calls total)
        self.mock_client.messages.create.side_effect = [tool_response, final_response]
        
        # Mock tool execution
        self.mock_tool_manager.execute_tool.return_value = "Course content about Python basics"

        # Call generate_response
        result = self.ai_generator.generate_response(
            "What is Python?",
            tools=mock_tools,
            tool_manager=self.mock_tool_manager
        )

        # Verify results
        self.assertEqual(result, "Python is a programming language")
        
        # Assert exactly 2 API calls made (tool use + final)
        self.assertEqual(self.mock_client.messages.create.call_count, 2)
        
        # Assert one tool executed
        self.mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="python basics"
        )

    def test_sequential_tool_calling_success(self):
        """Verify two-round sequential tool calling"""
        mock_tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]
        
        # Round 1: Get course outline
        tool1_block = MockContentBlock(
            "tool_use",
            name="get_course_outline",
            input={"course_name": "Python Course"},
            tool_use_id="tool1"
        )
        response1 = MockResponse(content=[tool1_block], stop_reason="tool_use")
        
        # Round 2: Search based on outline info  
        tool2_block = MockContentBlock(
            "tool_use",
            name="search_course_content",
            input={"query": "lesson 4 content"},
            tool_use_id="tool2"
        )
        response2 = MockResponse(content=[tool2_block], stop_reason="tool_use")
        
        # Final response
        final_content = [MockContentBlock("text", text="Lesson 4 covers functions and covers similar topics to Advanced Python")]
        final_response = MockResponse(content=final_content, stop_reason="end_turn")
        
        # Mock API sequence: round1 + round2 + final (3 calls total)
        self.mock_client.messages.create.side_effect = [response1, response2, final_response]
        
        # Mock tool executions
        self.mock_tool_manager.execute_tool.side_effect = [
            "Course: Python\nLesson 4: Functions and Scope",  # Outline result
            "Advanced Python course also covers functions"     # Search result
        ]

        # Call generate_response
        result = self.ai_generator.generate_response(
            "Find a course that discusses the same topic as lesson 4 of Python Course",
            tools=mock_tools,
            tool_manager=self.mock_tool_manager
        )

        # Verify final result
        self.assertEqual(result, "Lesson 4 covers functions and covers similar topics to Advanced Python")
        
        # Assert exactly 3 API calls made (2 tool rounds + final)
        self.assertEqual(self.mock_client.messages.create.call_count, 3)
        
        # Assert 2 tools executed in sequence
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)
        calls = self.mock_tool_manager.execute_tool.call_args_list
        self.assertEqual(calls[0][0], ("get_course_outline",))
        self.assertEqual(calls[1][0], ("search_course_content",))

    def test_max_rounds_termination(self):
        """Verify system stops after 2 rounds even if Claude wants more tools"""
        mock_tools = [{"name": "search_course_content"}]
        
        # Round 1: Tool use
        tool1_response = MockResponse(
            content=[MockContentBlock("tool_use", name="search_course_content", input={"query": "test1"})],
            stop_reason="tool_use"
        )
        
        # Round 2: Tool use  
        tool2_response = MockResponse(
            content=[MockContentBlock("tool_use", name="search_course_content", input={"query": "test2"})],
            stop_reason="tool_use"
        )
        
        # Final response (no tools available)
        final_response = MockResponse(
            content=[MockContentBlock("text", text="Final answer after 2 tool rounds")],
            stop_reason="end_turn"
        )
        
        # Mock API sequence: 2 tool rounds + forced final
        self.mock_client.messages.create.side_effect = [tool1_response, tool2_response, final_response]
        self.mock_tool_manager.execute_tool.return_value = "Tool result"

        # Call generate_response
        result = self.ai_generator.generate_response(
            "Complex query requiring multiple tools",
            tools=mock_tools,
            tool_manager=self.mock_tool_manager
        )

        # Verify termination
        self.assertEqual(result, "Final answer after 2 tool rounds")
        
        # Assert exactly 3 API calls (2 tool rounds + final without tools)
        self.assertEqual(self.mock_client.messages.create.call_count, 3)
        
        # Assert exactly 2 tool executions (max rounds enforced)
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)
        
        # Verify final call had no tools
        final_call_args = self.mock_client.messages.create.call_args_list[-1]
        self.assertNotIn("tools", final_call_args.kwargs)

    def test_tool_execution_error_handling(self):
        """Verify graceful error handling in sequential calls"""
        mock_tools = [{"name": "search_course_content"}]
        
        # Round 1: Tool use that will error
        tool_response = MockResponse(
            content=[MockContentBlock("tool_use", name="search_course_content", input={"query": "test"})],
            stop_reason="tool_use"
        )
        
        # Round 2: Continue after error
        continue_response = MockResponse(
            content=[MockContentBlock("text", text="I handled the tool error gracefully")],
            stop_reason="end_turn"
        )
        
        self.mock_client.messages.create.side_effect = [tool_response, continue_response]
        
        # Mock tool to raise exception
        self.mock_tool_manager.execute_tool.side_effect = Exception("Database unavailable")

        result = self.ai_generator.generate_response(
            "Test error handling",
            tools=mock_tools,
            tool_manager=self.mock_tool_manager
        )

        # Should continue and return response despite tool error
        self.assertEqual(result, "I handled the tool error gracefully")
        
        # Verify error was caught and passed to Claude
        self.assertEqual(self.mock_client.messages.create.call_count, 2)
        
        # Check that error message was passed to Claude in the conversation
        second_call_args = self.mock_client.messages.create.call_args_list[1]
        messages = second_call_args.kwargs["messages"]
        
        # Should have user message with tool error
        tool_result_message = messages[-1]
        self.assertEqual(tool_result_message["role"], "user")
        self.assertIn("Tool execution error", str(tool_result_message["content"]))

    def test_no_tool_use_immediate_response(self):
        """Verify immediate response when Claude doesn't use tools"""
        mock_tools = [{"name": "search_course_content"}]
        
        # Claude responds directly without tool use
        direct_response = MockResponse(
            content=[MockContentBlock("text", text="This is general knowledge, no tools needed")],
            stop_reason="end_turn"
        )
        
        self.mock_client.messages.create.return_value = direct_response

        result = self.ai_generator.generate_response(
            "What is 2+2?",
            tools=mock_tools,
            tool_manager=self.mock_tool_manager
        )

        # Should return immediate response
        self.assertEqual(result, "This is general knowledge, no tools needed")
        
        # Should make only 1 API call
        self.assertEqual(self.mock_client.messages.create.call_count, 1)
        
        # Should not execute any tools
        self.mock_tool_manager.execute_tool.assert_not_called()

    def test_message_accumulation_across_rounds(self):
        """Verify message history builds correctly across sequential rounds"""
        mock_tools = [{"name": "search_course_content"}]
        
        # Two sequential tool uses
        tool1_response = MockResponse(
            content=[MockContentBlock("tool_use", name="search_course_content", input={"query": "test1"})],
            stop_reason="tool_use"
        )
        tool2_response = MockResponse(
            content=[MockContentBlock("tool_use", name="search_course_content", input={"query": "test2"})],
            stop_reason="tool_use"
        )
        final_response = MockResponse(
            content=[MockContentBlock("text", text="Final synthesis")],
            stop_reason="end_turn"
        )
        
        self.mock_client.messages.create.side_effect = [tool1_response, tool2_response, final_response]
        self.mock_tool_manager.execute_tool.side_effect = ["Result1", "Result2"]

        result = self.ai_generator.generate_response(
            "Test message accumulation",
            tools=mock_tools,
            tool_manager=self.mock_tool_manager
        )

        # Verify result and API calls
        self.assertEqual(result, "Final synthesis")
        self.assertEqual(self.mock_client.messages.create.call_count, 3)
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)
        
        # Verify tools were called correctly
        tool_calls = self.mock_tool_manager.execute_tool.call_args_list
        self.assertEqual(tool_calls[0][1], {"query": "test1"})
        self.assertEqual(tool_calls[1][1], {"query": "test2"})
        
        # Note: Due to the current implementation using a single messages array,
        # we can't easily verify the intermediate message states, but we can verify
        # the overall behavior works correctly (3 API calls, 2 tool executions)


if __name__ == '__main__':
    print("Testing AIGenerator tool calling functionality...")
    unittest.main(verbosity=2)