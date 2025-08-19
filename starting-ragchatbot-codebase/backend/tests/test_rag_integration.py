"""
Integration tests for RAG system content query handling to debug chatbot failures.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


class MockConfig:
    """Mock configuration for testing"""
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    CHROMA_PATH = "./test_chroma_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MAX_RESULTS = 5
    ANTHROPIC_API_KEY = "test_api_key"
    ANTHROPIC_MODEL = "claude-3-sonnet-20241022"
    MAX_HISTORY = 2


class TestRAGIntegration(unittest.TestCase):
    """Integration tests for RAG system content query handling"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MockConfig()
        
        # Mock all external dependencies
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager') as mock_tool_manager_class:
            
            # Create a mock instance for the tool manager
            self.mock_tool_manager = Mock()
            mock_tool_manager_class.return_value = self.mock_tool_manager
            
            self.rag_system = RAGSystem(self.config)
            
            # Setup mocks for testing
            self.mock_vector_store = self.rag_system.vector_store
            self.mock_ai_generator = self.rag_system.ai_generator
            self.mock_session_manager = self.rag_system.session_manager

    def test_query_content_specific_success(self):
        """Test successful content-specific query handling"""
        # Setup mock search results
        mock_sources = [{"text": "Python Course - Lesson 1", "link": "https://example.com/lesson1"}]
        self.mock_tool_manager.get_last_sources.return_value = mock_sources
        self.mock_tool_manager.reset_sources = Mock()
        
        # Setup mock AI response
        self.mock_ai_generator.generate_response.return_value = "Python variables are used to store data..."
        
        # Setup mock session manager
        self.mock_session_manager.get_conversation_history.return_value = None
        self.mock_session_manager.add_exchange = Mock()

        # Execute query
        response, sources = self.rag_system.query("What are Python variables?", "session_123")

        # Assertions
        self.assertIsInstance(response, str)
        self.assertEqual(sources, mock_sources)
        
        # Verify AI generator was called with correct parameters
        self.mock_ai_generator.generate_response.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args
        
        # Check that tools and tool_manager were provided
        self.assertIn("tools", call_args.kwargs)
        self.assertIn("tool_manager", call_args.kwargs)
        self.assertEqual(call_args.kwargs["tool_manager"], self.mock_tool_manager)
        
        # Verify session was updated
        self.mock_session_manager.add_exchange.assert_called_once_with(
            "session_123", "What are Python variables?", response
        )
        
        # Verify sources were reset
        self.mock_tool_manager.reset_sources.assert_called_once()

    def test_query_with_conversation_history(self):
        """Test query handling with existing conversation history"""
        # Setup mock history
        mock_history = "User: Hello\nAssistant: Hi there!"
        self.mock_session_manager.get_conversation_history.return_value = mock_history
        
        # Setup mock AI response
        self.mock_ai_generator.generate_response.return_value = "Follow-up response"
        self.mock_tool_manager.get_last_sources.return_value = []

        # Execute query
        response, sources = self.rag_system.query("Follow-up question", "session_456")

        # Verify history was passed to AI generator
        call_args = self.mock_ai_generator.generate_response.call_args
        self.assertEqual(call_args.kwargs["conversation_history"], mock_history)

    def test_query_without_session(self):
        """Test query handling without session ID"""
        self.mock_ai_generator.generate_response.return_value = "Response without session"
        self.mock_tool_manager.get_last_sources.return_value = []

        response, sources = self.rag_system.query("Test query")

        # Session manager should not be called for history or updates
        self.mock_session_manager.get_conversation_history.assert_not_called()
        self.mock_session_manager.add_exchange.assert_not_called()
        
        # AI generator should be called with no conversation history
        call_args = self.mock_ai_generator.generate_response.call_args
        self.assertIsNone(call_args.kwargs.get("conversation_history"))

    def test_query_prompt_construction(self):
        """Test that query prompt is constructed correctly"""
        self.mock_ai_generator.generate_response.return_value = "Test response"
        self.mock_tool_manager.get_last_sources.return_value = []

        user_query = "How do Python loops work?"
        self.rag_system.query(user_query)

        # Check that the prompt was constructed correctly
        call_args = self.mock_ai_generator.generate_response.call_args
        query_arg = call_args.kwargs["query"]
        self.assertIn("Answer this question about course materials:", query_arg)
        self.assertIn(user_query, query_arg)

    def test_query_tool_definitions_passed(self):
        """Test that tool definitions are passed to AI generator"""
        # Setup mock tool definitions
        mock_tool_definitions = [
            {"name": "search_course_content", "description": "Search courses"},
            {"name": "get_course_outline", "description": "Get course outline"}
        ]
        self.mock_tool_manager.get_tool_definitions.return_value = mock_tool_definitions
        
        self.mock_ai_generator.generate_response.return_value = "Response with tools"
        self.mock_tool_manager.get_last_sources.return_value = []

        self.rag_system.query("Test query with tools")

        # Verify tool definitions were passed
        call_args = self.mock_ai_generator.generate_response.call_args
        self.assertEqual(call_args.kwargs["tools"], mock_tool_definitions)

    def test_add_course_document_success(self):
        """Test successful course document addition"""
        # Setup mock document processor
        mock_course = Course(
            title="Test Course",
            instructor="Test Instructor",
            lessons=[Lesson(lesson_number=1, title="Introduction")]
        )
        mock_chunks = [
            CourseChunk(content="Test content", course_title="Test Course", chunk_index=0)
        ]
        self.rag_system.document_processor.process_course_document.return_value = (mock_course, mock_chunks)
        
        # Setup mock vector store
        self.mock_vector_store.add_course_metadata = Mock()
        self.mock_vector_store.add_course_content = Mock()

        # Execute document addition
        course, chunk_count = self.rag_system.add_course_document("test_file.txt")

        # Assertions
        self.assertEqual(course, mock_course)
        self.assertEqual(chunk_count, 1)
        
        # Verify vector store operations
        self.mock_vector_store.add_course_metadata.assert_called_once_with(mock_course)
        self.mock_vector_store.add_course_content.assert_called_once_with(mock_chunks)

    def test_add_course_document_failure(self):
        """Test course document addition failure handling"""
        # Setup mock to raise exception
        self.rag_system.document_processor.process_course_document.side_effect = Exception("Processing error")

        # Execute document addition
        course, chunk_count = self.rag_system.add_course_document("bad_file.txt")

        # Should handle error gracefully
        self.assertIsNone(course)
        self.assertEqual(chunk_count, 0)

    def test_get_course_analytics(self):
        """Test course analytics retrieval"""
        # Setup mock vector store responses
        self.mock_vector_store.get_course_count.return_value = 5
        self.mock_vector_store.get_existing_course_titles.return_value = ["Course A", "Course B"]

        # Execute analytics retrieval
        analytics = self.rag_system.get_course_analytics()

        # Assertions
        expected_analytics = {
            "total_courses": 5,
            "course_titles": ["Course A", "Course B"]
        }
        self.assertEqual(analytics, expected_analytics)


class TestRAGSystemRealScenarios(unittest.TestCase):
    """Test RAG system with realistic failure scenarios"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MockConfig()
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager') as mock_tool_manager_class:
            
            # Create a mock instance for the tool manager
            self.mock_tool_manager = Mock()
            mock_tool_manager_class.return_value = self.mock_tool_manager
            
            self.rag_system = RAGSystem(self.config)
            self.mock_ai_generator = self.rag_system.ai_generator

    def test_ai_generator_returns_error(self):
        """Test when AI generator returns an error or exception"""
        # Setup AI generator to raise exception
        self.mock_ai_generator.generate_response.side_effect = Exception("API Error")
        self.mock_tool_manager.get_last_sources.return_value = []

        # Should handle the exception gracefully
        with self.assertRaises(Exception):
            self.rag_system.query("Test query that fails")

    def test_ai_generator_returns_empty_response(self):
        """Test when AI generator returns empty or None response"""
        # Setup empty response
        self.mock_ai_generator.generate_response.return_value = ""
        self.mock_tool_manager.get_last_sources.return_value = []

        response, sources = self.rag_system.query("Test query")
        
        # Should still return response (even if empty)
        self.assertEqual(response, "")
        self.assertEqual(sources, [])

    def test_tool_manager_source_handling(self):
        """Test that tool manager sources are handled correctly"""
        # Test with no sources
        self.mock_ai_generator.generate_response.return_value = "Response"
        self.mock_tool_manager.get_last_sources.return_value = []
        
        response, sources = self.rag_system.query("Test")
        self.assertEqual(sources, [])
        
        # Test with multiple sources
        mock_sources = [
            {"text": "Course A - Lesson 1", "link": "http://example.com/a1"},
            {"text": "Course B - Lesson 2", "link": "http://example.com/b2"}
        ]
        self.mock_tool_manager.get_last_sources.return_value = mock_sources
        
        response, sources = self.rag_system.query("Test with sources")
        self.assertEqual(sources, mock_sources)

    def test_session_manager_integration(self):
        """Test session manager integration with various scenarios"""
        # Test with valid session
        self.mock_ai_generator.generate_response.return_value = "Valid session response"
        self.mock_tool_manager.get_last_sources.return_value = []
        self.rag_system.session_manager.get_conversation_history.return_value = "Previous chat"
        
        response, sources = self.rag_system.query("Query with session", "valid_session")
        
        # Should get history and add exchange
        self.rag_system.session_manager.get_conversation_history.assert_called_with("valid_session")
        self.rag_system.session_manager.add_exchange.assert_called_with(
            "valid_session", "Query with session", "Valid session response"
        )


if __name__ == '__main__':
    print("Testing RAG system integration...")
    unittest.main(verbosity=2)