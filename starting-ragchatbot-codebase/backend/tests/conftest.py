"""
Pytest configuration and shared fixtures for RAG system tests.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, MagicMock
from pathlib import Path

# Test configuration
os.environ["ANTHROPIC_API_KEY"] = "test-key"


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for AI generation tests"""
    mock_client = Mock()
    mock_message = Mock()
    mock_message.content = [Mock(text="Test AI response")]
    mock_message.stop_reason = "end_turn"
    mock_client.messages.create = Mock(return_value=mock_message)
    return mock_client


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing without ChromaDB"""
    mock_store = Mock()
    mock_store.add_course_chunks = Mock(return_value=None)
    mock_store.search_similar = Mock(
        return_value=[
            {
                "content": "Test content 1",
                "course_title": "Test Course",
                "lesson_title": "Test Lesson",
                "course_link": "https://example.com/course",
                "lesson_link": "https://example.com/lesson",
            }
        ]
    )
    mock_store.get_all_courses = Mock(
        return_value=[
            {
                "title": "Test Course 1",
                "instructor": "Test Instructor",
                "link": "https://example.com/course1",
            },
            {
                "title": "Test Course 2",
                "instructor": "Test Instructor",
                "link": "https://example.com/course2",
            },
        ]
    )
    return mock_store


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    mock_manager = Mock()
    mock_manager.create_session = Mock(return_value="test-session-id")
    mock_manager.add_exchange = Mock()
    mock_manager.get_conversation_history = Mock(
        return_value="User: Hello\nAssistant: Hi there!"
    )
    return mock_manager


@pytest.fixture
def sample_course_data():
    """Sample course data for testing"""
    return {
        "title": "Introduction to Python",
        "instructor": "Jane Doe",
        "link": "https://example.com/python-course",
        "lessons": [
            {
                "title": "Variables and Data Types",
                "content": "In Python, variables are used to store data. Common data types include strings, integers, and floats.",
                "link": "https://example.com/python-course/lesson1",
            },
            {
                "title": "Control Structures",
                "content": "Control structures like if statements and loops allow you to control program flow.",
                "link": "https://example.com/python-course/lesson2",
            },
        ],
    }


@pytest.fixture
def temp_docs_directory():
    """Create temporary directory with sample course documents"""
    temp_dir = tempfile.mkdtemp()

    # Create sample course document
    course_content = """Course Title: Introduction to Python
Course Link: https://example.com/python-course
Course Instructor: Jane Doe

Lesson 0: Variables and Data Types
Lesson Link: https://example.com/python-course/lesson1

In Python, variables are used to store data. You can create variables by simply assigning values to names.

Common data types in Python include:
- Strings: text data enclosed in quotes
- Integers: whole numbers
- Floats: decimal numbers
- Booleans: True or False values

Lesson 1: Control Structures
Lesson Link: https://example.com/python-course/lesson2

Control structures allow you to control the flow of your program.

If statements let you execute code conditionally:
if condition:
    # do something

For loops let you repeat code:
for item in items:
    # process item
"""

    with open(os.path.join(temp_dir, "python_course.txt"), "w") as f:
        f.write(course_content)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = Mock()
    config.anthropic_api_key = "test-key"
    config.model_name = "claude-sonnet-4-20250514"
    config.max_tokens = 800
    config.temperature = 0
    config.embedding_model = "all-MiniLM-L6-v2"
    config.chunk_size = 800
    config.chunk_overlap = 100
    config.max_results = 5
    config.max_conversation_history = 2
    return config


@pytest.fixture
def mock_rag_system(mock_vector_store, mock_session_manager, mock_config):
    """Mock RAG system for API testing"""
    mock_system = Mock()
    mock_system.query = AsyncMock(
        return_value=(
            "This is a test response about Python variables.",
            [
                {
                    "text": "Variables and Data Types",
                    "link": "https://example.com/lesson1",
                }
            ],
        )
    )
    mock_system.get_course_analytics = Mock(
        return_value={
            "total_courses": 2,
            "course_titles": ["Introduction to Python", "Advanced Python"],
        }
    )
    mock_system.add_course_folder = Mock(return_value=(2, 10))
    mock_system.session_manager = mock_session_manager
    return mock_system


@pytest.fixture
def test_query_request():
    """Sample query request for API testing"""
    return {"query": "What are Python variables?", "session_id": "test-session-123"}


@pytest.fixture
def expected_query_response():
    """Expected query response structure for API testing"""
    return {
        "answer": "This is a test response about Python variables.",
        "sources": [
            {"text": "Variables and Data Types", "link": "https://example.com/lesson1"}
        ],
        "session_id": "test-session-123",
    }


@pytest.fixture
def expected_course_stats():
    """Expected course statistics response for API testing"""
    return {
        "total_courses": 2,
        "course_titles": ["Introduction to Python", "Advanced Python"],
    }


@pytest.fixture(scope="function")
def cleanup_chroma_db():
    """Ensure ChromaDB is cleaned up between tests"""
    yield
    # Clean up any test ChromaDB instances
    chroma_path = "./test_chroma_db"
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
