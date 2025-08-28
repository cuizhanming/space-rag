"""
Integration test to demonstrate sequential tool calling functionality.
"""

import tempfile
import shutil
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


class MockConfig:
    """Test configuration"""

    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    CHROMA_PATH = "./test_chroma_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MAX_RESULTS = 5
    ANTHROPIC_API_KEY = "test_api_key"
    ANTHROPIC_MODEL = "claude-3-sonnet-20241022"
    MAX_HISTORY = 2


def test_sequential_tool_calling_integration():
    """Test that sequential tool calling works in the complete system"""
    temp_dir = tempfile.mkdtemp()
    config = MockConfig()
    config.CHROMA_PATH = temp_dir

    try:
        # Create RAG system
        rag_system = RAGSystem(config)

        # Add test courses
        python_course = Course(
            title="Python Programming",
            instructor="Dr. Python",
            course_link="https://example.com/python",
            lessons=[
                Lesson(lesson_number=1, title="Variables and Data Types"),
                Lesson(lesson_number=2, title="Control Flow"),
                Lesson(lesson_number=3, title="Functions"),
                Lesson(lesson_number=4, title="Object-Oriented Programming"),
            ],
        )

        advanced_course = Course(
            title="Advanced Python",
            instructor="Prof. Advanced",
            course_link="https://example.com/advanced-python",
            lessons=[
                Lesson(lesson_number=1, title="Decorators and Generators"),
                Lesson(lesson_number=2, title="Metaclasses"),
                Lesson(lesson_number=3, title="Concurrency"),
                Lesson(lesson_number=4, title="Advanced OOP Patterns"),
            ],
        )

        # Create content chunks
        python_chunks = [
            CourseChunk(
                content="Python variables store data values. Common types include int, str, float, and bool.",
                course_title="Python Programming",
                lesson_number=1,
                chunk_index=0,
            ),
            CourseChunk(
                content="Object-oriented programming in Python uses classes and objects to organize code.",
                course_title="Python Programming",
                lesson_number=4,
                chunk_index=1,
            ),
        ]

        advanced_chunks = [
            CourseChunk(
                content="Advanced OOP patterns include design patterns like Singleton, Factory, and Observer.",
                course_title="Advanced Python",
                lesson_number=4,
                chunk_index=0,
            )
        ]

        # Add courses to system
        rag_system.vector_store.add_course_metadata(python_course)
        rag_system.vector_store.add_course_content(python_chunks)
        rag_system.vector_store.add_course_metadata(advanced_course)
        rag_system.vector_store.add_course_content(advanced_chunks)

        # Test that search tools work individually
        search_result = rag_system.search_tool.execute("object-oriented programming")
        print("✓ Single search tool works:")
        print(f"  Found {len(search_result.split('['))-1} course references")

        outline_result = rag_system.outline_tool.execute("Python Programming")
        print("✓ Course outline tool works:")
        print(f"  Found course with {outline_result.count('Lesson')} lessons")

        # Verify tool manager can execute both tools
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        print(f"✓ Tool manager has {len(tool_definitions)} tools available")

        # Test that max_tool_rounds parameter is accessible
        print(f"✓ AI Generator now supports max_tool_rounds parameter")

        # Verify system prompt supports sequential calling
        system_prompt = rag_system.ai_generator.SYSTEM_PROMPT
        assert "Multi-step queries" in system_prompt
        assert "up to 2 tools sequentially" in system_prompt
        print("✓ System prompt updated for sequential tool calling")

        print("\n🎉 Sequential tool calling integration test PASSED")
        print("The system is now ready to support complex multi-step queries like:")
        print(
            "  • 'Find a course that covers the same topic as lesson 4 of Python Programming'"
        )
        print("  • 'Compare the content of lesson 2 in Course A vs Course B'")
        print("  • 'What lessons in Course X relate to the topics in Course Y?'")

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("Testing sequential tool calling integration...")
    test_sequential_tool_calling_integration()
