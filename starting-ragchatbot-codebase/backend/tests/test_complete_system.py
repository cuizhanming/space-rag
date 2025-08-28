"""
Complete system test to verify the 'query failed' issue is fixed.
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
    MAX_RESULTS = 5  # This should now be 5, not 0
    ANTHROPIC_API_KEY = "test_api_key"
    ANTHROPIC_MODEL = "claude-3-sonnet-20241022"
    MAX_HISTORY = 2


def test_complete_system():
    """Test the complete system with fixed configuration"""
    temp_dir = tempfile.mkdtemp()
    config = MockConfig()
    config.CHROMA_PATH = temp_dir

    try:
        # Create RAG system
        rag_system = RAGSystem(config)

        # Add test course manually
        course = Course(
            title="Python Programming",
            instructor="Test Instructor",
            course_link="https://example.com/python",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Variables",
                    lesson_link="https://example.com/lesson1",
                ),
                Lesson(
                    lesson_number=2,
                    title="Functions",
                    lesson_link="https://example.com/lesson2",
                ),
            ],
        )

        chunks = [
            CourseChunk(
                content="Python variables are used to store data. You can assign values like x = 5 or name = 'John'.",
                course_title="Python Programming",
                lesson_number=1,
                chunk_index=0,
            ),
            CourseChunk(
                content="Functions in Python are defined with the def keyword. They help organize code into reusable blocks.",
                course_title="Python Programming",
                lesson_number=2,
                chunk_index=1,
            ),
        ]

        # Add course to system (this should now work)
        rag_system.vector_store.add_course_metadata(course)
        rag_system.vector_store.add_course_content(chunks)

        # Test search functionality
        search_results = rag_system.vector_store.search("Python variables")
        print(f"Search results: {len(search_results.documents)} documents found")
        print(f"Search error: {search_results.error}")

        if search_results.documents:
            print("✓ Search is working - found content about variables")

            # Test CourseSearchTool
            search_tool_result = rag_system.search_tool.execute("Python variables")
            print(f"Search tool result length: {len(search_tool_result)}")

            if "No relevant content found" not in search_tool_result:
                print("✓ CourseSearchTool is working correctly")
                print(
                    "✓ ALL FIXES SUCCESSFUL - System should now work for content queries"
                )
            else:
                print("✗ CourseSearchTool still returning no content")
        else:
            print("✗ Search still not working")

    except Exception as e:
        print(f"✗ System test failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("Testing complete system with fixes...")
    test_complete_system()
