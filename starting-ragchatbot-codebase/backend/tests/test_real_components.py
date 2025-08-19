"""
Real component tests to identify actual system failures.
These tests use actual instances to debug the 'query failed' issue.
"""
import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import patch

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, ToolManager
from models import Course, Lesson, CourseChunk


class TestRealVectorStore(unittest.TestCase):
    """Test actual VectorStore functionality"""

    def setUp(self):
        """Set up test fixtures with temporary database"""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore(
            chroma_path=self.temp_dir,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_vector_store_initialization(self):
        """Test that VectorStore initializes correctly"""
        self.assertIsNotNone(self.vector_store)
        self.assertIsNotNone(self.vector_store.client)
        self.assertIsNotNone(self.vector_store.course_catalog)
        self.assertIsNotNone(self.vector_store.course_content)

    def test_add_and_search_course_content(self):
        """Test adding course content and searching"""
        # Create test course data
        course = Course(
            title="Test Python Course",
            instructor="Test Instructor", 
            course_link="https://example.com/python",
            lessons=[Lesson(lesson_number=1, title="Variables", lesson_link="https://example.com/lesson1")]
        )
        
        # Create test chunks
        chunks = [
            CourseChunk(
                content="Python variables are used to store data values. Variables are created when you assign a value to them.",
                course_title="Test Python Course",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="In Python you can create variables like x = 5 or name = 'John'. Python has different data types.",
                course_title="Test Python Course", 
                lesson_number=1,
                chunk_index=1
            )
        ]

        # Add course metadata and content
        self.vector_store.add_course_metadata(course)
        self.vector_store.add_course_content(chunks)

        # Test search functionality
        results = self.vector_store.search("Python variables")
        
        # Verify search results
        self.assertIsInstance(results, SearchResults)
        self.assertIsNone(results.error, f"Search returned error: {results.error}")
        self.assertFalse(results.is_empty(), "Search should return results")
        self.assertTrue(len(results.documents) > 0, "Should have document results")
        
        # Check if content is found
        found_variable_content = any("variables" in doc.lower() for doc in results.documents)
        self.assertTrue(found_variable_content, "Should find content about variables")

    def test_course_name_resolution(self):
        """Test course name resolution functionality"""
        # Add a course
        course = Course(title="Machine Learning Fundamentals", instructor="Dr. Smith")
        self.vector_store.add_course_metadata(course)
        
        # Test resolving with exact match
        resolved = self.vector_store._resolve_course_name("Machine Learning Fundamentals")
        self.assertEqual(resolved, "Machine Learning Fundamentals")
        
        # Test resolving with partial match
        resolved_partial = self.vector_store._resolve_course_name("Machine Learning")
        self.assertEqual(resolved_partial, "Machine Learning Fundamentals")

    def test_search_with_filters(self):
        """Test search with course and lesson filters"""
        # Add test data with multiple courses and lessons
        course1 = Course(title="Python Basics", lessons=[Lesson(lesson_number=1, title="Intro")])
        course2 = Course(title="Advanced Python", lessons=[Lesson(lesson_number=1, title="Classes")])
        
        chunks1 = [CourseChunk(content="Python basic concepts", course_title="Python Basics", lesson_number=1, chunk_index=0)]
        chunks2 = [CourseChunk(content="Python advanced features", course_title="Advanced Python", lesson_number=1, chunk_index=0)]
        
        self.vector_store.add_course_metadata(course1)
        self.vector_store.add_course_metadata(course2)
        self.vector_store.add_course_content(chunks1)
        self.vector_store.add_course_content(chunks2)
        
        # Search with course filter
        results = self.vector_store.search("Python", course_name="Python Basics")
        self.assertFalse(results.is_empty(), "Should find results with course filter")
        
        # Verify results are from correct course
        if results.metadata:
            self.assertEqual(results.metadata[0]['course_title'], "Python Basics")


class TestRealCourseSearchTool(unittest.TestCase):
    """Test actual CourseSearchTool functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore(
            chroma_path=self.temp_dir,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )
        self.search_tool = CourseSearchTool(self.vector_store)

    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_search_tool_with_real_data(self):
        """Test CourseSearchTool with real data"""
        # Add test course data
        course = Course(
            title="Data Science 101",
            instructor="Prof. Data",
            lessons=[
                Lesson(lesson_number=1, title="Introduction to Data Science"),
                Lesson(lesson_number=2, title="Data Analysis with Python")
            ]
        )
        
        chunks = [
            CourseChunk(
                content="Data science is an interdisciplinary field that uses algorithms and statistics to extract insights from data.",
                course_title="Data Science 101",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Python is a popular programming language for data analysis because of libraries like pandas and numpy.",
                course_title="Data Science 101", 
                lesson_number=2,
                chunk_index=1
            )
        ]

        self.vector_store.add_course_metadata(course)
        self.vector_store.add_course_content(chunks)

        # Test search tool execution
        result = self.search_tool.execute("data science")
        
        # Verify results
        self.assertIsInstance(result, str)
        self.assertNotIn("No relevant content found", result)
        self.assertIn("Data Science 101", result)
        
        # Verify sources were tracked
        self.assertTrue(len(self.search_tool.last_sources) > 0)

    def test_search_tool_empty_database(self):
        """Test search tool behavior with empty database"""
        result = self.search_tool.execute("nonexistent query")
        
        self.assertIn("No relevant content found", result)
        self.assertEqual(len(self.search_tool.last_sources), 0)


class TestRealSystemIntegration(unittest.TestCase):
    """Test system integration without mocking"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create real vector store and tool manager
        self.vector_store = VectorStore(
            chroma_path=self.temp_dir,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )
        
        self.tool_manager = ToolManager()
        self.search_tool = CourseSearchTool(self.vector_store)
        self.tool_manager.register_tool(self.search_tool)

    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tool_manager_functionality(self):
        """Test ToolManager with real tools"""
        # Add test data
        course = Course(title="Test Course")
        chunks = [
            CourseChunk(
                content="This is test content about programming concepts.",
                course_title="Test Course",
                chunk_index=0
            )
        ]
        
        self.vector_store.add_course_metadata(course)
        self.vector_store.add_course_content(chunks)
        
        # Test tool definitions
        tool_defs = self.tool_manager.get_tool_definitions()
        self.assertTrue(len(tool_defs) > 0)
        self.assertEqual(tool_defs[0]["name"], "search_course_content")
        
        # Test tool execution
        result = self.tool_manager.execute_tool("search_course_content", query="programming")
        self.assertIsInstance(result, str)
        self.assertNotIn("Tool 'search_course_content' not found", result)
        
        # Test sources
        sources = self.tool_manager.get_last_sources()
        self.assertIsInstance(sources, list)

    def test_search_with_anthropic_api_key_missing(self):
        """Test system behavior when API key is missing"""
        # This test checks if the search components work independent of AI
        course = Course(title="API Test Course")
        chunks = [CourseChunk(content="API testing content", course_title="API Test Course", chunk_index=0)]
        
        self.vector_store.add_course_metadata(course)
        self.vector_store.add_course_content(chunks)
        
        # Search should work even without API key
        result = self.search_tool.execute("API testing")
        self.assertNotIn("No relevant content found", result)
        self.assertIn("API Test Course", result)


class TestSystemDiagnostics(unittest.TestCase):
    """Diagnostic tests to identify specific failure points"""

    def setUp(self):
        """Set up diagnostics"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_chromadb_functionality(self):
        """Test if ChromaDB is working correctly"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Test ChromaDB creation
            client = chromadb.PersistentClient(
                path=self.temp_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Test embedding function
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            
            # Test collection creation
            collection = client.get_or_create_collection(
                name="test_collection",
                embedding_function=embedding_function
            )
            
            # Test basic operations
            collection.add(
                documents=["This is a test document"],
                metadatas=[{"test": "metadata"}],
                ids=["test_id"]
            )
            
            results = collection.query(query_texts=["test"], n_results=1)
            self.assertTrue(len(results['documents'][0]) > 0)
            
        except Exception as e:
            self.fail(f"ChromaDB functionality test failed: {e}")

    def test_sentence_transformers(self):
        """Test if sentence transformers are working"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(["test sentence"])
            self.assertTrue(len(embeddings) > 0)
            self.assertTrue(len(embeddings[0]) > 0)
        except Exception as e:
            self.fail(f"Sentence transformers test failed: {e}")


if __name__ == '__main__':
    print("Testing real system components...")
    unittest.main(verbosity=2)