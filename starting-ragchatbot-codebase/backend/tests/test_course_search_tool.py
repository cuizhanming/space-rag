"""
Tests for CourseSearchTool.execute method to debug RAG chatbot failures.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool(unittest.TestCase):
    """Test CourseSearchTool execute method functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_execute_basic_search_success(self):
        """Test basic search execution with successful results"""
        # Setup mock search results
        mock_results = SearchResults(
            documents=["Course content about machine learning"],
            metadata=[{"course_title": "Introduction to AI", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/lesson1"
        )

        # Execute search
        result = self.search_tool.execute("machine learning")

        # Assertions
        self.assertIsInstance(result, str)
        self.assertIn("Introduction to AI", result)
        self.assertIn("Lesson 1", result)
        self.assertIn("machine learning", result)

        # Verify vector store was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="machine learning", course_name=None, lesson_number=None
        )

    def test_execute_with_course_filter(self):
        """Test search execution with course name filter"""
        mock_results = SearchResults(
            documents=["Specific course content"],
            metadata=[{"course_title": "Advanced Python", "lesson_number": 2}],
            distances=[0.2],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        result = self.search_tool.execute(
            "python functions", course_name="Advanced Python"
        )

        # Verify search was called with correct parameters
        self.mock_vector_store.search.assert_called_once_with(
            query="python functions", course_name="Advanced Python", lesson_number=None
        )
        self.assertIn("Advanced Python", result)

    def test_execute_with_lesson_filter(self):
        """Test search execution with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson specific content"],
            metadata=[{"course_title": "Data Science 101", "lesson_number": 3}],
            distances=[0.15],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/lesson3"
        )

        result = self.search_tool.execute("data analysis", lesson_number=3)

        # Verify search was called with lesson filter
        self.mock_vector_store.search.assert_called_once_with(
            query="data analysis", course_name=None, lesson_number=3
        )
        self.assertIn("Lesson 3", result)

    def test_execute_with_both_filters(self):
        """Test search execution with both course and lesson filters"""
        mock_results = SearchResults(
            documents=["Very specific content"],
            metadata=[{"course_title": "Machine Learning Course", "lesson_number": 5}],
            distances=[0.05],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/ml/lesson5"
        )

        result = self.search_tool.execute(
            "neural networks", course_name="Machine Learning Course", lesson_number=5
        )

        # Verify search was called with both filters
        self.mock_vector_store.search.assert_called_once_with(
            query="neural networks",
            course_name="Machine Learning Course",
            lesson_number=5,
        )
        self.assertIn("Machine Learning Course", result)
        self.assertIn("Lesson 5", result)

    def test_execute_handles_search_error(self):
        """Test that execute method handles search errors properly"""
        # Setup mock to return error
        error_results = SearchResults(
            documents=[], metadata=[], distances=[], error="Database connection failed"
        )
        self.mock_vector_store.search.return_value = error_results

        result = self.search_tool.execute("test query")

        # Should return the error message
        self.assertEqual(result, "Database connection failed")

    def test_execute_handles_empty_results(self):
        """Test that execute method handles empty results properly"""
        # Setup mock to return empty results
        empty_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        self.mock_vector_store.search.return_value = empty_results

        result = self.search_tool.execute("nonexistent content")

        # Should return appropriate no results message
        self.assertIn("No relevant content found", result)

    def test_execute_empty_results_with_filters(self):
        """Test empty results message includes filter information"""
        empty_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        self.mock_vector_store.search.return_value = empty_results

        result = self.search_tool.execute(
            "test", course_name="Nonexistent Course", lesson_number=99
        )

        self.assertIn("No relevant content found", result)
        self.assertIn("Nonexistent Course", result)
        self.assertIn("lesson 99", result)

    def test_format_results_with_lesson_links(self):
        """Test that results are formatted correctly with lesson links"""
        mock_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/courseA/lesson1",
            "https://example.com/courseB/lesson2",
        ]

        result = self.search_tool.execute("test query")

        # Check that sources were stored correctly
        self.assertEqual(len(self.search_tool.last_sources), 2)
        self.assertEqual(
            self.search_tool.last_sources[0]["text"], "Course A - Lesson 1"
        )
        self.assertEqual(
            self.search_tool.last_sources[0]["link"],
            "https://example.com/courseA/lesson1",
        )

    def test_format_results_without_lesson_numbers(self):
        """Test formatting when metadata lacks lesson numbers"""
        mock_results = SearchResults(
            documents=["General course content"],
            metadata=[{"course_title": "Course C"}],  # No lesson_number
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("general query")

        self.assertIn("[Course C]", result)
        self.assertNotIn("Lesson", result)

        # Check sources without lesson info
        self.assertEqual(len(self.search_tool.last_sources), 1)
        self.assertEqual(self.search_tool.last_sources[0]["text"], "Course C")

    def test_vector_store_search_call_parameters(self):
        """Test that vector store search is called with correct parameters"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        self.mock_vector_store.search.return_value = mock_results

        # Test all parameter combinations
        test_cases = [
            ("query1", None, None),
            ("query2", "Course Name", None),
            ("query3", None, 5),
            ("query4", "Another Course", 10),
        ]

        for query, course, lesson in test_cases:
            self.mock_vector_store.search.reset_mock()
            self.search_tool.execute(query, course_name=course, lesson_number=lesson)

            self.mock_vector_store.search.assert_called_once_with(
                query=query, course_name=course, lesson_number=lesson
            )


if __name__ == "__main__":
    print("Testing CourseSearchTool.execute method...")
    unittest.main(verbosity=2)
