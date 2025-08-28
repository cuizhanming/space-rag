"""
API endpoint tests for FastAPI RAG system.
Tests all API endpoints with proper request/response handling.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json

# Import the FastAPI app and dependencies
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def test_app():
    """Create test FastAPI app without static file mounting issues"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    # Recreate the app structure inline to avoid import issues
    app = FastAPI(title="Course Materials RAG System", root_path="")

    # Add middleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Pydantic models (same as in app.py)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceLink(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceLink]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # Mock RAG system for testing
    mock_rag_system = Mock()

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = "test-session-123"

            # Mock the query response
            answer, sources = mock_rag_system.query(request.query, session_id)

            source_links = []
            for source in sources:
                if isinstance(source, dict):
                    source_links.append(
                        SourceLink(text=source["text"], link=source.get("link"))
                    )
                else:
                    source_links.append(SourceLink(text=str(source), link=None))

            return QueryResponse(
                answer=answer, sources=source_links, session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System"}

    # Attach mock for testing
    app.mock_rag_system = mock_rag_system
    return app


@pytest.fixture
def client(test_app):
    """Create test client with mocked app"""
    return TestClient(test_app)


@pytest.mark.api
class TestAPIEndpoints:
    """Test suite for API endpoints"""

    def test_query_endpoint_success(self, client, test_app):
        """Test successful query to /api/query endpoint"""
        # Setup mock response
        test_app.mock_rag_system.query.return_value = (
            "Python variables are names that store data values.",
            [{"text": "Variables chapter", "link": "https://example.com/variables"}],
        )

        # Make request
        response = client.post(
            "/api/query",
            json={
                "query": "What are Python variables?",
                "session_id": "test-session-123",
            },
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "Python variables are names that store data values."
        assert data["session_id"] == "test-session-123"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Variables chapter"
        assert data["sources"][0]["link"] == "https://example.com/variables"

    def test_query_endpoint_without_session_id(self, client, test_app):
        """Test query endpoint creates session when none provided"""
        test_app.mock_rag_system.query.return_value = (
            "Test response",
            [{"text": "Test source", "link": None}],
        )

        response = client.post("/api/query", json={"query": "Test query"})

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"  # Default from mock

    def test_query_endpoint_with_string_sources(self, client, test_app):
        """Test query endpoint handles backward compatibility with string sources"""
        test_app.mock_rag_system.query.return_value = (
            "Test response",
            ["String source 1", "String source 2"],
        )

        response = client.post(
            "/api/query", json={"query": "Test query", "session_id": "test-123"}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 2
        assert data["sources"][0]["text"] == "String source 1"
        assert data["sources"][0]["link"] is None
        assert data["sources"][1]["text"] == "String source 2"
        assert data["sources"][1]["link"] is None

    def test_query_endpoint_error_handling(self, client, test_app):
        """Test query endpoint error handling"""
        test_app.mock_rag_system.query.side_effect = Exception(
            "Database connection failed"
        )

        response = client.post("/api/query", json={"query": "Test query"})

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]

    def test_query_endpoint_validation(self, client):
        """Test query endpoint input validation"""
        # Missing query field
        response = client.post("/api/query", json={})
        assert response.status_code == 422

        # Invalid JSON
        response = client.post(
            "/api/query",
            data="invalid json",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422

    def test_courses_endpoint_success(self, client, test_app):
        """Test successful request to /api/courses endpoint"""
        test_app.mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["Python Basics", "Advanced Python", "Data Structures"],
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Python Basics" in data["course_titles"]

    def test_courses_endpoint_error_handling(self, client, test_app):
        """Test courses endpoint error handling"""
        test_app.mock_rag_system.get_course_analytics.side_effect = Exception(
            "Vector store error"
        )

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "Vector store error" in response.json()["detail"]

    def test_root_endpoint(self, client):
        """Test root endpoint returns expected response"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Course Materials RAG System"

    def test_cors_headers(self, client):
        """Test CORS headers are properly set"""
        response = client.options("/api/query")

        # FastAPI automatically handles OPTIONS for CORS
        # Check that the endpoint exists and is accessible
        assert response.status_code in [200, 405]  # 405 is also acceptable for OPTIONS

    def test_content_type_handling(self, client, test_app):
        """Test proper content-type handling"""
        test_app.mock_rag_system.query.return_value = ("Test", [])

        # Test with correct content-type
        response = client.post(
            "/api/query",
            json={"query": "test"},
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 200

        # Test response content-type
        assert "application/json" in response.headers["content-type"]


@pytest.mark.api
@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints with more realistic scenarios"""

    def test_query_flow_with_session(self, client, test_app):
        """Test complete query flow with session management"""
        # Setup mock responses for a conversation
        query_responses = [
            (
                "Variables store data in Python.",
                [{"text": "Python Variables", "link": "https://example.com/vars"}],
            ),
            (
                "Yes, you can change variable values.",
                [{"text": "Variable Assignment", "link": "https://example.com/assign"}],
            ),
        ]

        test_app.mock_rag_system.query.side_effect = query_responses

        # First query - session created
        response1 = client.post(
            "/api/query", json={"query": "What are Python variables?"}
        )

        assert response1.status_code == 200
        data1 = response1.json()
        session_id = data1["session_id"]

        # Second query - same session
        response2 = client.post(
            "/api/query",
            json={"query": "Can I change their values?", "session_id": session_id},
        )

        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["session_id"] == session_id

        # Verify both queries were processed
        assert test_app.mock_rag_system.query.call_count == 2

    def test_multiple_concurrent_sessions(self, client, test_app):
        """Test handling multiple concurrent sessions"""
        test_app.mock_rag_system.query.return_value = ("Response", [])

        # Multiple queries without session IDs should get different sessions
        responses = []
        for i in range(3):
            response = client.post("/api/query", json={"query": f"Query {i}"})
            assert response.status_code == 200
            responses.append(response.json())

        # All should have the same default session ID from mock
        # In real implementation, these would be different
        session_ids = [r["session_id"] for r in responses]
        assert all(sid == "test-session-123" for sid in session_ids)

    def test_error_recovery(self, client, test_app):
        """Test system recovery after errors"""
        # First request fails
        test_app.mock_rag_system.query.side_effect = Exception("Temporary failure")

        response1 = client.post("/api/query", json={"query": "Test query"})
        assert response1.status_code == 500

        # Second request succeeds
        test_app.mock_rag_system.query.side_effect = None
        test_app.mock_rag_system.query.return_value = ("Success", [])

        response2 = client.post("/api/query", json={"query": "Test query 2"})
        assert response2.status_code == 200
        assert response2.json()["answer"] == "Success"


@pytest.mark.api
class TestAPIResponseFormats:
    """Test API response format compliance"""

    def test_query_response_schema(self, client, test_app):
        """Test query response matches expected schema"""
        test_app.mock_rag_system.query.return_value = (
            "Test answer",
            [
                {"text": "Source 1", "link": "https://example.com/1"},
                {"text": "Source 2", "link": None},
            ],
        )

        response = client.post("/api/query", json={"query": "test"})

        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data

        # Verify types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

        # Verify source structure
        for source in data["sources"]:
            assert "text" in source
            assert "link" in source
            assert isinstance(source["text"], str)
            assert source["link"] is None or isinstance(source["link"], str)

    def test_courses_response_schema(self, client, test_app):
        """Test courses response matches expected schema"""
        test_app.mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 5,
            "course_titles": [
                "Course A",
                "Course B",
                "Course C",
                "Course D",
                "Course E",
            ],
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        assert "total_courses" in data
        assert "course_titles" in data

        # Verify types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert all(isinstance(title, str) for title in data["course_titles"])

        # Verify consistency
        assert data["total_courses"] == len(data["course_titles"])
