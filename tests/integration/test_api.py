import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

# Mock the config module before importing app
sys.modules['config'] = Mock()
sys.modules['config'].config = Mock()
sys.modules['config'].config.CHUNK_SIZE = 1000
sys.modules['config'].config.CHUNK_OVERLAP = 200
sys.modules['config'].config.CHROMA_PATH = "/test/chroma"
sys.modules['config'].config.EMBEDDING_MODEL = "test-model"
sys.modules['config'].config.MAX_RESULTS = 5
sys.modules['config'].config.ANTHROPIC_API_KEY = "test-key"
sys.modules['config'].config.ANTHROPIC_MODEL = "claude-3-sonnet"
sys.modules['config'].config.MAX_HISTORY = 10

from app import app


class TestAPIEndpoints:
    """Integration tests for FastAPI endpoints"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    @patch('app.rag_system')
    def test_query_endpoint_new_session(self, mock_rag_system):
        """Test /api/query endpoint with new session"""
        # Mock RAG system responses
        mock_rag_system.query.return_value = ("Test answer", ["Source 1"])
        mock_rag_system.session_manager.create_session.return_value = "session_123"
        
        response = self.client.post(
            "/api/query",
            json={"query": "What is AI?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer"
        assert data["sources"] == ["Source 1"]
        assert data["session_id"] == "session_123"
        
        # Verify RAG system was called
        mock_rag_system.query.assert_called_once_with("What is AI?", "session_123")
        mock_rag_system.session_manager.create_session.assert_called_once()
    
    @patch('app.rag_system')
    def test_query_endpoint_existing_session(self, mock_rag_system):
        """Test /api/query endpoint with existing session"""
        mock_rag_system.query.return_value = ("Test answer with context", ["Source 1", "Source 2"])
        
        response = self.client.post(
            "/api/query",
            json={
                "query": "What is machine learning?",
                "session_id": "existing_session"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer with context"
        assert data["sources"] == ["Source 1", "Source 2"]
        assert data["session_id"] == "existing_session"
        
        # Verify session was not created
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("What is machine learning?", "existing_session")
    
    @patch('app.rag_system')
    def test_query_endpoint_error(self, mock_rag_system):
        """Test /api/query endpoint with error"""
        mock_rag_system.query.side_effect = Exception("RAG system error")
        mock_rag_system.session_manager.create_session.return_value = "session_123"
        
        response = self.client.post(
            "/api/query",
            json={"query": "Test query"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert data["detail"] == "RAG system error"
    
    @patch('app.rag_system')
    def test_courses_endpoint(self, mock_rag_system):
        """Test /api/courses endpoint"""
        # Mock analytics data
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["Course 1", "Course 2", "Course 3"]
        }
        
        response = self.client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Course 1" in data["course_titles"]
        
        # Verify method was called
        mock_rag_system.get_course_analytics.assert_called_once()
    
    @patch('app.rag_system')
    def test_courses_endpoint_error(self, mock_rag_system):
        """Test /api/courses endpoint with error"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Database error")
        
        response = self.client.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert data["detail"] == "Database error"
    
    def test_query_endpoint_validation_error(self):
        """Test /api/query endpoint with invalid request"""
        response = self.client.post(
            "/api/query",
            json={"invalid_field": "value"}  # Missing required 'query' field
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_query_endpoint_empty_query(self):
        """Test /api/query endpoint with empty query"""
        response = self.client.post(
            "/api/query",
            json={"query": ""}  # Empty query
        )
        
        # Should still work as empty string is valid
        assert response.status_code == 200
    
    @patch('app.rag_system')
    def test_query_unicode_content(self, mock_rag_system):
        """Test /api/query endpoint with unicode characters"""
        mock_rag_system.query.return_value = ("测试答案", ["来源 1"])
        mock_rag_system.session_manager.create_session.return_value = "session_123"
        
        response = self.client.post(
            "/api/query",
            json={"query": "什么是人工智能？"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "测试答案"
        assert data["sources"] == ["来源 1"]
    
    @patch('app.rag_system')
    def test_query_long_content(self, mock_rag_system):
        """Test /api/query endpoint with long content"""
        long_answer = "A" * 10000  # 10KB response
        long_sources = ["Source " + "B" * 1000] * 5  # 5 long source names
        
        mock_rag_system.query.return_value = (long_answer, long_sources)
        mock_rag_system.session_manager.create_session.return_value = "session_123"
        
        response = self.client.post(
            "/api/query",
            json={"query": "Give me a detailed explanation"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["answer"]) == 10000
        assert len(data["sources"]) == 5
    
    @patch('app.rag_system')
    def test_concurrent_queries(self, mock_rag_system):
        """Test handling concurrent queries"""
        import threading
        import time
        
        results = []
        
        def make_query(query_text):
            mock_rag_system.query.return_value = (f"Answer for {query_text}", [f"Source for {query_text}"])
            mock_rag_system.session_manager.create_session.return_value = f"session_{query_text}"
            
            response = self.client.post(
                "/api/query",
                json={"query": query_text}
            )
            results.append(response.status_code)
        
        # Create multiple threads for concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_query, args=[f"Query {i}"])
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5


class TestStaticFiles:
    """Test static file serving"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    @patch('fastapi.staticfiles.StaticFiles.get_response')
    def test_root_endpoint_serves_index(self, mock_get_response):
        """Test that root endpoint serves index.html"""
        from fastapi.responses import FileResponse
        
        # Mock the response
        mock_response = FileResponse("test.html")
        mock_get_response.return_value = mock_response
        
        response = self.client.get("/")
        
        # Should serve static files
        assert response.status_code == 200
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = self.client.options("/api/query")
        
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "*"
        assert "access-control-allow-methods" in response.headers


class TestStartupEvent:
    """Test application startup event"""
    
    @patch('app.os.path.exists')
    @patch('app.rag_system')
    def test_startup_loads_documents(self, mock_rag_system, mock_exists):
        """Test that startup event loads documents"""
        mock_exists.return_value = True
        mock_rag_system.add_course_folder.return_value = (2, 10)
        
        # Import and call startup event
        from app import startup_event
        
        # Call startup event
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(startup_event())
        loop.close()
        
        # Verify documents were loaded
        mock_exists.assert_called_once_with("../docs")
        mock_rag_system.add_course_folder.assert_called_once_with("../docs", clear_existing=False)
    
    @patch('app.os.path.exists')
    @patch('app.rag_system')
    def test_startup_docs_not_exist(self, mock_rag_system, mock_exists):
        """Test startup when docs directory doesn't exist"""
        mock_exists.return_value = False
        
        from app import startup_event
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(startup_event())
        loop.close()
        
        # Should not try to load documents
        mock_rag_system.add_course_folder.assert_not_called()
    
    @patch('app.os.path.exists')
    @patch('app.rag_system')
    @patch('builtins.print')
    def test_startup_error_handling(self, mock_print, mock_rag_system, mock_exists):
        """Test startup error handling"""
        mock_exists.return_value = True
        mock_rag_system.add_course_folder.side_effect = Exception("Load error")
        
        from app import startup_event
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(startup_event())
        loop.close()
        
        # Should print error message
        print_calls = [call for call in mock_print.call_args_list if "Error loading documents" in str(call)]
        assert len(print_calls) > 0