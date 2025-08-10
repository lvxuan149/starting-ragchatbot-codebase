import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


class TestRAGSystem:
    """Test cases for RAGSystem class"""
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_init(self, mock_search_tool, mock_tool_manager, mock_session_manager, 
                  mock_ai_generator, mock_vector_store, mock_document_processor):
        """Test RAGSystem initialization"""
        # Mock config
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 1000
        mock_config.CHUNK_OVERLAP = 200
        mock_config.CHROMA_PATH = "/test/chroma"
        mock_config.EMBEDDING_MODEL = "test-model"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "test-key"
        mock_config.ANTHROPIC_MODEL = "claude-3-sonnet"
        mock_config.MAX_HISTORY = 10
        
        # Create instances
        mock_doc_processor_instance = Mock()
        mock_vector_store_instance = Mock()
        mock_ai_generator_instance = Mock()
        mock_session_manager_instance = Mock()
        mock_tool_manager_instance = Mock()
        mock_search_tool_instance = Mock()
        
        mock_document_processor.return_value = mock_doc_processor_instance
        mock_vector_store.return_value = mock_vector_store_instance
        mock_ai_generator.return_value = mock_ai_generator_instance
        mock_session_manager.return_value = mock_session_manager_instance
        mock_tool_manager.return_value = mock_tool_manager_instance
        mock_search_tool.return_value = mock_search_tool_instance
        
        rag_system = RAGSystem(mock_config)
        
        # Verify all components are initialized
        assert rag_system.document_processor == mock_doc_processor_instance
        assert rag_system.vector_store == mock_vector_store_instance
        assert rag_system.ai_generator == mock_ai_generator_instance
        assert rag_system.session_manager == mock_session_manager_instance
        assert rag_system.tool_manager == mock_tool_manager_instance
        assert rag_system.search_tool == mock_search_tool_instance
        
        # Verify tool registration
        mock_tool_manager_instance.register_tool.assert_called_once_with(mock_search_tool_instance)
        
        # Verify initialization calls
        mock_document_processor.assert_called_once_with(1000, 200)
        mock_vector_store.assert_called_once_with("/test/chroma", "test-model", 5)
        mock_ai_generator.assert_called_once_with("test-key", "claude-3-sonnet")
        mock_session_manager.assert_called_once_with(10)
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_add_course_document_success(self, mock_search_tool, mock_tool_manager, 
                                          mock_session_manager, mock_ai_generator, 
                                          mock_vector_store, mock_document_processor):
        """Test successfully adding a course document"""
        # Setup mocks
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 1000
        mock_config.CHUNK_OVERLAP = 200
        
        mock_doc_processor_instance = Mock()
        mock_vector_store_instance = Mock()
        mock_document_processor.return_value = mock_doc_processor_instance
        mock_vector_store.return_value = mock_vector_store_instance
        
        # Mock course and chunks
        mock_course = Mock()
        mock_course.title = "Test Course"
        mock_chunks = [Mock(), Mock(), Mock()]  # 3 chunks
        
        mock_doc_processor_instance.process_course_document.return_value = (mock_course, mock_chunks)
        
        rag_system = RAGSystem(mock_config)
        
        # Test adding document
        result_course, chunk_count = rag_system.add_course_document("/test/course.txt")
        
        # Verify results
        assert result_course == mock_course
        assert chunk_count == 3
        
        # Verify calls
        mock_doc_processor_instance.process_course_document.assert_called_once_with("/test/course.txt")
        mock_vector_store_instance.add_course_metadata.assert_called_once_with(mock_course)
        mock_vector_store_instance.add_course_content.assert_called_once_with(mock_chunks)
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_add_course_document_error(self, mock_search_tool, mock_tool_manager, 
                                       mock_session_manager, mock_ai_generator, 
                                       mock_vector_store, mock_document_processor):
        """Test handling error when adding course document"""
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 1000
        mock_config.CHUNK_OVERLAP = 200
        
        mock_doc_processor_instance = Mock()
        mock_document_processor.return_value = mock_doc_processor_instance
        
        # Mock error
        mock_doc_processor_instance.process_course_document.side_effect = Exception("Processing error")
        
        rag_system = RAGSystem(mock_config)
        
        # Test adding document with error
        result_course, chunk_count = rag_system.add_course_document("/test/course.txt")
        
        # Verify error handling
        assert result_course is None
        assert chunk_count == 0
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_add_course_folder_success(self, mock_listdir, mock_exists, mock_search_tool, 
                                       mock_tool_manager, mock_session_manager, 
                                       mock_ai_generator, mock_vector_store, mock_document_processor):
        """Test successfully adding course folder"""
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 1000
        mock_config.CHUNK_OVERLAP = 200
        
        mock_doc_processor_instance = Mock()
        mock_vector_store_instance = Mock()
        mock_document_processor.return_value = mock_doc_processor_instance
        mock_vector_store.return_value = mock_vector_store_instance
        
        # Mock folder exists and has files
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.txt", "notes.pdf"]
        
        # Mock existing courses
        mock_vector_store_instance.get_existing_course_titles.return_value = ["Existing Course"]
        
        # Mock processing results
        mock_course1 = Mock()
        mock_course1.title = "New Course 1"
        mock_chunks1 = [Mock(), Mock()]
        
        mock_course2 = Mock()
        mock_course2.title = "Existing Course"  # Already exists
        mock_chunks2 = [Mock()]
        
        mock_doc_processor_instance.process_course_document.side_effect = [
            (mock_course1, mock_chunks1),
            (mock_course2, mock_chunks2)
        ]
        
        rag_system = RAGSystem(mock_config)
        
        # Test adding folder
        total_courses, total_chunks = rag_system.add_course_folder("/test/folder")
        
        # Verify results
        assert total_courses == 1  # Only new course added
        assert total_chunks == 2  # Chunks from new course
        
        # Verify calls
        assert mock_doc_processor_instance.process_course_document.call_count == 2
        mock_vector_store_instance.add_course_metadata.assert_called_once_with(mock_course1)
        mock_vector_store_instance.add_course_content.assert_called_once_with(mock_chunks1)
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    @patch('os.path.exists')
    def test_add_course_folder_not_exists(self, mock_exists, mock_search_tool, 
                                           mock_tool_manager, mock_session_manager, 
                                           mock_ai_generator, mock_vector_store, mock_document_processor):
        """Test adding course folder that doesn't exist"""
        mock_config = Mock()
        
        mock_exists.return_value = False
        
        rag_system = RAGSystem(mock_config)
        
        # Test adding non-existent folder
        total_courses, total_chunks = rag_system.add_course_folder("/nonexistent/folder")
        
        # Verify results
        assert total_courses == 0
        assert total_chunks == 0
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_add_course_folder_clear_existing(self, mock_listdir, mock_exists, mock_search_tool, 
                                             mock_tool_manager, mock_session_manager, 
                                             mock_ai_generator, mock_vector_store, mock_document_processor):
        """Test adding course folder with clear_existing=True"""
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 1000
        mock_config.CHUNK_OVERLAP = 200
        
        mock_doc_processor_instance = Mock()
        mock_vector_store_instance = Mock()
        mock_document_processor.return_value = mock_doc_processor_instance
        mock_vector_store.return_value = mock_vector_store_instance
        
        # Mock folder exists
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt"]
        
        # Mock processing
        mock_course = Mock()
        mock_course.title = "Test Course"
        mock_chunks = [Mock()]
        
        mock_doc_processor_instance.process_course_document.return_value = (mock_course, mock_chunks)
        mock_vector_store_instance.get_existing_course_titles.return_value = []
        
        rag_system = RAGSystem(mock_config)
        
        # Test adding folder with clear
        total_courses, total_chunks = rag_system.add_course_folder("/test/folder", clear_existing=True)
        
        # Verify clear was called
        mock_vector_store_instance.clear_all_data.assert_called_once()
        
        # Verify results
        assert total_courses == 1
        assert total_chunks == 1
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_query_without_session(self, mock_search_tool, mock_tool_manager, 
                                   mock_session_manager, mock_ai_generator, 
                                   mock_vector_store, mock_document_processor):
        """Test querying without a session"""
        mock_config = Mock()
        
        mock_ai_generator_instance = Mock()
        mock_ai_generator.return_value = mock_ai_generator_instance
        mock_tool_manager_instance = Mock()
        mock_tool_manager.return_value = mock_tool_manager_instance
        
        mock_ai_generator_instance.generate_response.return_value = "AI response"
        mock_tool_manager_instance.get_last_sources.return_value = ["Source 1"]
        
        rag_system = RAGSystem(mock_config)
        
        # Test query
        response, sources = rag_system.query("What is AI?")
        
        # Verify results
        assert response == "AI response"
        assert sources == ["Source 1"]
        
        # Verify calls
        mock_ai_generator_instance.generate_response.assert_called_once()
        mock_tool_manager_instance.get_last_sources.assert_called_once()
        mock_tool_manager_instance.reset_sources.assert_called_once()
        # No session manager calls since no session_id
        mock_session_manager_instance.add_exchange.assert_not_called()
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_query_with_session(self, mock_search_tool, mock_tool_manager, 
                                mock_session_manager, mock_ai_generator, 
                                mock_vector_store, mock_document_processor):
        """Test querying with a session"""
        mock_config = Mock()
        
        mock_ai_generator_instance = Mock()
        mock_ai_generator.return_value = mock_ai_generator_instance
        mock_tool_manager_instance = Mock()
        mock_tool_manager.return_value = mock_tool_manager_instance
        mock_session_manager_instance = Mock()
        mock_session_manager.return_value = mock_session_manager_instance
        
        # Mock session history
        mock_session_manager_instance.get_conversation_history.return_value = "Previous conversation"
        mock_ai_generator_instance.generate_response.return_value = "AI response with context"
        mock_tool_manager_instance.get_last_sources.return_value = []
        
        rag_system = RAGSystem(mock_config)
        
        # Test query with session
        response, sources = rag_system.query("What is ML?", session_id="session_123")
        
        # Verify results
        assert response == "AI response with context"
        assert sources == []
        
        # Verify calls
        mock_session_manager_instance.get_conversation_history.assert_called_once_with("session_123")
        mock_ai_generator_instance.generate_response.assert_called_once()
        call_args = mock_ai_generator_instance.generate_response.call_args
        assert call_args.kwargs['conversation_history'] == "Previous conversation"
        assert 'tools' in call_args.kwargs
        assert 'tool_manager' in call_args.kwargs
        
        # Verify conversation was updated
        mock_session_manager_instance.add_exchange.assert_called_once_with(
            "session_123", "What is ML?", "AI response with context"
        )
        
        # Verify sources were reset
        mock_tool_manager_instance.reset_sources.assert_called_once()
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_get_course_analytics(self, mock_search_tool, mock_tool_manager, 
                                  mock_session_manager, mock_ai_generator, 
                                  mock_vector_store, mock_document_processor):
        """Test getting course analytics"""
        mock_config = Mock()
        
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        
        # Mock analytics data
        mock_vector_store_instance.get_course_count.return_value = 5
        mock_vector_store_instance.get_existing_course_titles.return_value = [
            "Course 1", "Course 2", "Course 3", "Course 4", "Course 5"
        ]
        
        rag_system = RAGSystem(mock_config)
        
        # Test getting analytics
        analytics = rag_system.get_course_analytics()
        
        # Verify results
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course 1" in analytics["course_titles"]
        
        # Verify calls
        mock_vector_store_instance.get_course_count.assert_called_once()
        mock_vector_store_instance.get_existing_course_titles.assert_called_once()
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_query_prompt_formatting(self, mock_search_tool, mock_tool_manager, 
                                      mock_session_manager, mock_ai_generator, 
                                      mock_vector_store, mock_document_processor):
        """Test that query prompt is formatted correctly"""
        mock_config = Mock()
        
        mock_ai_generator_instance = Mock()
        mock_ai_generator.return_value = mock_ai_generator_instance
        mock_tool_manager_instance = Mock()
        mock_tool_manager.return_value = mock_tool_manager_instance
        
        rag_system = RAGSystem(mock_config)
        
        # Test query
        rag_system.query("What is neural networks?")
        
        # Verify prompt format
        call_args = mock_ai_generator_instance.generate_response.call_args
        expected_prompt = "Answer this question about course materials: What is neural networks?"
        assert call_args.kwargs['query'] == expected_prompt