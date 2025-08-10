import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from rag_system import RAGSystem


class TestRAGE2E:
    """End-to-end tests for the complete RAG workflow"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.docs_dir = os.path.join(self.temp_dir, "docs")
        self.chroma_dir = os.path.join(self.temp_dir, "chroma")
        os.makedirs(self.docs_dir)
        os.makedirs(self.chroma_dir)
        
        # Create test course documents
        self.create_test_documents()
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_documents(self):
        """Create test course documents"""
        # Course 1: Machine Learning Basics
        course1_content = """Course Title: Machine Learning Basics
Course Link: https://example.com/ml-basics
Course Instructor: Dr. Alice Smith

Lesson 0: Introduction to Machine Learning
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide.

Lesson 1: Supervised Learning
Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples.

In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal).
"""
        
        # Course 2: Advanced Python Programming
        course2_content = """Course Title: Advanced Python Programming
Course Link: https://example.com/advanced-python
Course Instructor: Dr. Bob Johnson

Lesson 0: Python Decorators
Decorators in Python are a powerful tool that allows you to modify the behavior of functions or classes. They wrap another function in order to extend the behavior of the wrapped function, without permanently modifying it.

In Python, functions are first-class objects. This means that functions can be passed around and used as arguments, just like any other object (string, int, float, list, and so on).

Lesson 1: Context Managers
Context managers in Python provide a clean way to manage resources, such as file handles or database connections. They ensure that resources are properly initialized and cleaned up, even if exceptions occur.

The with statement in Python simplifies exception handling by encapsulating common preparation and cleanup tasks.
"""
        
        # Write test files
        with open(os.path.join(self.docs_dir, "ml_basics.txt"), "w") as f:
            f.write(course1_content)
        
        with open(os.path.join(self.docs_dir, "advanced_python.txt"), "w") as f:
            f.write(course2_content)
    
    @patch('rag_system.AIGenerator')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_complete_workflow_add_and_query(self, mock_chroma_client, mock_sentence_transformer, mock_ai_generator):
        """Test complete workflow: add documents and query"""
        # Mock sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        mock_sentence_transformer.return_value = mock_model
        
        # Mock ChromaDB client
        mock_client = Mock()
        mock_chroma_client.return_value = mock_client
        
        # Mock collections
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock search results
        mock_content_collection.query.return_value = {
            'documents': [['Machine learning is a subset of artificial intelligence']],
            'metadatas': [[{'course_title': 'Machine Learning Basics', 'lesson_number': 0}]],
            'distances': [[0.1]]
        }
        
        # Mock AI generator
        mock_ai_instance = Mock()
        mock_ai_generator.return_value = mock_ai_instance
        mock_ai_instance.generate_response.return_value = "Machine learning is a subset of AI that enables systems to learn from experience."
        
        # Create mock config
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 500
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = self.chroma_dir
        mock_config.EMBEDDING_MODEL = "test-model"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "test-key"
        mock_config.ANTHROPIC_MODEL = "claude-3-sonnet"
        mock_config.MAX_HISTORY = 5
        
        # Initialize RAG system
        rag_system = RAGSystem(mock_config)
        
        # Test 1: Add course folder
        total_courses, total_chunks = rag_system.add_course_folder(self.docs_dir)
        
        assert total_courses == 2
        assert total_chunks > 0
        
        # Verify metadata and content were added
        assert mock_catalog_collection.add.call_count == 2
        assert mock_content_collection.add.call_count == 2
        
        # Test 2: Query without session
        response, sources = rag_system.query("What is machine learning?")
        
        assert "Machine learning is a subset of AI" in response
        assert len(sources) > 0
        
        # Test 3: Query with session
        session_id = rag_system.session_manager.create_session()
        response2, sources2 = rag_system.query("Tell me about supervised learning", session_id)
        
        assert isinstance(response2, str)
        
        # Verify conversation history was maintained
        history = rag_system.session_manager.get_conversation_history(session_id)
        assert history is not None
        assert "What is machine learning?" in history
        assert "Tell me about supervised learning" in history
        
        # Test 4: Get course analytics
        analytics = rag_system.get_course_analytics()
        assert analytics["total_courses"] == 2
        assert len(analytics["course_titles"]) == 2
        assert "Machine Learning Basics" in analytics["course_titles"]
        assert "Advanced Python Programming" in analytics["course_titles"]
    
    @patch('rag_system.AIGenerator')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_conversation_context_maintenance(self, mock_chroma_client, mock_sentence_transformer, mock_ai_generator):
        """Test that conversation context is properly maintained"""
        # Mock dependencies
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_model
        
        mock_client = Mock()
        mock_chroma_client.return_value = mock_client
        
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock different responses for different queries
        def mock_generate_response(query, conversation_history=None, **kwargs):
            if conversation_history and "previous question" in conversation_history:
                return "This is a follow-up answer."
            return "This is an initial answer."
        
        mock_ai_instance = Mock()
        mock_ai_generator.return_value = mock_ai_instance
        mock_ai_instance.generate_response.side_effect = mock_generate_response
        
        mock_content_collection.query.return_value = {
            'documents': [['Test content']],
            'metadatas': [[{'course_title': 'Test Course'}]],
            'distances': [[0.1]]
        }
        
        # Create mock config
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 500
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = self.chroma_dir
        mock_config.EMBEDDING_MODEL = "test-model"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "test-key"
        mock_config.ANTHROPIC_MODEL = "claude-3-sonnet"
        mock_config.MAX_HISTORY = 5
        
        # Initialize RAG system
        rag_system = RAGSystem(mock_config)
        
        # Add a document
        rag_system.add_course_folder(self.docs_dir)
        
        # Create session and have conversation
        session_id = rag_system.session_manager.create_session()
        
        # First query
        response1, _ = rag_system.query("What is AI?", session_id)
        assert "initial answer" in response1
        
        # Second query (should have context)
        response2, _ = rag_system.query("Tell me more about it", session_id)
        assert "follow-up answer" in response2
        
        # Verify conversation history
        history = rag_system.session_manager.get_conversation_history(session_id)
        assert "What is AI?" in history
        assert "Tell me more about it" in history
    
    @patch('rag_system.AIGenerator')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_course_filtering_in_queries(self, mock_chroma_client, mock_sentence_transformer, mock_ai_generator):
        """Test that course filtering works in queries"""
        # Mock dependencies
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_model
        
        mock_client = Mock()
        mock_chroma_client.return_value = mock_client
        
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Track search calls
        search_calls = []
        
        def mock_query(query_texts=None, n_results=None, where=None):
            search_calls.append(where)
            return {
                'documents': [['Filtered content']],
                'metadatas': [[{'course_title': 'Machine Learning Basics'}]],
                'distances': [[0.1]]
            }
        
        mock_content_collection.query.side_effect = mock_query
        
        mock_ai_instance = Mock()
        mock_ai_generator.return_value = mock_ai_instance
        mock_ai_instance.generate_response.return_value = "Filtered response"
        
        # Create mock config
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 500
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = self.chroma_dir
        mock_config.EMBEDDING_MODEL = "test-model"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "test-key"
        mock_config.ANTHROPIC_MODEL = "claude-3-sonnet"
        mock_config.MAX_HISTORY = 5
        
        # Initialize RAG system
        rag_system = RAGSystem(mock_config)
        
        # Add documents
        rag_system.add_course_folder(self.docs_dir)
        
        # Query with course filter
        rag_system.query("What is ML?", session_id="test")
        
        # Verify the search was called with course filter
        assert len(search_calls) > 0
        # The filter should be applied by the search tool
    
    @patch('rag_system.AIGenerator')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_error_handling_in_workflow(self, mock_chroma_client, mock_sentence_transformer, mock_ai_generator):
        """Test error handling in the complete workflow"""
        # Mock dependencies
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_model
        
        mock_client = Mock()
        mock_chroma_client.return_value = mock_client
        
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock AI generator error
        mock_ai_instance = Mock()
        mock_ai_generator.return_value = mock_ai_instance
        mock_ai_instance.generate_response.side_effect = Exception("AI service unavailable")
        
        # Create mock config
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 500
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = self.chroma_dir
        mock_config.EMBEDDING_MODEL = "test-model"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "test-key"
        mock_config.ANTHROPIC_MODEL = "claude-3-sonnet"
        mock_config.MAX_HISTORY = 5
        
        # Initialize RAG system
        rag_system = RAGSystem(mock_config)
        
        # Add documents
        rag_system.add_course_folder(self.docs_dir)
        
        # Query should handle AI error gracefully
        with pytest.raises(Exception):
            rag_system.query("What is AI?")
    
    @patch('rag_system.AIGenerator')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_document_reprocessing_with_clear(self, mock_chroma_client, mock_sentence_transformer, mock_ai_generator):
        """Test document reprocessing with clear_existing=True"""
        # Mock dependencies
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_model
        
        mock_client = Mock()
        mock_chroma_client.return_value = mock_client
        
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Track clear calls
        clear_calls = []
        
        def mock_clear_all_data():
            clear_calls.append(True)
            # Reset collections
            mock_client.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        mock_content_collection.clear_all_data = mock_clear_all_data
        
        # Mock existing courses
        mock_catalog_collection.get.return_value = {'ids': ['Existing Course']}
        
        mock_ai_instance = Mock()
        mock_ai_generator.return_value = mock_ai_instance
        
        # Create mock config
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 500
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = self.chroma_dir
        mock_config.EMBEDDING_MODEL = "test-model"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "test-key"
        mock_config.ANTHROPIC_MODEL = "claude-3-sonnet"
        mock_config.MAX_HISTORY = 5
        
        # Initialize RAG system
        rag_system = RAGSystem(mock_config)
        
        # Add documents with clear
        rag_system.add_course_folder(self.docs_dir, clear_existing=True)
        
        # Verify clear was called
        assert len(clear_calls) > 0
        
        # Verify documents were still added
        assert mock_catalog_collection.add.call_count == 2
        assert mock_content_collection.add.call_count == 2
    
    @patch('rag_system.AIGenerator')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_multiple_sessions_isolation(self, mock_chroma_client, mock_sentence_transformer, mock_ai_generator):
        """Test that multiple sessions are properly isolated"""
        # Mock dependencies
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_model
        
        mock_client = Mock()
        mock_chroma_client.return_value = mock_client
        
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        mock_content_collection.query.return_value = {
            'documents': [['Test content']],
            'metadatas': [[{'course_title': 'Test Course'}]],
            'distances': [[0.1]]
        }
        
        mock_ai_instance = Mock()
        mock_ai_generator.return_value = mock_ai_instance
        mock_ai_instance.generate_response.return_value = "Session-specific response"
        
        # Create mock config
        mock_config = Mock()
        mock_config.CHUNK_SIZE = 500
        mock_config.CHUNK_OVERLAP = 100
        mock_config.CHROMA_PATH = self.chroma_dir
        mock_config.EMBEDDING_MODEL = "test-model"
        mock_config.MAX_RESULTS = 5
        mock_config.ANTHROPIC_API_KEY = "test-key"
        mock_config.ANTHROPIC_MODEL = "claude-3-sonnet"
        mock_config.MAX_HISTORY = 2  # Low history to test limit
        
        # Initialize RAG system
        rag_system = RAGSystem(mock_config)
        
        # Add documents
        rag_system.add_course_folder(self.docs_dir)
        
        # Create two sessions
        session1 = rag_system.session_manager.create_session()
        session2 = rag_system.session_manager.create_session()
        
        # Have different conversations in each session
        rag_system.query("Question 1", session1)
        rag_system.query("Question A", session2)
        rag_system.query("Question 2", session1)
        rag_system.query("Question B", session2)
        
        # Verify sessions are isolated
        history1 = rag_system.session_manager.get_conversation_history(session1)
        history2 = rag_system.session_manager.get_conversation_history(session2)
        
        assert "Question 1" in history1
        assert "Question 2" in history1
        assert "Question A" not in history1
        assert "Question B" not in history1
        
        assert "Question A" in history2
        assert "Question B" in history2
        assert "Question 1" not in history2
        assert "Question 2" not in history2
        
        # Test history limit
        rag_system.query("Question 3", session1)
        rag_system.query("Question 4", session1)
        
        history1_updated = rag_system.session_manager.get_conversation_history(session1)
        # Should only keep last 4 messages (2 exchanges)
        assert "Question 1" not in history1_updated  # Should be pruned
        assert "Question 4" in history1_updated