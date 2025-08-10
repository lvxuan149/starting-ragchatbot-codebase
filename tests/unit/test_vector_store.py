import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import json

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Test cases for SearchResults class"""
    
    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results with data"""
        chroma_results = {
            'documents': [['Doc 1', 'Doc 2']],
            'metadatas': [[{'source': 'doc1'}, {'source': 'doc2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['Doc 1', 'Doc 2']
        assert results.metadata == [{'source': 'doc1'}, {'source': 'doc2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
    
    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None
    
    def test_from_chroma_none_values(self):
        """Test creating SearchResults when ChromaDB returns None"""
        chroma_results = {
            'documents': None,
            'metadatas': None,
            'distances': None
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
    
    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        results = SearchResults.empty("Test error message")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Test error message"
    
    def test_is_empty_true(self):
        """Test is_empty when no documents"""
        results = SearchResults(documents=[], metadata=[], distances=[])
        assert results.is_empty() is True
    
    def test_is_empty_false(self):
        """Test is_empty when documents exist"""
        results = SearchResults(documents=['Doc 1'], metadata=[{}], distances=[0.1])
        assert results.is_empty() is False


class TestVectorStore:
    """Test cases for VectorStore class"""
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_init(self, mock_embedding_func, mock_chroma_client):
        """Test VectorStore initialization"""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        # Mock collection creation
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore(
            chroma_path="/test/path",
            embedding_model="test-model",
            max_results=10
        )
        
        assert store.max_results == 10
        mock_chroma_client.assert_called_once()
        mock_embedding_func.assert_called_once_with(model_name="test-model")
        assert mock_client_instance.get_or_create_collection.call_count == 2
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_basic(self, mock_embedding_func, mock_chroma_client):
        """Test basic search functionality"""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        # Mock collections
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock search results
        mock_content_collection.query.return_value = {
            'documents': [['Result 1', 'Result 2']],
            'metadatas': [[{'course': 'test'}, {'course': 'test'}]],
            'distances': [[0.1, 0.2]]
        }
        
        store = VectorStore(chroma_path="/test/path", embedding_model="test-model")
        results = store.search("test query")
        
        assert results.documents == ['Result 1', 'Result 2']
        assert len(results.documents) == 2
        mock_content_collection.query.assert_called_once()
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_course_filter(self, mock_embedding_func, mock_chroma_client):
        """Test search with course name filter"""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        # Mock collections
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock course resolution
        mock_catalog_collection.query.return_value = {
            'documents': [['Course Title']],
            'metadatas': [[{'title': 'Resolved Course Title'}]]
        }
        
        # Mock content search
        mock_content_collection.query.return_value = {
            'documents': [['Filtered Result']],
            'metadatas': [[{'course_title': 'Resolved Course Title'}]],
            'distances': [[0.1]]
        }
        
        store = VectorStore(chroma_path="/test/path", embedding_model="test-model")
        results = store.search("test query", course_name="Test Course")
        
        assert results.documents == ['Filtered Result']
        # Verify course resolution was called
        mock_catalog_collection.query.assert_called_once()
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_course_not_found(self, mock_embedding_func, mock_chroma_client):
        """Test search when course is not found"""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        # Mock collections
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock empty course resolution
        mock_catalog_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        store = VectorStore(chroma_path="/test/path", embedding_model="test-model")
        results = store.search("test query", course_name="Nonexistent Course")
        
        assert results.is_empty() is True
        assert "No course found matching" in results.error
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_lesson_filter(self, mock_embedding_func, mock_chroma_client):
        """Test search with lesson number filter"""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        # Mock collections
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock search results
        mock_content_collection.query.return_value = {
            'documents': [['Lesson Result']],
            'metadatas': [[{'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        
        store = VectorStore(chroma_path="/test/path", embedding_model="test-model")
        results = store.search("test query", lesson_number=1)
        
        assert results.documents == ['Lesson Result']
        # Verify filter was applied
        call_args = mock_content_collection.query.call_args
        assert 'where' in call_args.kwargs
        assert call_args.kwargs['where']['lesson_number'] == 1
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_error(self, mock_embedding_func, mock_chroma_client):
        """Test search when an error occurs"""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        # Mock collections
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock search error
        mock_content_collection.query.side_effect = Exception("Database error")
        
        store = VectorStore(chroma_path="/test/path", embedding_model="test-model")
        results = store.search("test query")
        
        assert results.is_empty() is True
        assert "Search error" in results.error
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_add_course_metadata(self, mock_embedding_func, mock_chroma_client):
        """Test adding course metadata"""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        # Mock collections
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Create test course
        course = Course(
            title="Test Course",
            course_link="https://example.com",
            instructor="Dr. Test"
        )
        course.lessons.append(Lesson(
            lesson_number=1,
            title="Introduction",
            lesson_link="https://example.com/lesson1"
        ))
        
        store = VectorStore(chroma_path="/test/path", embedding_model="test-model")
        store.add_course_metadata(course)
        
        # Verify metadata was added
        mock_catalog_collection.add.assert_called_once()
        call_args = mock_catalog_collection.add.call_args
        assert call_args.kwargs['documents'] == ["Test Course"]
        assert call_args.kwargs['metadatas'][0]['title'] == "Test Course"
        assert call_args.kwargs['metadatas'][0]['instructor'] == "Dr. Test"
        # Verify lessons are serialized as JSON
        lessons_json = call_args.kwargs['metadatas'][0]['lessons_json']
        lessons_data = json.loads(lessons_json)
        assert len(lessons_data) == 1
        assert lessons_data[0]['lesson_number'] == 1
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_add_course_content(self, mock_embedding_func, mock_chroma_client):
        """Test adding course content chunks"""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        # Mock collections
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Create test chunks
        chunks = [
            CourseChunk(
                content="Chunk 1 content",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Chunk 2 content",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=1
            )
        ]
        
        store = VectorStore(chroma_path="/test/path", embedding_model="test-model")
        store.add_course_content(chunks)
        
        # Verify chunks were added
        mock_content_collection.add.assert_called_once()
        call_args = mock_content_collection.add.call_args
        assert len(call_args.kwargs['documents']) == 2
        assert call_args.kwargs['documents'][0] == "Chunk 1 content"
        assert call_args.kwargs['metadatas'][0]['course_title'] == "Test Course"
        assert call_args.kwargs['metadatas'][0]['lesson_number'] == 1
        assert call_args.kwargs['ids'][0] == "Test_Course_0"
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_add_course_content_empty(self, mock_embedding_func, mock_chroma_client):
        """Test adding empty course content"""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        # Mock collections
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        store = VectorStore(chroma_path="/test/path", embedding_model="test-model")
        store.add_course_content([])
        
        # Should not call add when chunks are empty
        mock_content_collection.add.assert_not_called()
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_clear_all_data(self, mock_embedding_func, mock_chroma_client):
        """Test clearing all data"""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        # Mock collections
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        store = VectorStore(chroma_path="/test/path", embedding_model="test-model")
        store.clear_all_data()
        
        # Verify collections were deleted and recreated
        assert mock_client_instance.delete_collection.call_count == 2
        assert mock_client_instance.get_or_create_collection.call_count == 4  # Initial 2 + 2 after clear
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_existing_course_titles(self, mock_embedding_func, mock_chroma_client):
        """Test getting existing course titles"""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        # Mock collections
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock get results
        mock_catalog_collection.get.return_value = {
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }
        
        store = VectorStore(chroma_path="/test/path", embedding_model="test-model")
        titles = store.get_existing_course_titles()
        
        assert titles == ['Course 1', 'Course 2', 'Course 3']
        mock_catalog_collection.get.assert_called_once()
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_course_count(self, mock_embedding_func, mock_chroma_client):
        """Test getting course count"""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        # Mock collections
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock get results
        mock_catalog_collection.get.return_value = {
            'ids': ['Course 1', 'Course 2']
        }
        
        store = VectorStore(chroma_path="/test/path", embedding_model="test-model")
        count = store.get_course_count()
        
        assert count == 2
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_all_courses_metadata(self, mock_embedding_func, mock_chroma_client):
        """Test getting all courses metadata"""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        # Mock collections
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock get results with lessons JSON
        lessons_data = [
            {"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "link1"},
            {"lesson_number": 2, "lesson_title": "Advanced", "lesson_link": "link2"}
        ]
        mock_catalog_collection.get.return_value = {
            'metadatas': [{
                'title': 'Test Course',
                'instructor': 'Dr. Test',
                'lessons_json': json.dumps(lessons_data)
            }]
        }
        
        store = VectorStore(chroma_path="/test/path", embedding_model="test-model")
        metadata = store.get_all_courses_metadata()
        
        assert len(metadata) == 1
        assert metadata[0]['title'] == 'Test Course'
        assert 'lessons' in metadata[0]
        assert 'lessons_json' not in metadata[0]  # Should be removed after parsing
        assert len(metadata[0]['lessons']) == 2
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_course_link(self, mock_embedding_func, mock_chroma_client):
        """Test getting course link"""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        # Mock collections
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock get results
        mock_catalog_collection.get.return_value = {
            'metadatas': [{'course_link': 'https://example.com/course'}]
        }
        
        store = VectorStore(chroma_path="/test/path", embedding_model="test-model")
        link = store.get_course_link("Test Course")
        
        assert link == "https://example.com/course"
        mock_catalog_collection.get.assert_called_once_with(ids=["Test Course"])
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_lesson_link(self, mock_embedding_func, mock_chroma_client):
        """Test getting lesson link"""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        # Mock collections
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog_collection, mock_content_collection]
        
        # Mock get results with lessons
        lessons_data = [
            {"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "link1"},
            {"lesson_number": 2, "lesson_title": "Advanced", "lesson_link": "link2"}
        ]
        mock_catalog_collection.get.return_value = {
            'metadatas': [{'lessons_json': json.dumps(lessons_data)}]
        }
        
        store = VectorStore(chroma_path="/test/path", embedding_model="test-model")
        link = store.get_lesson_link("Test Course", 2)
        
        assert link == "link2"
    
    def test_build_filter_no_filters(self):
        """Test _build_filter with no filters"""
        with patch.object(VectorStore, '_create_collection'):
            store = VectorStore.__new__(VectorStore)
            filter_dict = store._build_filter(None, None)
            assert filter_dict is None
    
    def test_build_filter_course_only(self):
        """Test _build_filter with course title only"""
        with patch.object(VectorStore, '_create_collection'):
            store = VectorStore.__new__(VectorStore)
            filter_dict = store._build_filter("Test Course", None)
            assert filter_dict == {"course_title": "Test Course"}
    
    def test_build_filter_lesson_only(self):
        """Test _build_filter with lesson number only"""
        with patch.object(VectorStore, '_create_collection'):
            store = VectorStore.__new__(VectorStore)
            filter_dict = store._build_filter(None, 1)
            assert filter_dict == {"lesson_number": 1}
    
    def test_build_filter_both(self):
        """Test _build_filter with both course and lesson filters"""
        with patch.object(VectorStore, '_create_collection'):
            store = VectorStore.__new__(VectorStore)
            filter_dict = store._build_filter("Test Course", 1)
            expected = {"$and": [{"course_title": "Test Course"}, {"lesson_number": 1}]}
            assert filter_dict == expected