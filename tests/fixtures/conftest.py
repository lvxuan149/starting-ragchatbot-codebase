import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    with patch('anthropic.Anthropic') as mock:
        client = Mock()
        mock.return_value = client
        client.messages.create.return_value = Mock(
            content=[Mock(text="Mock AI response")],
            id="msg_123",
            type="message",
            role="assistant",
            model="claude-3-sonnet-20240229",
            stop_reason="end_turn"
        )
        yield client

@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client for testing."""
    with patch('chromadb.Client') as mock:
        client = Mock()
        mock.return_value = client
        
        # Mock collection
        collection = Mock()
        collection.query.return_value = {
            "documents": ["Document 1 content", "Document 2 content"],
            "metadatas": [
                {"source": "doc1.txt", "course_id": "course1"},
                {"source": "doc2.txt", "course_id": "course2"}
            ],
            "ids": ["doc1_chunk1", "doc2_chunk1"],
            "distances": [0.1, 0.2]
        }
        collection.add.return_value = {"ids": ["doc1_chunk1"]}
        collection.get.return_value = {
            "documents": ["Document 1 content"],
            "metadatas": [{"source": "doc1.txt"}]
        }
        
        client.get_or_create_collection.return_value = collection
        yield client

@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence transformer for testing."""
    with patch('sentence_transformers.SentenceTransformer') as mock:
        model = Mock()
        model.encode.return_value = [[0.1, 0.2, 0.3]]  # Mock embedding
        mock.return_value = model
        yield model

@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    with patch('backend.config.Config') as mock:
        config = Mock()
        config.anthropic_api_key = "test_api_key"
        config.model_name = "claude-3-sonnet-20240229"
        config.embedding_model = "all-MiniLM-L6-v2"
        config.chunk_size = 1000
        config.chunk_overlap = 200
        config.max_tokens = 1000
        config.temperature = 0.7
        config.top_k = 5
        config.similarity_threshold = 0.7
        yield config

@pytest.fixture
def temp_vector_store_dir(tmp_path):
    """Create a temporary directory for vector store testing."""
    vector_store_dir = tmp_path / "vector_store"
    vector_store_dir.mkdir()
    return str(vector_store_dir)

@pytest.fixture
def sample_chunks():
    """Provide sample document chunks for testing."""
    return [
        "This is the first chunk of text about machine learning.",
        "This is the second chunk discussing supervised learning.",
        "This is the third chunk covering neural networks."
    ]

@pytest.fixture
def sample_metadatas():
    """Provide sample metadata for testing."""
    return [
        {"source": "course1.txt", "chunk_index": 0, "course_id": "course1"},
        {"source": "course1.txt", "chunk_index": 1, "course_id": "course1"},
        {"source": "course2.txt", "chunk_index": 0, "course_id": "course2"}
    ]