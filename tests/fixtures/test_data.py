import pytest
from pathlib import Path
from typing import Dict, Any

# Sample course document content for testing
SAMPLE_COURSE_CONTENT = """
Course 1: Introduction to Machine Learning
========================================

This course provides a comprehensive introduction to machine learning concepts and applications.

Week 1: Introduction
- What is Machine Learning?
- Types of ML: Supervised, Unsupervised, Reinforcement
- Basic terminology and concepts

Week 2: Supervised Learning
- Classification algorithms
- Regression algorithms
- Model evaluation metrics

Week 3: Unsupervised Learning
- Clustering algorithms
- Dimensionality reduction
- Anomaly detection

Instructor: Dr. Jane Smith
Duration: 8 weeks
Level: Beginner
"""

# Mock course metadata
MOCK_COURSE_METADATA = {
    "course1": {
        "title": "Introduction to Machine Learning",
        "instructor": "Dr. Jane Smith",
        "duration": "8 weeks",
        "level": "Beginner",
        "description": "Comprehensive introduction to ML concepts"
    },
    "course2": {
        "title": "Advanced Deep Learning",
        "instructor": "Dr. John Doe",
        "duration": "12 weeks",
        "level": "Advanced",
        "description": "Deep dive into neural networks and deep learning"
    }
}

# Sample user queries for testing
SAMPLE_QUERIES = [
    "What is machine learning?",
    "Tell me about supervised learning algorithms",
    "Who teaches the machine learning course?",
    "How long is the deep learning course?",
    "What are the prerequisites for advanced courses?"
]

# Sample AI responses for mocking
MOCK_AI_RESPONSES = {
    "basic_response": "Machine learning is a subset of artificial intelligence...",
    "detailed_response": "Supervised learning involves training models on labeled data...",
    "course_info_response": "The Introduction to Machine Learning course is taught by Dr. Jane Smith..."
}

@pytest.fixture
def sample_document_path(tmp_path) -> Path:
    """Create a temporary document file for testing."""
    doc_file = tmp_path / "sample_course.txt"
    doc_file.write_text(SAMPLE_COURSE_CONTENT)
    return doc_file

@pytest.fixture
def mock_course_metadata() -> Dict[str, Any]:
    """Provide mock course metadata."""
    return MOCK_COURSE_METADATA

@pytest.fixture
def sample_queries() -> list:
    """Provide sample user queries for testing."""
    return SAMPLE_QUERIES

@pytest.fixture
def mock_ai_responses() -> Dict[str, str]:
    """Provide mock AI responses."""
    return MOCK_AI_RESPONSES

@pytest.fixture
def test_session_id() -> str:
    """Provide a test session ID."""
    return "test_session_123"

@pytest.fixture
def test_user_message() -> Dict[str, Any]:
    """Provide a test user message."""
    return {
        "role": "user",
        "content": "What is machine learning?",
        "timestamp": "2024-01-01T10:00:00Z"
    }

@pytest.fixture
def test_ai_message() -> Dict[str, Any]:
    """Provide a test AI message."""
    return {
        "role": "assistant",
        "content": "Machine learning is a subset of artificial intelligence...",
        "timestamp": "2024-01-01T10:00:05Z"
    }

@pytest.fixture
def mock_search_results() -> list:
    """Provide mock search results."""
    return [
        {
            "content": "Machine learning is a subset of artificial intelligence...",
            "metadata": {
                "source": "course1_script.txt",
                "course_id": "course1",
                "chunk_index": 0
            },
            "score": 0.95
        },
        {
            "content": "Supervised learning involves training models on labeled data...",
            "metadata": {
                "source": "course1_script.txt",
                "course_id": "course1",
                "chunk_index": 1
            },
            "score": 0.87
        }
    ]