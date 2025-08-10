import pytest
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from document_processor import DocumentProcessor
from models import Course, Lesson, CourseChunk


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class"""
    
    def test_init(self):
        """Test DocumentProcessor initialization"""
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 200
    
    def test_read_file_utf8(self, sample_document_path):
        """Test reading file with UTF-8 encoding"""
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        content = processor.read_file(str(sample_document_path))
        assert "Introduction to Machine Learning" in content
        assert "Dr. Jane Smith" in content
    
    def test_read_file_with_encoding_error(self, tmp_path):
        """Test reading file with encoding error handling"""
        # Create a file with non-UTF-8 content
        test_file = tmp_path / "binary_file.txt"
        with open(test_file, 'wb') as f:
            f.write(b'Hello\xffWorld')
        
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        content = processor.read_file(str(test_file))
        # Should handle encoding error gracefully
        assert "Hello" in content
    
    def test_chunk_text_simple(self):
        """Test basic text chunking"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = processor.chunk_text(text)
        
        assert len(chunks) >= 1
        assert all(len(chunk) <= processor.chunk_size for chunk in chunks)
        assert "sentence one" in chunks[0]
    
    def test_chunk_text_with_overlap(self):
        """Test text chunking with overlap"""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=20)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = processor.chunk_text(text)
        
        if len(chunks) > 1:
            # Check for overlap between consecutive chunks
            first_chunk = chunks[0]
            second_chunk = chunks[1]
            # Should have some overlapping words
            assert len(set(first_chunk.split()) & set(second_chunk.split())) > 0
    
    def test_chunk_text_empty(self):
        """Test chunking empty text"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        chunks = processor.chunk_text("")
        assert chunks == []
    
    def test_chunk_text_single_long_sentence(self):
        """Test chunking a single long sentence"""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        text = "This is a very long sentence that should be split into multiple chunks because it exceeds the chunk size limit significantly."
        chunks = processor.chunk_text(text)
        
        assert len(chunks) >= 1
        assert all(len(chunk) <= processor.chunk_size for chunk in chunks)
    
    def test_chunk_text_with_abbreviations(self):
        """Test chunking text with abbreviations like Dr., Mr., etc."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        text = "Dr. Smith teaches the course. Mr. Jones is the assistant. The course covers A.I. and M.L. concepts."
        chunks = processor.chunk_text(text)
        
        # Should not split on abbreviations
        assert "Dr. Smith" in chunks[0]
        assert len(chunks) >= 1
    
    def test_process_course_document_standard_format(self, tmp_path):
        """Test processing a course document with standard format"""
        # Create a test course document
        course_file = tmp_path / "test_course.txt"
        course_content = """Course Title: Test Course
Course Link: https://example.com/course
Course Instructor: Dr. Test Instructor

Lesson 0: Introduction
This is the introduction lesson content.
It covers basic concepts.

Lesson 1: Advanced Topics
This lesson covers advanced topics.
It includes complex theories and applications."""
        
        course_file.write_text(course_content)
        
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        course, chunks = processor.process_course_document(str(course_file))
        
        # Verify course metadata
        assert course.title == "Test Course"
        assert course.course_link == "https://example.com/course"
        assert course.instructor == "Dr. Test Instructor"
        
        # Verify lessons
        assert len(course.lessons) == 2
        assert course.lessons[0].lesson_number == 0
        assert course.lessons[0].title == "Introduction"
        assert course.lessons[1].lesson_number == 1
        assert course.lessons[1].title == "Advanced Topics"
        
        # Verify chunks
        assert len(chunks) > 0
        assert all(chunk.course_title == "Test Course" for chunk in chunks)
    
    def test_process_course_document_without_metadata(self, tmp_path):
        """Test processing a course document without explicit metadata"""
        course_file = tmp_path / "simple_course.txt"
        course_content = """Simple Course Title
No specific metadata here.

Lesson 1: Getting Started
This is the first lesson.
It introduces basic concepts."""
        
        course_file.write_text(course_content)
        
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        course, chunks = processor.process_course_document(str(course_file))
        
        # Should use filename as title
        assert course.title == "simple_course.txt"
        assert course.instructor is None
    
    def test_process_course_document_no_lessons(self, tmp_path):
        """Test processing a course document with no lesson markers"""
        course_file = tmp_path / "no_lessons.txt"
        course_content = """Course Title: Free Form Course
Course Instructor: Dr. Author

This is just free form content without lesson markers.
It should still be processed into chunks.
The content covers various topics related to the subject."""
        
        course_file.write_text(course_content)
        
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        course, chunks = processor.process_course_document(str(course_file))
        
        # Should still create course and chunks
        assert course.title == "Free Form Course"
        assert len(chunks) > 0
    
    def test_process_course_document_with_lesson_links(self, tmp_path):
        """Test processing course document with lesson links"""
        course_file = tmp_path / "course_with_links.txt"
        course_content = """Course Title: Linked Course
Course Instructor: Dr. Linker

Lesson 0: Overview
Lesson Link: https://example.com/lesson0
This is the overview lesson.

Lesson 1: Details
Lesson Link: https://example.com/lesson1
This lesson provides details."""
        
        course_file.write_text(course_content)
        
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        course, chunks = processor.process_course_document(str(course_file))
        
        # Verify lesson links are captured
        assert course.lessons[0].lesson_link == "https://example.com/lesson0"
        assert course.lessons[1].lesson_link == "https://example.com/lesson1"
    
    def test_process_course_document_empty_file(self, tmp_path):
        """Test processing an empty course file"""
        course_file = tmp_path / "empty.txt"
        course_file.write_text("")
        
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        course, chunks = processor.process_course_document(str(course_file))
        
        # Should handle empty file gracefully
        assert course.title == "empty.txt"
        assert len(course.lessons) == 0
        assert len(chunks) == 0
    
    def test_process_course_document_very_long_content(self, tmp_path):
        """Test processing a document with very long content"""
        course_file = tmp_path / "long_content.txt"
        # Create content that will definitely be chunked
        long_content = "Course Title: Long Course\n\n" + "\n".join([f"This is sentence {i}." for i in range(100)])
        course_file.write_text(long_content)
        
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        course, chunks = processor.process_course_document(str(course_file))
        
        # Should create multiple chunks
        assert len(chunks) > 1
        # All chunks should have reasonable size
        assert all(len(chunk.content) <= processor.chunk_size + 50 for chunk in chunks)  # Allow some buffer