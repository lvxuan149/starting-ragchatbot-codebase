import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from search_tools import Tool, CourseSearchTool, ToolManager
from vector_store import SearchResults


class MockTool(Tool):
    """Mock tool for testing"""
    
    def get_tool_definition(self):
        return {
            "name": "mock_tool",
            "description": "A mock tool for testing",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                },
                "required": ["param1"]
            }
        }
    
    def execute(self, **kwargs):
        return f"Mock tool executed with: {kwargs}"


class TestTool:
    """Test cases for Tool abstract base class"""
    
    def test_tool_interface(self):
        """Test that Tool enforces interface implementation"""
        # This should work since MockTool implements all abstract methods
        tool = MockTool()
        assert hasattr(tool, 'get_tool_definition')
        assert hasattr(tool, 'execute')
    
    def test_cannot_instantiate_tool_directly(self):
        """Test that Tool cannot be instantiated directly"""
        with pytest.raises(TypeError):
            Tool()


class TestCourseSearchTool:
    """Test cases for CourseSearchTool class"""
    
    def test_init(self):
        """Test CourseSearchTool initialization"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)
        
        assert tool.store == mock_vector_store
        assert tool.last_sources == []
    
    def test_get_tool_definition(self):
        """Test tool definition structure"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)
        
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert "query" in definition["input_schema"]["required"]
    
    def test_execute_basic_search(self):
        """Test basic search execution"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)
        
        # Mock search results
        mock_results = SearchResults(
            documents=["Result 1 content", "Result 2 content"],
            metadata=[
                {"course_title": "Course 1", "lesson_number": 1},
                {"course_title": "Course 2", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.search.return_value = mock_results
        
        result = tool.execute(query="test query")
        
        # Verify search was called
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result formatting
        assert "[Course 1 - Lesson 1]" in result
        assert "[Course 2 - Lesson 2]" in result
        assert "Result 1 content" in result
        assert "Result 2 content" in result
        assert tool.last_sources == ["Course 1 - Lesson 1", "Course 2 - Lesson 2"]
    
    def test_execute_with_course_filter(self):
        """Test search with course name filter"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)
        
        mock_results = SearchResults(
            documents=["Filtered result"],
            metadata=[{"course_title": "ML Course", "lesson_number": None}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        
        result = tool.execute(query="test", course_name="ML")
        
        mock_vector_store.search.assert_called_once_with(
            query="test",
            course_name="ML",
            lesson_number=None
        )
        
        assert "[ML Course]" in result
        assert "Filtered result" in result
        assert tool.last_sources == ["ML Course"]
    
    def test_execute_with_lesson_filter(self):
        """Test search with lesson number filter"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)
        
        mock_results = SearchResults(
            documents=["Lesson result"],
            metadata=[{"course_title": "Python Course", "lesson_number": 3}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        
        result = tool.execute(query="functions", lesson_number=3)
        
        mock_vector_store.search.assert_called_once_with(
            query="functions",
            course_name=None,
            lesson_number=3
        )
        
        assert "[Python Course - Lesson 3]" in result
        assert "Lesson result" in result
        assert tool.last_sources == ["Python Course - Lesson 3"]
    
    def test_execute_with_both_filters(self):
        """Test search with both course and lesson filters"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)
        
        mock_results = SearchResults(
            documents=["Specific result"],
            metadata=[{"course_title": "AI Course", "lesson_number": 5}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        
        result = tool.execute(
            query="neural networks",
            course_name="AI",
            lesson_number=5
        )
        
        mock_vector_store.search.assert_called_once_with(
            query="neural networks",
            course_name="AI",
            lesson_number=5
        )
        
        assert "[AI Course - Lesson 5]" in result
        assert "Specific result" in result
    
    def test_execute_empty_results(self):
        """Test search with empty results"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)
        
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = mock_results
        
        result = tool.execute(query="nonexistent")
        
        assert result == "No relevant content found."
        assert tool.last_sources == []
    
    def test_execute_empty_results_with_filters(self):
        """Test search with empty results and filters"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)
        
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = mock_results
        
        result = tool.execute(
            query="nonexistent",
            course_name="Physics",
            lesson_number=10
        )
        
        assert result == "No relevant content found in course 'Physics' in lesson 10."
    
    def test_execute_with_error(self):
        """Test search when vector store returns error"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)
        
        mock_results = SearchResults.empty("Database connection error")
        mock_vector_store.search.return_value = mock_results
        
        result = tool.execute(query="test")
        
        assert result == "Database connection error"
        assert tool.last_sources == []
    
    def test_format_results_with_missing_metadata(self):
        """Test formatting results with incomplete metadata"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)
        
        mock_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course 1"},  # Missing lesson_number
                {}  # Missing all metadata
            ],
            distances=[0.1, 0.2]
        )
        
        formatted = tool._format_results(mock_results)
        
        assert "[Course 1]" in formatted
        assert "[unknown]" in formatted
        assert "Content 1" in formatted
        assert "Content 2" in formatted
        assert tool.last_sources == ["Course 1", "unknown"]


class TestToolManager:
    """Test cases for ToolManager class"""
    
    def test_init(self):
        """Test ToolManager initialization"""
        manager = ToolManager()
        assert manager.tools == {}
    
    def test_register_tool(self):
        """Test registering a tool"""
        manager = ToolManager()
        tool = MockTool()
        
        manager.register_tool(tool)
        
        assert "mock_tool" in manager.tools
        assert manager.tools["mock_tool"] == tool
    
    def test_register_tool_without_name(self):
        """Test registering a tool without a name should raise error"""
        manager = ToolManager()
        
        # Create a tool without a name
        class BadTool(Tool):
            def get_tool_definition(self):
                return {"description": "Bad tool"}
            def execute(self, **kwargs):
                return "bad"
        
        with pytest.raises(ValueError, match="Tool must have a 'name' in its definition"):
            manager.register_tool(BadTool())
    
    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        manager = ToolManager()
        tool1 = MockTool()
        
        # Create another mock tool
        class MockTool2(Tool):
            def get_tool_definition(self):
                return {
                    "name": "mock_tool_2",
                    "description": "Another mock tool",
                    "input_schema": {"type": "object", "properties": {}}
                }
            def execute(self, **kwargs):
                return "Mock tool 2 executed"
        
        tool2 = MockTool2()
        
        manager.register_tool(tool1)
        manager.register_tool(tool2)
        
        definitions = manager.get_tool_definitions()
        
        assert len(definitions) == 2
        assert definitions[0]["name"] == "mock_tool"
        assert definitions[1]["name"] == "mock_tool_2"
    
    def test_execute_tool(self):
        """Test executing a tool by name"""
        manager = ToolManager()
        tool = MockTool()
        
        manager.register_tool(tool)
        
        result = manager.execute_tool("mock_tool", param1="test")
        
        assert result == "Mock tool executed with: {'param1': 'test'}"
    
    def test_execute_nonexistent_tool(self):
        """Test executing a nonexistent tool"""
        manager = ToolManager()
        
        result = manager.execute_tool("nonexistent_tool")
        
        assert result == "Tool 'nonexistent_tool' not found"
    
    def test_get_last_sources_no_tools(self):
        """Test getting last sources when no tools have sources"""
        manager = ToolManager()
        
        sources = manager.get_last_sources()
        
        assert sources == []
    
    def test_get_last_sources_with_search_tool(self):
        """Test getting last sources from search tool"""
        manager = ToolManager()
        
        # Mock vector store
        mock_vector_store = Mock()
        search_tool = CourseSearchTool(mock_vector_store)
        
        # Set up some sources
        search_tool.last_sources = ["Course 1 - Lesson 1", "Course 2"]
        
        manager.register_tool(search_tool)
        
        sources = manager.get_last_sources()
        
        assert sources == ["Course 1 - Lesson 1", "Course 2"]
    
    def test_get_last_sources_multiple_tools(self):
        """Test getting last sources when multiple tools have sources"""
        manager = ToolManager()
        
        # Create a custom tool that also tracks sources
        class SourceTool(Tool):
            def __init__(self):
                self.last_sources = ["Custom source"]
            
            def get_tool_definition(self):
                return {"name": "source_tool", "input_schema": {"type": "object"}}
            
            def execute(self, **kwargs):
                return "Custom tool result"
        
        search_tool = CourseSearchTool(Mock())
        search_tool.last_sources = ["Search source"]
        source_tool = SourceTool()
        
        manager.register_tool(search_tool)
        manager.register_tool(source_tool)
        
        sources = manager.get_last_sources()
        
        # Should return the first tool's sources it finds
        assert sources == ["Search source"]
    
    def test_reset_sources(self):
        """Test resetting sources from all tools"""
        manager = ToolManager()
        
        # Mock vector store
        mock_vector_store = Mock()
        search_tool = CourseSearchTool(mock_vector_store)
        search_tool.last_sources = ["Source 1", "Source 2"]
        
        manager.register_tool(search_tool)
        
        # Verify sources exist
        assert len(manager.get_last_sources()) == 2
        
        # Reset sources
        manager.reset_sources()
        
        # Verify sources are cleared
        assert len(manager.get_last_sources()) == 0
        assert len(search_tool.last_sources) == 0
    
    def test_reset_sources_no_sources_attribute(self):
        """Test resetting sources when tool doesn't have last_sources"""
        manager = ToolManager()
        tool = MockTool()  # MockTool doesn't have last_sources
        
        manager.register_tool(tool)
        
        # Should not raise an error
        manager.reset_sources()
    
    def test_multiple_tools_registration(self):
        """Test registering multiple tools of different types"""
        manager = ToolManager()
        
        # Register different types of tools
        search_tool = CourseSearchTool(Mock())
        mock_tool = MockTool()
        
        manager.register_tool(search_tool)
        manager.register_tool(mock_tool)
        
        assert len(manager.tools) == 2
        assert "search_course_content" in manager.tools
        assert "mock_tool" in manager.tools