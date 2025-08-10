import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test cases for AIGenerator class"""
    
    @patch('anthropic.Anthropic')
    def test_init(self, mock_anthropic):
        """Test AIGenerator initialization"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        
        assert generator.model == "claude-3-sonnet"
        mock_anthropic.assert_called_once_with(api_key="test_key")
        assert generator.base_params["model"] == "claude-3-sonnet"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
    
    @patch('anthropic.Anthropic')
    def test_generate_response_direct(self, mock_anthropic):
        """Test generating a direct response without tools"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock API response
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Direct AI response")]
        mock_client.messages.create.return_value = mock_response
        
        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        response = generator.generate_response("What is AI?")
        
        assert response == "Direct AI response"
        
        # Verify API call
        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["messages"][0]["content"] == "What is AI?"
        assert call_args.kwargs["system"] == generator.SYSTEM_PROMPT
        assert "tools" not in call_args.kwargs
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_history(self, mock_anthropic):
        """Test generating response with conversation history"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response with context")]
        mock_client.messages.create.return_value = mock_response
        
        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        history = "User: Hello\nAI: Hi there!"
        response = generator.generate_response("How are you?", conversation_history=history)
        
        assert response == "Response with context"
        
        # Verify system prompt includes history
        call_args = mock_client.messages.create.call_args
        system_content = call_args.kwargs["system"]
        assert "Previous conversation:" in system_content
        assert history in system_content
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tools(self, mock_anthropic):
        """Test generating response with available tools"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response without tool use")]
        mock_client.messages.create.return_value = mock_response
        
        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        
        # Mock tools
        tools = [{
            "name": "search",
            "description": "Search for information",
            "input_schema": {"type": "object", "properties": {}}
        }]
        
        response = generator.generate_response("Search for something", tools=tools)
        
        assert response == "Response without tool use"
        
        # Verify tools were included
        call_args = mock_client.messages.create.call_args
        assert "tools" in call_args.kwargs
        assert call_args.kwargs["tools"] == tools
        assert call_args.kwargs["tool_choice"] == {"type": "auto"}
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tool_execution(self, mock_anthropic):
        """Test generating response that requires tool execution"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock initial response with tool use
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search"
        mock_tool_use.id = "tool_123"
        mock_tool_use.input = {"query": "test search"}
        
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_use]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Tool execution result")]
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        
        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results found"
        
        tools = [{
            "name": "search",
            "description": "Search for information",
            "input_schema": {"type": "object", "properties": {}}
        }]
        
        response = generator.generate_response(
            "Search for test", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        
        assert response == "Tool execution result"
        
        # Verify tool execution
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search", 
            query="test search"
        )
        
        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tool_execution_no_tools(self, mock_anthropic):
        """Test tool execution when no tool manager is provided"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock response with tool use but no tool manager
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search"
        
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_use]
        
        mock_client.messages.create.return_value = mock_initial_response
        
        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        
        tools = [{
            "name": "search",
            "description": "Search for information",
            "input_schema": {"type": "object", "properties": {}}
        }]
        
        # Should return the tool use response directly when no tool manager
        response = generator.generate_response("Search", tools=tools)
        
        # Since there's no tool manager, it should return the text content
        # but since tool_use response doesn't have text, this might be None
        assert response is None or isinstance(response, str)
    
    @patch('anthropic.Anthropic')
    def test_generate_response_multiple_tool_calls(self, mock_anthropic):
        """Test handling multiple tool calls in a single response"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock multiple tool uses
        mock_tool1 = Mock()
        mock_tool1.type = "tool_use"
        mock_tool1.name = "search1"
        mock_tool1.id = "tool_123"
        mock_tool1.input = {"query": "search1"}
        
        mock_tool2 = Mock()
        mock_tool2.type = "tool_use"
        mock_tool2.name = "search2"
        mock_tool2.id = "tool_456"
        mock_tool2.input = {"query": "search2"}
        
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool1, mock_tool2]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Combined results")]
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        
        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        tools = [{
            "name": "search1",
            "description": "Search 1",
            "input_schema": {"type": "object", "properties": {}}
        }, {
            "name": "search2",
            "description": "Search 2",
            "input_schema": {"type": "object", "properties": {}}
        }]
        
        response = generator.generate_response(
            "Multiple searches", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        
        assert response == "Combined results"
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search1", query="search1")
        mock_tool_manager.execute_tool.assert_any_call("search2", query="search2")
    
    @patch('anthropic.Anthropic')
    def test_generate_response_empty_query(self, mock_anthropic):
        """Test generating response with empty query"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="I'm here to help!")]
        mock_client.messages.create.return_value = mock_response
        
        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        response = generator.generate_response("")
        
        assert response == "I'm here to help!"
        
        # Should still make API call even with empty query
        mock_client.messages.create.assert_called_once()
    
    @patch('anthropic.Anthropic')
    def test_generate_response_api_error(self, mock_anthropic):
        """Test handling API errors"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock API error
        mock_client.messages.create.side_effect = Exception("API Error")
        
        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        
        # Should raise the exception
        with pytest.raises(Exception, match="API Error"):
            generator.generate_response("Test query")
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        
        # Check key elements of system prompt
        assert "Search Tool Usage" in generator.SYSTEM_PROMPT
        assert "Response Protocol" in generator.SYSTEM_PROMPT
        assert "Educational" in generator.SYSTEM_PROMPT
        assert "Brief, Concise and focused" in generator.SYSTEM_PROMPT
    
    def test_base_params_configuration(self):
        """Test base API parameters configuration"""
        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
        assert generator.base_params["model"] == "claude-3-sonnet"