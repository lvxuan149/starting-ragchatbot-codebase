import pytest
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from session_manager import SessionManager, Message


class TestMessage:
    """Test cases for Message dataclass"""
    
    def test_message_creation(self):
        """Test creating a message"""
        message = Message(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"
    
    def test_message_with_assistant_role(self):
        """Test creating an assistant message"""
        message = Message(role="assistant", content="Hi there!")
        assert message.role == "assistant"
        assert message.content == "Hi there!"


class TestSessionManager:
    """Test cases for SessionManager class"""
    
    def test_init_default(self):
        """Test SessionManager initialization with default values"""
        manager = SessionManager()
        assert manager.max_history == 5
        assert manager.sessions == {}
        assert manager.session_counter == 0
    
    def test_init_custom_max_history(self):
        """Test SessionManager initialization with custom max_history"""
        manager = SessionManager(max_history=10)
        assert manager.max_history == 10
    
    def test_create_session(self):
        """Test creating a new session"""
        manager = SessionManager()
        session_id = manager.create_session()
        
        assert session_id == "session_1"
        assert session_id in manager.sessions
        assert manager.sessions[session_id] == []
        assert manager.session_counter == 1
    
    def test_create_multiple_sessions(self):
        """Test creating multiple sessions"""
        manager = SessionManager()
        
        session1 = manager.create_session()
        session2 = manager.create_session()
        session3 = manager.create_session()
        
        assert session1 == "session_1"
        assert session2 == "session_2"
        assert session3 == "session_3"
        assert manager.session_counter == 3
        assert len(manager.sessions) == 3
    
    def test_add_message_to_existing_session(self):
        """Test adding a message to an existing session"""
        manager = SessionManager()
        session_id = manager.create_session()
        
        manager.add_message(session_id, "user", "Hello")
        
        assert len(manager.sessions[session_id]) == 1
        assert manager.sessions[session_id][0].role == "user"
        assert manager.sessions[session_id][0].content == "Hello"
    
    def test_add_message_to_nonexistent_session(self):
        """Test adding a message to a nonexistent session"""
        manager = SessionManager()
        
        # Should create the session if it doesn't exist
        manager.add_message("new_session", "user", "Hello")
        
        assert "new_session" in manager.sessions
        assert len(manager.sessions["new_session"]) == 1
        assert manager.sessions["new_session"][0].content == "Hello"
    
    def test_add_multiple_messages(self):
        """Test adding multiple messages to a session"""
        manager = SessionManager()
        session_id = manager.create_session()
        
        manager.add_message(session_id, "user", "Question 1")
        manager.add_message(session_id, "assistant", "Answer 1")
        manager.add_message(session_id, "user", "Question 2")
        manager.add_message(session_id, "assistant", "Answer 2")
        
        messages = manager.sessions[session_id]
        assert len(messages) == 4
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[2].role == "user"
        assert messages[3].role == "assistant"
    
    def test_add_exchange(self):
        """Test adding a complete question-answer exchange"""
        manager = SessionManager()
        session_id = manager.create_session()
        
        manager.add_exchange(session_id, "What is AI?", "AI is artificial intelligence.")
        
        messages = manager.sessions[session_id]
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "What is AI?"
        assert messages[1].role == "assistant"
        assert messages[1].content == "AI is artificial intelligence."
    
    def test_max_history_limit(self):
        """Test that conversation history is limited"""
        manager = SessionManager(max_history=2)  # Only keep 2 exchanges (4 messages)
        session_id = manager.create_session()
        
        # Add more messages than the limit
        for i in range(6):  # 3 exchanges = 6 messages
            role = "user" if i % 2 == 0 else "assistant"
            manager.add_message(session_id, role, f"Message {i}")
        
        # Should only keep the last 4 messages
        messages = manager.sessions[session_id]
        assert len(messages) == 4
        assert messages[0].content == "Message 2"
        assert messages[1].content == "Message 3"
        assert messages[2].content == "Message 4"
        assert messages[3].content == "Message 5"
    
    def test_get_conversation_history_existing_session(self):
        """Test getting conversation history for an existing session"""
        manager = SessionManager()
        session_id = manager.create_session()
        
        manager.add_exchange(session_id, "Hello", "Hi there!")
        manager.add_exchange(session_id, "How are you?", "I'm doing well!")
        
        history = manager.get_conversation_history(session_id)
        
        expected = "User: Hello\nAssistant: Hi there!\nUser: How are you?\nAssistant: I'm doing well!"
        assert history == expected
    
    def test_get_conversation_history_empty_session(self):
        """Test getting conversation history for an empty session"""
        manager = SessionManager()
        session_id = manager.create_session()
        
        history = manager.get_conversation_history(session_id)
        
        assert history is None
    
    def test_get_conversation_history_nonexistent_session(self):
        """Test getting conversation history for a nonexistent session"""
        manager = SessionManager()
        
        history = manager.get_conversation_history("nonexistent")
        
        assert history is None
    
    def test_get_conversation_history_none_session_id(self):
        """Test getting conversation history with None session_id"""
        manager = SessionManager()
        
        history = manager.get_conversation_history(None)
        
        assert history is None
    
    def test_clear_session(self):
        """Test clearing a session"""
        manager = SessionManager()
        session_id = manager.create_session()
        
        manager.add_exchange(session_id, "Hello", "Hi!")
        assert len(manager.sessions[session_id]) == 2
        
        manager.clear_session(session_id)
        
        assert len(manager.sessions[session_id]) == 0
    
    def test_clear_nonexistent_session(self):
        """Test clearing a nonexistent session (should not raise error)"""
        manager = SessionManager()
        
        # Should not raise an error
        manager.clear_session("nonexistent")
    
    def test_session_persistence_across_operations(self):
        """Test that sessions persist across various operations"""
        manager = SessionManager()
        
        # Create multiple sessions
        session1 = manager.create_session()
        session2 = manager.create_session()
        
        # Add messages to both
        manager.add_exchange(session1, "Q1", "A1")
        manager.add_exchange(session2, "Q2", "A2")
        
        # Verify both sessions have their data
        assert len(manager.sessions[session1]) == 2
        assert len(manager.sessions[session2]) == 2
        
        # Clear one session
        manager.clear_session(session1)
        
        # Verify other session is unaffected
        assert len(manager.sessions[session1]) == 0
        assert len(manager.sessions[session2]) == 2
    
    def test_message_role_formatting(self):
        """Test that message roles are formatted correctly in history"""
        manager = SessionManager()
        session_id = manager.create_session()
        
        manager.add_message(session_id, "user", "Test message")
        manager.add_message(session_id, "assistant", "Test response")
        
        history = manager.get_conversation_history(session_id)
        
        assert "User: Test message" in history
        assert "Assistant: Test response" in history
    
    def test_large_number_of_sessions(self):
        """Test creating a large number of sessions"""
        manager = SessionManager()
        
        sessions = []
        for i in range(100):
            session_id = manager.create_session()
            sessions.append(session_id)
            manager.add_message(session_id, "user", f"Message {i}")
        
        assert len(sessions) == 100
        assert manager.session_counter == 100
        assert len(manager.sessions) == 100
        
        # Verify last session
        last_session = sessions[-1]
        assert last_session == "session_100"
        assert len(manager.sessions[last_session]) == 1