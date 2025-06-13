# Pizza Ordering Chatbot Web Interface Plan

## Overview

This plan outlines the steps to create a web-based chat interface for the pizza ordering LangGraph agent, accessible via localhost.

## Architecture

### Technology Stack

- **Backend**: FastAPI (Python)
  - Async support for handling multiple chat sessions
  - WebSocket support for real-time communication
  - Easy integration with existing LangGraph code
- **Frontend**: Vanilla HTML/CSS/JavaScript
  - Simple and lightweight
  - No build process required
  - WebSocket client for real-time chat

### System Architecture

```
┌─────────────────┐     WebSocket      ┌──────────────────┐
│                 │◄──────────────────►│                  │
│  Web Browser    │                    │  FastAPI Server  │
│  (Chat UI)      │     HTTP/REST      │  (localhost:8000)│
│                 │◄──────────────────►│                  │
└─────────────────┘                    └──────────────────┘
                                               │
                                               ▼
                                       ┌──────────────────┐
                                       │  LangGraph Agent │
                                       │  (Pizza Bot)     │
                                       └──────────────────┘
```

## Implementation Plan

### Phase 1: Backend Setup with Abstraction Layer

#### 1. Create FastAPI Server Structure with Clean Architecture

```
src/
├── web/
│   ├── __init__.py
│   ├── app.py              # FastAPI application
│   ├── websocket.py        # WebSocket handlers
│   ├── models.py           # Pydantic models for requests/responses
│   ├── agents/             # Agent abstraction layer
│   │   ├── __init__.py
│   │   ├── base.py         # Abstract base class for agents
│   │   ├── pizza_agent.py  # Pizza bot implementation
│   │   └── mock_agent.py   # Mock agent for testing
│   ├── config.py           # Configuration management
│   └── static/             # Static files for frontend
│       ├── index.html
│       ├── style.css
│       └── script.js
```

#### 2. FastAPI Application (`app.py`)

```python
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

# Root endpoint serves the chat interface
@app.get("/")
async def get_chat_interface():
    return HTMLResponse(open("src/web/static/index.html").read())

# WebSocket endpoint for chat
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    # Handle WebSocket connection
    pass
```

#### 3. Agent Abstraction Layer (`agents/base.py`)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ChatAgent(ABC):
    """Abstract base class for chat agents"""

    @abstractmethod
    async def process_message(self, message: str, session_id: str) -> str:
        """Process a user message and return agent response"""
        pass

    @abstractmethod
    async def reset_session(self, session_id: str) -> None:
        """Reset the conversation for a session"""
        pass

    @abstractmethod
    async def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get current state of a session"""
        pass
```

#### 4. Pizza Agent Implementation (`agents/pizza_agent.py`)

```python
from typing import Dict, Any
from .base import ChatAgent
from src.agent.graph import graph
from src.agent.state import PizzaState

class PizzaAgent(ChatAgent):
    def __init__(self):
        self.sessions: Dict[str, PizzaState] = {}

    async def process_message(self, message: str, session_id: str) -> str:
        # Get or create session state
        if session_id not in self.sessions:
            self.sessions[session_id] = PizzaState()

        state = self.sessions[session_id]

        # Add user message to messages
        state.messages.append({
            "role": "caller",
            "content": message
        })

        # Run the graph
        result = await graph.ainvoke(state)

        # Update session state
        self.sessions[session_id] = result

        # Extract bot response
        if result.messages:
            return result.messages[-1]["content"]
        return "I'm sorry, I couldn't process that request."

    async def reset_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            del self.sessions[session_id]

    async def get_session_state(self, session_id: str) -> Dict[str, Any]:
        if session_id in self.sessions:
            state = self.sessions[session_id]
            return {
                "pizzas": state.pizzas,
                "messages_length": len(state.messages),
                "has_errors": len(state.errors) > 0
            }
        return {}
```

#### 5. Mock Agent for Testing (`agents/mock_agent.py`)

```python
from typing import Dict, List
from .base import ChatAgent

class MockAgent(ChatAgent):
    """Mock agent for testing the web interface independently"""

    def __init__(self):
        self.sessions: Dict[str, List[Dict]] = {}
        self.responses = [
            "Welcome to Pizza Bot! What kind of pizza would you like?",
            "Great choice! What size would you like - small, medium, or large?",
            "Perfect! Would you like thin, classic, or stuffed crust?",
            "Your order is complete! You've ordered a {size} {crust} crust pizza.",
            "Is there anything else you'd like to add?"
        ]

    async def process_message(self, message: str, session_id: str) -> str:
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        conversation = self.sessions[session_id]
        conversation.append({"role": "user", "content": message})

        # Simple response logic
        response_index = min(len(conversation) - 1, len(self.responses) - 1)
        response = self.responses[response_index]

        # Extract size and crust from conversation for final message
        if response_index == 3:
            size = "medium"
            crust = "thin"
            for msg in conversation:
                if "small" in msg["content"].lower():
                    size = "small"
                elif "large" in msg["content"].lower():
                    size = "large"
                if "classic" in msg["content"].lower():
                    crust = "classic"
                elif "stuffed" in msg["content"].lower():
                    crust = "stuffed"
            response = response.format(size=size, crust=crust)

        conversation.append({"role": "bot", "content": response})
        return response

    async def reset_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            del self.sessions[session_id]

    async def get_session_state(self, session_id: str) -> Dict[str, Any]:
        return {
            "messages_length": len(self.sessions.get(session_id, [])),
            "is_mock": True
        }
```

#### 6. Configuration Management (`config.py`)

```python
import os
from enum import Enum

class AgentType(Enum):
    PIZZA = "pizza"
    MOCK = "mock"

class Config:
    # Agent configuration
    AGENT_TYPE = AgentType(os.getenv("AGENT_TYPE", "pizza"))

    # Server configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    # Development settings
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    RELOAD = os.getenv("RELOAD", "true").lower() == "true"

    # API Keys (for real agent)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

config = Config()
```

#### 7. WebSocket Handler with Dependency Injection (`websocket.py`)

```python
from typing import Dict
from fastapi import WebSocket, WebSocketDisconnect
from .agents.base import ChatAgent
from .agents.pizza_agent import PizzaAgent
from .agents.mock_agent import MockAgent
from .config import config, AgentType
import json
import logging

logger = logging.getLogger(__name__)

class ChatManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.agent = self._create_agent()

    def _create_agent(self) -> ChatAgent:
        """Factory method to create the appropriate agent"""
        if config.AGENT_TYPE == AgentType.MOCK:
            logger.info("Using MockAgent for testing")
            return MockAgent()
        elif config.AGENT_TYPE == AgentType.PIZZA:
            logger.info("Using PizzaAgent")
            return PizzaAgent()
        else:
            raise ValueError(f"Unknown agent type: {config.AGENT_TYPE}")

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def handle_message(self, client_id: str, message: str):
        try:
            response = await self.agent.process_message(message, client_id)
            await self.send_message(client_id, response)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.send_error(client_id, "Sorry, I encountered an error.")

    async def send_message(self, client_id: str, message: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json({
                "type": "message",
                "message": message
            })

    async def send_error(self, client_id: str, error: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json({
                "type": "error",
                "message": error
            })

# Global chat manager instance
chat_manager = ChatManager()
```

### Phase 2: Frontend Implementation

#### 1. HTML Structure (`index.html`)

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Pizza Order Bot</title>
    <link rel="stylesheet" href="/static/style.css" />
  </head>
  <body>
    <div class="chat-container">
      <div class="chat-header">
        <h2>🍕 Pizza Order Assistant</h2>
      </div>
      <div class="chat-messages" id="messages">
        <!-- Messages will be inserted here -->
      </div>
      <div class="chat-input">
        <input type="text" id="messageInput" placeholder="Type your order..." />
        <button id="sendButton">Send</button>
      </div>
    </div>
    <script src="/static/script.js"></script>
  </body>
</html>
```

#### 2. CSS Styling (`style.css`)

```css
.chat-container {
  max-width: 600px;
  margin: 50px auto;
  border: 1px solid #ddd;
  border-radius: 10px;
  overflow: hidden;
}

.chat-header {
  background-color: #ff6b6b;
  color: white;
  padding: 15px;
  text-align: center;
}

.chat-messages {
  height: 400px;
  overflow-y: auto;
  padding: 20px;
  background-color: #f9f9f9;
}

.message {
  margin-bottom: 15px;
  padding: 10px;
  border-radius: 5px;
}

.user-message {
  background-color: #e3f2fd;
  text-align: right;
}

.bot-message {
  background-color: #f5f5f5;
  text-align: left;
}

.chat-input {
  display: flex;
  padding: 15px;
  background-color: white;
  border-top: 1px solid #ddd;
}

#messageInput {
  flex: 1;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 5px;
  margin-right: 10px;
}

#sendButton {
  padding: 10px 20px;
  background-color: #ff6b6b;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}
```

#### 3. JavaScript Client (`script.js`)

```javascript
class ChatClient {
  constructor() {
    this.clientId = this.generateClientId();
    this.ws = null;
    this.connectWebSocket();
    this.setupEventListeners();
  }

  generateClientId() {
    return "client_" + Math.random().toString(36).substr(2, 9);
  }

  connectWebSocket() {
    this.ws = new WebSocket(`ws://localhost:8000/ws/${this.clientId}`);

    this.ws.onopen = () => {
      console.log("Connected to chat server");
      this.addMessage("bot", "Welcome to Pizza Bot! How can I help you today?");
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.addMessage("bot", data.message);
    };

    this.ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      this.addMessage("bot", "Connection error. Please refresh the page.");
    };
  }

  sendMessage(message) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ message }));
      this.addMessage("user", message);
    }
  }

  addMessage(type, content) {
    const messagesDiv = document.getElementById("messages");
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${type}-message`;
    messageDiv.textContent = content;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }

  setupEventListeners() {
    const input = document.getElementById("messageInput");
    const button = document.getElementById("sendButton");

    button.onclick = () => {
      const message = input.value.trim();
      if (message) {
        this.sendMessage(message);
        input.value = "";
      }
    };

    input.onkeypress = (e) => {
      if (e.key === "Enter") {
        button.click();
      }
    };
  }
}

// Initialize chat client when page loads
window.onload = () => {
  new ChatClient();
};
```

### Phase 3: Integration & Enhancement

#### 1. Complete WebSocket Handler

- Integrate with LangGraph agent
- Handle session state management
- Implement error handling

#### 2. Add Features

- Typing indicators
- Message history persistence
- Order summary display
- Reset conversation button

#### 3. Deployment Script

```bash
#!/bin/bash
# run_chatbot.sh

# Install dependencies
pip install fastapi uvicorn websockets

# Set PYTHONPATH
export PYTHONPATH=.

# Run the server
uvicorn src.web.app:app --reload --host 0.0.0.0 --port 8000
```

## Testing Plan

### 1. Independent Testing of Web Interface

#### Test with Mock Agent

```bash
# Run web interface with mock agent (no LangGraph dependencies)
AGENT_TYPE=mock python -m uvicorn src.web.app:app --reload

# Run automated tests
pytest tests/web/test_websocket.py
pytest tests/web/test_mock_agent.py
```

#### Unit Tests for Web Components

```python
# tests/web/test_mock_agent.py
import pytest
from src.web.agents.mock_agent import MockAgent

@pytest.mark.asyncio
async def test_mock_agent_messages():
    agent = MockAgent()

    # Test first message
    response1 = await agent.process_message("Hi", "session1")
    assert "Welcome" in response1

    # Test follow-up
    response2 = await agent.process_message("Large pizza", "session1")
    assert "size" in response2.lower()

# tests/web/test_agent_interface.py
from src.web.agents.base import ChatAgent

def test_agent_interface():
    # Ensure all agents implement the interface
    assert hasattr(ChatAgent, 'process_message')
    assert hasattr(ChatAgent, 'reset_session')
    assert hasattr(ChatAgent, 'get_session_state')
```

### 2. Testing the Real Agent

```bash
# Run with real agent
AGENT_TYPE=pizza GOOGLE_API_KEY=your_key python -m uvicorn src.web.app:app --reload

# Integration tests
pytest tests/integration/test_pizza_agent_web.py
```

### 3. Load Testing

```python
# tests/web/test_load.py
import asyncio
import websockets
import json

async def stress_test():
    # Create multiple concurrent connections
    tasks = []
    for i in range(50):
        task = simulate_user(f"user_{i}")
        tasks.append(task)

    await asyncio.gather(*tasks)

async def simulate_user(user_id):
    uri = f"ws://localhost:8000/ws/{user_id}"
    async with websockets.connect(uri) as websocket:
        # Send messages
        await websocket.send(json.dumps({"message": "I want a pizza"}))
        response = await websocket.recv()
        print(f"{user_id}: {response}")
```

## Portability Features

### 1. Agent Agnostic Design

The web interface is completely decoupled from the pizza agent implementation:

- **Abstract Interface**: All agents implement the `ChatAgent` base class
- **Dependency Injection**: Agent type is determined at runtime via configuration
- **No Direct Imports**: Web layer never imports from `src.agent` directly

### 2. Easy Agent Swapping

Create new agents by implementing the `ChatAgent` interface:

```python
# Example: Weather Bot Agent
from .base import ChatAgent
import aiohttp

class WeatherAgent(ChatAgent):
    async def process_message(self, message: str, session_id: str) -> str:
        # Parse location from message
        location = self.extract_location(message)

        # Call weather API
        weather = await self.get_weather(location)

        return f"The weather in {location} is {weather}"

    # ... implement other required methods
```

### 3. Configuration-Based Agent Selection

```bash
# Run with different agents without code changes
AGENT_TYPE=mock python -m uvicorn src.web.app:app      # Mock agent
AGENT_TYPE=pizza python -m uvicorn src.web.app:app     # Pizza agent
AGENT_TYPE=weather python -m uvicorn src.web.app:app   # Weather agent
```

### 4. Docker Support for Portability

```dockerfile
# Dockerfile.web
FROM python:3.11-slim

WORKDIR /app

# Install only web dependencies
COPY requirements-web.txt .
RUN pip install -r requirements-web.txt

# Copy only web code
COPY src/web ./src/web

# Default to mock agent
ENV AGENT_TYPE=mock

EXPOSE 8000
CMD ["uvicorn", "src.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# Dockerfile.full
FROM python:3.11-slim

WORKDIR /app

# Install all dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all code
COPY . .

# Use real agent
ENV AGENT_TYPE=pizza

EXPOSE 8000
CMD ["uvicorn", "src.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5. Separate Requirements Files

```txt
# requirements-web.txt (minimal dependencies)
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
pydantic==2.5.0

# requirements.txt (full dependencies)
-r requirements-web.txt
langgraph==0.0.26
langchain==0.1.0
google-generativeai==0.3.0
```

## Security Considerations

1. **Input Validation**

   - Sanitize user inputs
   - Limit message length
   - Rate limiting

2. **Session Management**
   - Use secure session IDs
   - Implement session timeout
   - Clean up inactive sessions

## Deployment Instructions

1. Install dependencies:

   ```bash
   pip install fastapi uvicorn websockets
   ```

2. Run the server:

   ```bash
   chmod +x run_chatbot.sh
   ./run_chatbot.sh
   ```

3. Access the chatbot:
   ```
   Open browser to http://localhost:8000
   ```

## Future Enhancements

1. **UI Improvements**

   - Add order progress visualization
   - Show pizza images
   - Add voice input support

2. **Backend Features**

   - Save order history
   - User authentication
   - Multi-language support

3. **Deployment**
   - Dockerize the application
   - Add HTTPS support
   - Deploy to cloud service

## Summary

This plan provides a complete roadmap for creating a web-based chat interface with the following key benefits:

### Independent Testing & Development

- **Mock Agent**: Develop and test the web interface without any LangGraph dependencies
- **Unit Testing**: Test web components in isolation
- **Load Testing**: Verify performance without the complexity of the real agent

### High Portability

- **Agent Abstraction**: Clean separation between web layer and agent implementation
- **Runtime Configuration**: Switch agents via environment variables
- **Docker Support**: Separate containers for web-only and full deployments
- **Minimal Dependencies**: Web interface can run with just FastAPI and WebSocket libraries

### Easy Integration

- **Standard Interface**: Any agent implementing the `ChatAgent` interface can be plugged in
- **No Code Changes**: Switch between agents without modifying the web layer
- **Multiple Deployment Options**: Run as monolith or separate services

This architecture ensures that:

1. The web interface can be developed and tested independently
2. The pizza agent remains portable and can be used in other contexts
3. New agents can be easily integrated without modifying the web layer
4. The system can scale from development (mock) to production (real agent) seamlessly
