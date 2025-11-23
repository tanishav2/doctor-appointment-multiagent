#code for the API creation 

from fastapi import FastAPI
from pydantic import BaseModel
from agent import DoctorAppointmentAgent
from langchain_core.messages import HumanMessage
from typing import List, Optional
import os

os.environ.pop("SSL_CERT_FILE", None)

app = FastAPI()

# Define Pydantic model to accept request body
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class UserQuery(BaseModel):
    id_number: int
    message: str  # Current user message
    conversation_history: Optional[List[Message]] = []  # Optional: previous conversation

agent = DoctorAppointmentAgent()

@app.post("/execute")
def execute_agent(user_input: UserQuery):
    app_graph = agent.workflow()
    
    # Build message list from conversation history
    from langchain_core.messages import AIMessage
    
    message_list = []
    
    # Add conversation history if provided
    if user_input.conversation_history:
        for msg in user_input.conversation_history:
            if msg.role == "user":
                message_list.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                message_list.append(AIMessage(content=msg.content))
    
    # Add current user message
    message_list.append(HumanMessage(content=user_input.message))

    query_data = {
        "messages": message_list,
        "id_number": user_input.id_number,
        "next": "",
        "query": "",
        "current_reasoning": "",
        "turns": 0,  # <-- REQUIRED FIELD
        "last_node": "",  # track last node visited
    }

    # recursion limit prevents infinite loops (increased for complex conversations)
    response = app_graph.invoke(query_data, config={"recursion_limit": 30})

    # Extract just the text content from the last AI message
    from langchain_core.messages import AIMessage
    
    messages = response["messages"]
    last_ai_message = None
    
    # Find the last AI message (skip human messages)
    for msg in reversed(messages):
        # Check if it's an AIMessage object
        if isinstance(msg, AIMessage):
            last_ai_message = msg.content
            break
        # Check if it's a dict with type 'ai'
        elif isinstance(msg, dict) and msg.get('type') == 'ai':
            last_ai_message = msg.get('content', '')
            break
        # Check if it has type attribute
        elif hasattr(msg, 'type') and getattr(msg, 'type', None) == 'ai':
            last_ai_message = getattr(msg, 'content', str(msg))
            break
    
    # If no AI message found, try to get any message content
    if last_ai_message is None and len(messages) > 0:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage):
            last_ai_message = last_msg.content
        elif hasattr(last_msg, 'content'):
            last_ai_message = last_msg.content
        elif isinstance(last_msg, dict):
            last_ai_message = last_msg.get('content', '')
        else:
            last_ai_message = str(last_msg)
    
    # Return response with conversation history for next turn
    conversation_history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            conversation_history.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, dict):
            if msg.get('type') == 'human':
                conversation_history.append({"role": "user", "content": msg.get('content', '')})
            elif msg.get('type') == 'ai':
                conversation_history.append({"role": "assistant", "content": msg.get('content', '')})
    
    # Return simple text response with conversation history
    return {
        "response": last_ai_message if last_ai_message else "No response generated.",
        "conversation_history": conversation_history,
        "status": "success"
    }
