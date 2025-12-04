"""
Single-Agent Baseline - Completely self-contained
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import os
import uuid
import time
import asyncio
from typing import Dict, Any, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Import LLM directly
from langchain.chat_models import init_chat_model

# Import database functions ONLY
from database import (
    lookup_user_by_email,
    lookup_orders_by_email,
    save_query_to_conversation
)

load_dotenv()

# Define MessageClassifier HERE (don't import from agents)
class MessageClassifier(BaseModel):
    message_type: Literal["Order", "Email", "Policy", "Message", "Change Information"] = Field(
        ...,
        description="Classify the user message type"
    )

# Initialize LLM - SAME config as multi-agent
CLAUDE_API_MODE = os.getenv("CLAUDE_API_MODE", "real")
MOCK_API_URL = os.getenv("MOCK_CLAUDE_URL", "http://localhost:8001")

if CLAUDE_API_MODE == "mock":
    llm = init_chat_model(
        "anthropic:claude-3-haiku-20240307",
        base_url=f"{MOCK_API_URL}/v1"
    )
else:
    llm = init_chat_model("anthropic:claude-3-haiku-20240307")

SESSION_EMAIL = os.getenv('USER_EMAIL', 'agenticstack@commerceconductor.com')


async def handle_single_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Single handler that does everything."""
    last_message = state["messages"][-1]
    user_message = last_message.get("content") if isinstance(last_message, dict) else last_message.content
    
    # Classify
    classifier_llm = llm.with_structured_output(MessageClassifier)
    
    classification = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as one of the following:
            - 'Order': if the user asks about specific order information
            - 'Email': if the user explicitly mentions email
            - 'Policy': if the user asks about returns, refunds, policies
            - 'Change Information': if the user requests to change personal information
            - 'Message': if the user asks a general question
            """
        },
        {"role": "user", "content": user_message}
    ])
    
    message_type = classification.message_type
    state["message_type"] = message_type
    
    # Get context
    user_email = state.get("user_email") or SESSION_EMAIL
    try:
        user_data = lookup_user_by_email(user_email) if user_email else None
    except:
        user_data = None
    

    
    # Get order context
    orders_info = ""
    if message_type == "Order" and user_email:
        try:
            orders = lookup_orders_by_email(user_email, limit=5)
        except:
            orders = []
    
    # Generate response
    system_prompt = f"""You are a customer service assistant.

MESSAGE TYPE: {message_type}

CUSTOMER INFO:
- Email: {user_email or "Not provided"}

ORDER HISTORY:
{orders_info if orders_info else "No orders available"}

Handle this {message_type} request. Be helpful and specific.
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    reply = await llm.ainvoke(messages)
    response = reply.content
    
    return {
        "messages": [{"role": "assistant", "content": response}],
        "message_type": message_type
    }


async def baseline_consumer(producer):
    """Single-agent baseline consumer."""
    from aiokafka import AIOKafkaConsumer
    import json
    
    # Get kafka config without importing from main
    KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(',')
    KAFKA_SECURITY_PROTOCOL = os.getenv('KAFKA_SECURITY_PROTOCOL', 'PLAINTEXT')
    
    kafka_config = {'bootstrap_servers': KAFKA_BOOTSTRAP_SERVERS}
    
    if KAFKA_SECURITY_PROTOCOL == 'SASL_SSL':
        import ssl
        ssl_context = ssl.create_default_context()
        kafka_config.update({
            'security_protocol': 'SASL_SSL',
            'sasl_mechanism': os.getenv('KAFKA_SASL_MECHANISM', 'PLAIN'),
            'sasl_plain_username': os.getenv('KAFKA_SASL_USERNAME', ''),
            'sasl_plain_password': os.getenv('KAFKA_SASL_PASSWORD', ''),
            'ssl_context': ssl_context,
        })
    
    consumer = AIOKafkaConsumer(
        'system.ingress',
        group_id='baseline-single-agent-group',
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        session_timeout_ms=30000,
        request_timeout_ms=30000,
        **kafka_config
    )
    
    try:
        await consumer.start()
        print("🟡 BASELINE SINGLE-AGENT CONSUMER STARTED")
        
        while not consumer.assignment():
            await asyncio.sleep(0.5)
        
        print(f"✅ Baseline assigned: {consumer.assignment()}")
        
        async for message in consumer:
            event = message.value
            session_id = event.get("session_id", "unknown")
            
            try:
                user_email = event.get("user_email")
                user_id = event.get("user_id")
                
                state = {
                    "messages": [{"role": "user", "content": event.get("query_text", "")}],
                    "message_type": None,
                    "user_email": user_email,
                    "user_id": user_id,
                    "conversation_id": event.get("conversation_id", str(uuid.uuid4())),
                    "session_id": session_id,
                }
                
                result = await handle_single_agent(state)
                
                response_message = result.get("messages", [{}])[-1].get("content", "")
                message_type = result.get("message_type", "Unknown")
                
                print(f"→ Baseline: {session_id} as {message_type}")
                
                # Publish response
                response_event = {
                    "session_id": session_id,
                    "conversation_id": event.get("conversation_id"),
                    "agent_type": "BASELINE_SINGLE_AGENT",
                    "message": response_message,
                    "status": "completed",
                    "priority": 2,
                    "timestamp": int(time.time() * 1000),
                    "correlation_id": event.get("correlation_id", f"corr-{uuid.uuid4()}"),
                    "classified_as": message_type
                }
                
                await producer.send_and_wait(
                    'agent.responses',
                    value=response_event,
                    key=session_id
                )
                
            except Exception as e:
                print(f"✗ Baseline error: {e}")
                import traceback
                traceback.print_exc()
    
    finally:
        await consumer.stop()
        print("🔴 Baseline consumer stopped")
# ============================================================================
# INTEGRATION WITH MAIN.PY
# ============================================================================

