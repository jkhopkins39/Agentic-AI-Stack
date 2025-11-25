"""Streaming utilities for agent responses. Sends text as it's generated, not waiting for full response (ChatGPT style).
Non-blocking operation, real time updates, completion tracking. The aim is to massively reduce latency using asyncronous operations."""
import asyncio
from typing import AsyncIterator, Optional
from aiokafka import AIOKafkaProducer


async def stream_agent_response(
    agent_type: str,
    session_id: str,
    conversation_id: Optional[str],
    correlation_id: Optional[str],
    producer: Optional[AIOKafkaProducer],
    response_topic: str,
    stream: AsyncIterator[str]
):
    f"""
    Stream agent response chunks to Kafka and return full response.
    
    Args:
        agent_type: [TYPE]_AGENT
        session_id: Sesh ID
        conversation_id: Convo ID
        correlation_id: Correlation ID for tracking
        producer: Kafka producer
        response_topic: Topic to publish responses to
        stream: Async iterator yielding response chunks
    
    Returns:
        Full response text
    """

    # Var inits
    full_response = ""
    chunk_count = 0
    
    async for chunk in stream:
        if not chunk:
            continue
        
        full_response += chunk
        chunk_count += 1
        
        # Publish chunk to Kafka for streaming
        if producer:
            try:
                chunk_event = {
                    "session_id": session_id,
                    "conversation_id": conversation_id,
                    "agent_type": agent_type,
                    "message": chunk,
                    "status": "streaming" if chunk_count == 1 else "chunk",
                    "chunk_index": chunk_count,
                    "correlation_id": correlation_id,
                    "timestamp": int(asyncio.get_event_loop().time() * 1000)
                }
                
                await producer.send_and_wait(
                    response_topic,
                    value=chunk_event,
                    key=session_id
                )
            except Exception as e:
                print(f"!!! Error publishing stream chunk: {e}")
    
    # Publish completion event
    if producer and full_response:
        try:
            completion_event = {
                "session_id": session_id,
                "conversation_id": conversation_id,
                "agent_type": agent_type,
                "message": "",  # Empty message indicates completion
                "status": "completed",
                "full_response": full_response,
                "correlation_id": correlation_id,
                "timestamp": int(asyncio.get_event_loop().time() * 1000)
            }
            
            await producer.send_and_wait(
                response_topic,
                value=completion_event,
                key=session_id
            )
        except Exception as e:
            print(f"!!! Error publishing completion event: {e}")
    
    return full_response

