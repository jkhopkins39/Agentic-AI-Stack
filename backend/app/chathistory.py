"""
Chat History API Endpoints
Handles conversation listing, details, and deletion
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any
from psycopg2.extras import RealDictCursor
from database import get_database_connection, get_conversation_history

router = APIRouter()


@router.get("/api/conversations")
async def get_conversations(user_email: Optional[str] = None, limit: int = 20):
    """Get list of conversations for a user"""
    conn = get_database_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            if user_email:
                query = """
                SELECT c.id as conversation_id, c.session_id, c.user_email, c.created_at, c.updated_at,
                       (SELECT query_text FROM queries WHERE conversation_id = c.id ORDER BY message_order DESC LIMIT 1) as last_message,
                       (SELECT COUNT(*) FROM queries WHERE conversation_id = c.id) as message_count
                FROM conversations c
                WHERE c.user_email = %s
                ORDER BY c.updated_at DESC
                LIMIT %s
                """
                cursor.execute(query, (user_email, limit))
            else:
                query = """
                SELECT c.id as conversation_id, c.session_id, c.user_email, c.created_at, c.updated_at,
                       (SELECT query_text FROM queries WHERE conversation_id = c.id ORDER BY message_order DESC LIMIT 1) as last_message,
                       (SELECT COUNT(*) FROM queries WHERE conversation_id = c.id) as message_count
                FROM conversations c
                ORDER BY c.updated_at DESC
                LIMIT %s
                """
                cursor.execute(query, (limit,))
            
            conversations = cursor.fetchall()
            return {"conversations": [dict(conv) for conv in conversations]}
    except Exception as e:
        print(f"Error fetching conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.get("/api/conversations/{conversation_id}")
async def get_conversation_details(conversation_id: str):
    """Get conversation details with all messages"""
    conn = get_database_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get conversation info
            cursor.execute("""
                SELECT id, session_id, user_id, user_email, created_at, updated_at
                FROM conversations
                WHERE id = %s
            """, (conversation_id,))
            conversation = cursor.fetchone()
            
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            # Get messages
            messages = get_conversation_history(conversation_id)
            
            return {
                "conversation": dict(conversation),
                "messages": messages
            }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching conversation details: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, user_email: Optional[str] = None):
    """Delete a conversation and all its messages"""
    conn = get_database_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # First verify the conversation exists and optionally check user_email
            cursor.execute("""
                SELECT id, user_email FROM conversations WHERE id = %s
            """, (conversation_id,))
            conversation = cursor.fetchone()
            
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            # Optional: verify user_email matches if provided
            if user_email and conversation['user_email'] != user_email:
                raise HTTPException(status_code=403, detail="Not authorized to delete this conversation")
            
            # Delete all queries (messages) for this conversation
            cursor.execute("""
                DELETE FROM queries WHERE conversation_id = %s
            """, (conversation_id,))
            
            # Delete the conversation itself
            cursor.execute("""
                DELETE FROM conversations WHERE id = %s
            """, (conversation_id,))
            
            conn.commit()
            
            print(f"✓ Deleted conversation {conversation_id}")
            return {"success": True, "message": "Conversation deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting conversation: {e}")
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

