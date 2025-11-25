import uuid
import json
from typing import Optional, Dict, Any, List
from .connection import get_database_connection
from .pool import get_pooled_connection


def create_conversation(session_id: str, user_id: Optional[str] = None, user_email: Optional[str] = None, 
                       conversation_type: str = 'general') -> str:
    """Create a new conversation and return conversation_id (optimized with connection pooling)"""
    try:
        with get_pooled_connection() as conn:
            if not conn:
                return None
            
            with conn.cursor() as cursor:
                conversation_id = str(uuid.uuid4())
                query = """
                INSERT INTO conversations (id, session_id, user_id, user_email, conversation_type, context)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(query, (conversation_id, session_id, user_id, user_email, conversation_type, json.dumps({})))
                conn.commit()
                return conversation_id
    except Exception as e:
        print(f"Error creating conversation: {e}")
        return None


def get_conversation(session_id: str) -> Optional[Dict[str, Any]]:
    """Get conversation by session_id (optimized with connection pooling)"""
    try:
        with get_pooled_connection() as conn:
            if not conn:
                return None
            
            with conn.cursor() as cursor:
                query = """
                SELECT id, session_id, user_id, user_email, status, conversation_type, context, created_at, updated_at
                FROM conversations 
                WHERE session_id = %s AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
                """
                cursor.execute(query, (session_id,))
                conversation = cursor.fetchone()
                return dict(conversation) if conversation else None
    except Exception as e:
        print(f"Error getting conversation: {e}")
        return None


def update_conversation_context(conversation_id: str, context: Dict[str, Any], 
                              user_id: Optional[str] = None, user_email: Optional[str] = None) -> bool:
    """Update conversation context and user information (optimized with connection pooling)"""
    try:
        with get_pooled_connection() as conn:
            if not conn:
                return False
            
            with conn.cursor() as cursor:
                # Build dynamic update query
                update_fields = ["context = %s", "updated_at = CURRENT_TIMESTAMP"]
                values = [json.dumps(context)]
                
                if user_id:
                    update_fields.append("user_id = %s")
                    values.append(user_id)
                    
                if user_email:
                    update_fields.append("user_email = %s")
                    values.append(user_email)
                
                values.append(conversation_id)
                
                query = f"""
                UPDATE conversations 
                SET {', '.join(update_fields)}
                WHERE id = %s
                """
                
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
    except Exception as e:
        print(f"Error updating conversation context: {e}")
        return False


def save_query_to_conversation(conversation_id: str, user_message: str, agent_type: str, 
                              agent_response: str, message_order: int, user_id: Optional[str] = None) -> str:
    """Save a query and response to the conversation (optimized with connection pooling)"""
    try:
        with get_pooled_connection() as conn:
            if not conn:
                return None
            
            with conn.cursor() as cursor:
                query_id = str(uuid.uuid4())
                query = """
                INSERT INTO queries (id, conversation_id, user_id, query_text, agent_type, agent_response, 
                                   message_order, status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """
                cursor.execute(query, (query_id, conversation_id, user_id, user_message, agent_type, 
                                     agent_response, message_order, 'completed'))
                conn.commit()
                return query_id
    except Exception as e:
        print(f"Error saving query to conversation: {e}")
        return None


def get_conversation_history(conversation_id: str) -> List[Dict[str, Any]]:
    """Get conversation history with all queries and responses (optimized with connection pooling)"""
    try:
        with get_pooled_connection() as conn:
            if not conn:
                return []
            
            with conn.cursor() as cursor:
                query = """
                SELECT query_text, agent_type, agent_response, message_order, created_at
                FROM queries 
                WHERE conversation_id = %s
                ORDER BY message_order ASC
                """
                cursor.execute(query, (conversation_id,))
                history = cursor.fetchall()
                return [dict(row) for row in history]
    except Exception as e:
        print(f"Error getting conversation history: {e}")
        return []

