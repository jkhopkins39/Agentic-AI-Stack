"""
Admin Dashboard API Endpoints
Handles admin statistics, sessions, and user management
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from psycopg2.extras import RealDictCursor
from database import get_database_connection, get_conversation_history

router = APIRouter()


@router.get("/api/admin/stats")
async def get_admin_stats():
    """Get admin dashboard statistics"""
    conn = get_database_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get total users
            cursor.execute("SELECT COUNT(*) as count FROM users")
            total_users = cursor.fetchone()['count']
            
            # Get total conversations
            cursor.execute("SELECT COUNT(*) as count FROM conversations")
            total_conversations = cursor.fetchone()['count']
            
            # Get total messages
            cursor.execute("SELECT COUNT(*) as count FROM queries")
            total_messages = cursor.fetchone()['count']
            
            # Get recent messages (last 24 hours)
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM queries 
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            recent_messages = cursor.fetchone()['count']
            
            # Get agent distribution
            cursor.execute("""
                SELECT agent_type, COUNT(*) as count
                FROM queries
                GROUP BY agent_type
                ORDER BY count DESC
            """)
            agent_distribution = [{"agent_type": row['agent_type'], "count": row['count']} for row in cursor.fetchall()]
            
            return {
                "stats": {
                    "total_users": total_users,
                    "total_conversations": total_conversations,
                    "total_messages": total_messages
                },
                "recent_activity": {
                    "recent_messages": recent_messages
                },
                "agent_distribution": agent_distribution
            }
    except Exception as e:
        print(f"Error fetching admin stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.get("/api/admin/sessions")
async def get_admin_sessions(limit: int = 50):
    """Get all conversation sessions for admin"""
    conn = get_database_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT c.id as conversation_id, c.session_id, c.user_id, c.user_email, 
                   u.first_name, u.last_name, u.phone, c.created_at, c.updated_at,
                   (SELECT COUNT(*) FROM queries WHERE conversation_id = c.id) as message_count
            FROM conversations c
            LEFT JOIN users u ON c.user_id = u.id OR (c.user_id IS NULL AND c.user_email = u.email)
            ORDER BY c.updated_at DESC
            LIMIT %s
            """
            cursor.execute(query, (limit,))
            sessions = cursor.fetchall()
            return {"sessions": [dict(session) for session in sessions]}
    except Exception as e:
        print(f"Error fetching admin sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.get("/api/admin/session/{conversation_id}")
async def get_admin_session_details(conversation_id: str):
    """Get detailed session information for admin"""
    conn = get_database_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get conversation info with user details
            cursor.execute("""
                SELECT c.id as conversation_id, c.session_id, c.user_id, c.user_email, 
                       c.created_at, c.updated_at, c.context,
                       u.first_name, u.last_name, u.phone
                FROM conversations c
                LEFT JOIN users u ON c.user_id = u.id
                WHERE c.id = %s
            """, (conversation_id,))
            conversation = cursor.fetchone()
            
            if not conversation:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Get messages
            messages = get_conversation_history(conversation_id)
            
            return {
                "session": dict(conversation),
                "messages": messages
            }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching session details: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.get("/api/admin/users")
async def get_admin_users(limit: int = 100):
    """Get all users for admin"""
    conn = get_database_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT id, email, first_name, last_name, phone, created_at, updated_at,
                   (SELECT COUNT(*) FROM conversations 
                    WHERE user_id = users.id OR (user_id IS NULL AND user_email = users.email)) as session_count,
                   (SELECT COUNT(*) FROM queries q 
                    JOIN conversations c ON q.conversation_id = c.id 
                    WHERE c.user_id = users.id OR (c.user_id IS NULL AND c.user_email = users.email)) as total_messages
            FROM users
            ORDER BY created_at DESC
            LIMIT %s
            """
            cursor.execute(query, (limit,))
            users = cursor.fetchall()
            return {"users": [dict(user) for user in users]}
    except Exception as e:
        print(f"Error fetching admin users: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

