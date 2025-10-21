"""
Authentication module for user login and session management
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, Dict, Any
from pydantic import BaseModel


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    success: bool
    message: str
    user: Optional[Dict[str, Any]] = None


def get_database_connection():
    """Get database connection using environment variables"""
    try:
        # Try DATABASE_URL first (for Docker)
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            conn = psycopg2.connect(database_url)
        else:
            # Fallback to individual variables
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', '127.0.0.1'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME', 'AgenticAIStackDB'),
                user=os.getenv('DB_USER', 'AgenticAIStackDB'),
                password=os.getenv('DB_PASSWORD')
            )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user with email and password"""
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # For now, we'll use a simple password check
            # In production, you'd hash passwords and compare hashes
            query = """
            SELECT id, email, first_name, last_name, phone, created_at, updated_at
            FROM users 
            WHERE LOWER(email) = LOWER(%s) AND password_hash = %s
            """
            cursor.execute(query, (email, password))
            user = cursor.fetchone()
            return dict(user) if user else None
    except Exception as e:
        print(f"Error authenticating user: {e}")
        return None
    finally:
        conn.close()


def lookup_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Look up user by email address"""
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT id, email, first_name, last_name, phone, created_at, updated_at
            FROM users 
            WHERE LOWER(email) = LOWER(%s)
            """
            cursor.execute(query, (email,))
            user = cursor.fetchone()
            return dict(user) if user else None
    except Exception as e:
        print(f"Error looking up user: {e}")
        return None
    finally:
        conn.close()


def hash_password(password: str) -> str:
    """Hash a password for secure storage"""
    import hashlib
    import secrets
    
    # Generate a random salt
    salt = secrets.token_hex(16)
    
    # Hash the password with the salt
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    
    # Return salt + hash as a single string
    return f"{salt}:{password_hash.hex()}"


def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verify a password against its hash"""
    import hashlib
    
    try:
        # Split salt and hash
        salt, password_hash = stored_password.split(':')
        
        # Hash the provided password with the stored salt
        provided_hash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt.encode('utf-8'), 100000)
        
        # Compare hashes
        return provided_hash.hex() == password_hash
    except Exception:
        return False


def create_user_session(user_id: str) -> str:
    """Create a user session and return session token"""
    import uuid
    import time
    
    # Generate a session token
    session_token = str(uuid.uuid4())
    
    # In a real application, you'd store this in Redis or database
    # For now, we'll just return the token
    return session_token


def validate_session(session_token: str) -> Optional[str]:
    """Validate a session token and return user ID"""
    # In a real application, you'd check Redis or database
    # For now, we'll just return None (no session validation)
    return None
