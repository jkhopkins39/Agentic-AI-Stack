import os
import psycopg2
from psycopg2.extras import RealDictCursor


def get_database_connection():
    """Get database connection using environment variables"""
    try:
        # Establish connection to our PostgreSQL database (contains user info and orders)
        # Check if we're running in Docker (DATABASE_URL takes precedence)
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            # Parse DATABASE_URL format: postgresql://user:password@host:port/database
            conn = psycopg2.connect(database_url, cursor_factory=RealDictCursor)
        else:
            # Use individual environment variables for local development
            # Defaults match AgenticAIStackDB from Infrastructure-Stack
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME', 'AgenticAIStackDB'),
                user=os.getenv('DB_USER', 'AgenticAIStackDB'),
                password=os.getenv('DB_PASSWORD', '8%w=r?D52Eo2EwcVW:'),  # Default to AgenticAIStackDB password
                cursor_factory=RealDictCursor
            )
        return conn
    except Exception as e:
        # Only print error if in verbose mode or if it's a critical error
        # This prevents spam when database is optional (conversation memory)
        if os.getenv('DEBUG', 'false').lower() == 'true':
            print(f"Database connection error: {e}")
        return None

