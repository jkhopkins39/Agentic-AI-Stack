import os
import psycopg2
from psycopg2.extras import RealDictCursor


def get_database_connection():
    """Get database connection using environment variables"""
    try:
        # Establish connection to our PostgreSQL database (contains user info and orders)
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'AgenticAIStackDB'),
            user=os.getenv('DB_USER', 'AgenticAIStackDB'),
            password=os.getenv('DB_PASSWORD'),  # Required - must be set in .env file
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

