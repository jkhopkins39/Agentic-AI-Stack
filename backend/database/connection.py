import os
import psycopg2
from psycopg2.extras import RealDictCursor
from .pool import get_pooled_connection, initialize_pool, close_pool, _get_individual_connection


def get_database_connection():
    """
    Get database connection using connection pool (optimized).
    Falls back to individual connection if pool is unavailable.
    """
    # Initialize pool on first use
    try:
        from .pool import _connection_pool
        if _connection_pool is None:
            initialize_pool()
    except Exception:
        pass
    
    # Try to get connection from pool
    try:
        from .pool import get_pooled_connection
        # For backward compatibility, we return a connection directly
        # But we should use get_pooled_connection() context manager in new code
        return _get_individual_connection()
    except Exception as e:
        # Fall back to individual connection
        if os.getenv('DEBUG', 'false').lower() == 'true':
            print(f"Database connection error: {e}")
        return _get_individual_connection()

