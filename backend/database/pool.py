"""Database connection pooling for improved performance; Initialize pool on startup, borrow connection
when needed (like a thread pool). The aim here is to reduce temporal overhead of creating and closing
cursors each time. Includes auto cleanup and graceful fallback for pooling fails."""
import os
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional

# Global connection pool
_connection_pool: Optional[pool.ThreadedConnectionPool] = None
_min_connections = 2
_max_connections = 20


def _get_pool_config():
    """Get database configuration for connection pool"""
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        return {'dsn': database_url}
    else:
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'AgenticAIStackDB'),
            'user': os.getenv('DB_USER', 'AgenticAIStackDB'),
            'password': os.getenv('DB_PASSWORD', '8%w=r?D52Eo2EwcVW:'),
        }


def initialize_pool():
    """Initialize the connection pool"""
    global _connection_pool
    
    if _connection_pool is not None:
        return
    
    try:
        config = _get_pool_config()
        
        if 'dsn' in config:
            # Use connection string
            _connection_pool = pool.ThreadedConnectionPool(
                _min_connections,
                _max_connections,
                config['dsn']
            )
        else:
            # Use individual parameters
            _connection_pool = pool.ThreadedConnectionPool(
                _min_connections,
                _max_connections,
                host=config['host'],
                port=config['port'],
                database=config['database'],
                user=config['user'],
                password=config['password']
            )
        
        print(f"Database connection pool initialized ({_min_connections}-{_max_connections} connections)")
    except Exception as e:
        print(f"!!! Failed to initialize connection pool: {e}")
        print("!!! Falling back to individual connections")
        _connection_pool = None


def close_pool():
    """Close all connections in the pool"""
    global _connection_pool
    
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None
        print("Connection pool closed")


@contextmanager
def get_pooled_connection(cursor_factory=RealDictCursor):
    """
    Get a connection from the pool with automatic cleanup.
    Usage:
        with get_pooled_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT ...")
    """
    global _connection_pool
    
    # Initialize pool if not already done
    if _connection_pool is None:
        initialize_pool()
    
    # If pool initialization failed, fall back to individual connection
    if _connection_pool is None:
        conn = _get_individual_connection(cursor_factory)
        try:
            yield conn
        finally:
            if conn:
                conn.close()
        return
    
    # Get connection from pool
    conn = None
    try:
        conn = _connection_pool.getconn()
        if conn:
            # Set cursor factory if needed
            if cursor_factory and not isinstance(conn.cursor().cursor_factory, type(cursor_factory)):
                # Note: cursor_factory is set per cursor, not per connection
                pass
            yield conn
    except pool.PoolError as e:
        print(f"!!! Pool error: {e}")
        # Fall back to individual connection
        conn = _get_individual_connection(cursor_factory)
        try:
            yield conn
        finally:
            if conn:
                conn.close()
    finally:
        if conn and _connection_pool:
            try:
                _connection_pool.putconn(conn)
            except Exception as e:
                print(f"!!! Error returning connection to pool: {e}")
                if conn:
                    conn.close()


def _get_individual_connection(cursor_factory=RealDictCursor):
    """Fallback: get individual connection (non-pooled)"""
    try:
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            return psycopg2.connect(database_url, cursor_factory=cursor_factory)
        else:
            return psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME', 'AgenticAIStackDB'),
                user=os.getenv('DB_USER', 'AgenticAIStackDB'),
                password=os.getenv('DB_PASSWORD', '8%w=r?D52Eo2EwcVW:'),
                cursor_factory=cursor_factory
            )
    except Exception as e:
        if os.getenv('DEBUG', 'false').lower() == 'true':
            print(f"Database connection error: {e}")
        return None

