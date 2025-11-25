from psycopg2.extras import RealDictCursor
from .connection import get_database_connection
from .pool import get_pooled_connection

def lookup_user_by_email(email: str):
    """Look up user by email address (optimized with connection pooling)"""
    try:
        with get_pooled_connection() as conn:
            if not conn:
                return None
            
            # Create a cursor which is a control structure that enables traversal over records in database
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Define a query to retrieve user information by email
                query = """
                SELECT id, email, first_name, last_name, phone, is_admin, created_at, updated_at
                FROM users 
                WHERE LOWER(email) = LOWER(%s)
                """
                # Use the cursor to execute the query with the provided email parameter
                cursor.execute(query, (email,))

                # Fetch one result from the executed query
                user = cursor.fetchone()
                return dict(user) if user else None
    except Exception as e:
        print(f"Error looking up user: {e}")
        return None


def update_user_information(user_id: str, updates: dict):
    """Update user information in database (optimized with connection pooling)"""
    try:
        with get_pooled_connection() as conn:
            if not conn:
                return False
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Build dynamic update query
                update_fields = []
                values = []
                
                # fields are provided in updates dictionary parameter
                for field, value in updates.items():
                    if value is not None:
                        update_fields.append(f"{field} = %s")
                        values.append(value)
                
                if not update_fields:
                    return False
                
                # Add user_id to values for WHERE clause
                values.append(user_id)
                
                query = f"""
                UPDATE users 
                SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """
                
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
                
    except Exception as e:
        print(f"Error updating user information: {e}")
        return False

