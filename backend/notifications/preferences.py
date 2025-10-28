"""User notification preference management."""
import uuid
from typing import Optional, Dict, Any
from database.connection import get_database_connection


def get_user_notification_preference(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user's notification preferences from database"""
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor() as cursor:
            query = """
            SELECT preferred_method, receipt_notifications, info_change_notifications, order_update_notifications
            FROM user_notification_preferences
            WHERE user_id = %s
            """
            cursor.execute(query, (user_id,))
            preference = cursor.fetchone()
            return dict(preference) if preference else None
    except Exception as e:
        print(f"Error getting user notification preference: {e}")
        return None
    finally:
        conn.close()


def set_user_notification_preference(user_id: str, preferences: dict) -> bool:
    """Set or update user's notification preferences"""
    conn = get_database_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cursor:
            # Check if preference exists
            check_query = "SELECT id FROM user_notification_preferences WHERE user_id = %s"
            cursor.execute(check_query, (user_id,))
            exists = cursor.fetchone()
            
            if exists:
                # Update existing preference
                update_fields = []
                values = []
                
                for field, value in preferences.items():
                    if value is not None and field in ['preferred_method', 'receipt_notifications', 
                                                        'info_change_notifications', 'order_update_notifications']:
                        update_fields.append(f"{field} = %s")
                        values.append(value)
                
                if not update_fields:
                    return False
                
                values.append(user_id)
                query = f"""
                UPDATE user_notification_preferences
                SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = %s
                """
                cursor.execute(query, values)
            else:
                # Insert new preference
                query = """
                INSERT INTO user_notification_preferences 
                (id, user_id, preferred_method, receipt_notifications, info_change_notifications, order_update_notifications)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(query, (
                    str(uuid.uuid4()),
                    user_id,
                    preferences.get('preferred_method', 'email'),
                    preferences.get('receipt_notifications', True),
                    preferences.get('info_change_notifications', True),
                    preferences.get('order_update_notifications', True)
                ))
            
            conn.commit()
            return True
    except Exception as e:
        print(f"Error setting user notification preference: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

