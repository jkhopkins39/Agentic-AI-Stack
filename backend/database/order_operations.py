from typing import Optional
from .connection import get_database_connection
from utils.validation import validate_email, sanitize_input


def lookup_order_by_number(order_number: str):
    """Look up an order by order number"""
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        # Use cursor to execute query
        with conn.cursor() as cursor:
            query = """
            SELECT o.id, o.order_number, o.status, o.total_amount, o.currency, o.created_at,
                   u.email, u.first_name, u.last_name,
                   array_agg(
                       json_build_object(
                           'product_name', p.name,
                           'quantity', oi.quantity,
                           'unit_price', oi.unit_price,
                           'total_price', oi.quantity * oi.unit_price
                       )
                   ) as items
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON o.id = oi.order_id
            JOIN products p ON oi.product_id = p.id
            WHERE UPPER(o.order_number) = UPPER(%s)
            GROUP BY o.id, o.order_number, o.status, o.total_amount, o.currency, o.created_at,
                     u.email, u.first_name, u.last_name
            """
            cursor.execute(query, (order_number,))
            order = cursor.fetchone()
            return dict(order) if order else None
    except Exception as e:
        print(f"Error looking up order by number: {e}")
        return None
    finally:
        conn.close()


def lookup_orders_by_email(email: str, limit: int = 10):
    """Look up orders by user email, ordered by most recent first"""
    # Validate email format
    is_valid, error_msg = validate_email(email)
    if not is_valid:
        print(f"[VALIDATION] Invalid email format: {error_msg}")
        return []
    
    conn = get_database_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor() as cursor:
            query = """
            SELECT o.id, o.order_number, o.status, o.total_amount, o.currency, o.created_at,
                   u.email, u.first_name, u.last_name,
                   array_agg(
                       json_build_object(
                           'product_name', p.name,
                           'quantity', oi.quantity,
                           'unit_price', oi.unit_price,
                           'total_price', oi.quantity * oi.unit_price
                       )
                   ) as items
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON o.id = oi.order_id
            JOIN products p ON oi.product_id = p.id
            WHERE LOWER(u.email) = LOWER(%s)
            GROUP BY o.id, o.order_number, o.status, o.total_amount, o.currency, o.created_at,
                     u.email, u.first_name, u.last_name
            ORDER BY o.created_at DESC
            LIMIT %s
            """
            cursor.execute(query, (email, limit))
            orders = cursor.fetchall()
            return [dict(order) for order in orders]
    except Exception as e:
        print(f"Error looking up orders by email: {e}")
        return []
    finally:
        conn.close()


def lookup_orders_by_product_name(product_name: str, user_email: Optional[str] = None):
    """Look up order by product name, optionally filtered by user email"""
    # Sanitize product name input to prevent SQL injection
    sanitized_product_name, is_malicious = sanitize_input(product_name, max_length=200)
    
    if is_malicious:
        print(f"[SECURITY] Blocked potentially malicious product name search: {product_name}")
        return []
    
    if not sanitized_product_name or len(sanitized_product_name.strip()) == 0:
        print(f"[VALIDATION] Empty product name after sanitization")
        return []
    
    # Validate email if provided
    if user_email:
        is_valid, error_msg = validate_email(user_email)
        if not is_valid:
            print(f"[VALIDATION] Invalid email in product search: {error_msg}")
            return []
    
    conn = get_database_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor() as cursor:
            base_query = """
            SELECT DISTINCT o.id, o.order_number, o.status, o.total_amount, o.currency, o.created_at,
                   u.email, u.first_name, u.last_name,
                   array_agg(
                       json_build_object(
                           'product_name', p2.name,
                           'quantity', oi2.quantity,
                           'unit_price', oi2.unit_price,
                           'total_price', oi2.quantity * oi2.unit_price
                       )
                   ) as items
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON o.id = oi.order_id
            JOIN products p ON oi.product_id = p.id
            JOIN order_items oi2 ON o.id = oi2.order_id
            JOIN products p2 ON oi2.product_id = p2.id
            WHERE LOWER(p.name) LIKE LOWER(%s)
            """
            
            params = [f"%{sanitized_product_name}%"]
            
            if user_email:
                base_query += " AND LOWER(u.email) = LOWER(%s)"
                params.append(user_email)
            
            base_query += """
            GROUP BY o.id, o.order_number, o.status, o.total_amount, o.currency, o.created_at,
                     u.email, u.first_name, u.last_name
            ORDER BY o.created_at DESC
            """
            
            cursor.execute(base_query, params)
            orders = cursor.fetchall()
            
            # If no exact matches, log for potential fuzzy matching improvement
            if len(orders) == 0:
                print(f"[SEARCH] No results for product: {sanitized_product_name}")
            
            return [dict(order) for order in orders]
    except Exception as e:
        print(f"Error looking up orders by product name: {e}")
        return []
    finally:
        conn.close()

