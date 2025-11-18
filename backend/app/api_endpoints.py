"""
API endpoints for user profile, orders, and authentication
"""

from fastapi import HTTPException
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from .auth import get_database_connection, lookup_user_by_email


# API Response Models
class UserProfile(BaseModel):
    id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    is_admin: bool = False
    created_at: str
    updated_at: str


class UserAddress(BaseModel):
    id: str
    address: str
    city: str
    state: Optional[str] = None
    postal_code: str
    country: str


class OrderItem(BaseModel):
    product_name: str
    quantity: int
    unit_price: float
    total_price: float


class Order(BaseModel):
    id: str
    order_number: str
    status: str
    total_amount: float
    currency: str
    created_at: str
    updated_at: str
    shipped_at: Optional[str] = None
    delivered_at: Optional[str] = None
    items: List[OrderItem] = []


class UserProfileResponse(BaseModel):
    profile: UserProfile
    addresses: List[UserAddress] = []
    total_orders: int = 0
    total_spent: float = 0.0


class OrdersResponse(BaseModel):
    orders: List[Order]
    total_count: int
    page: int = 1
    limit: int = 10


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    database: str
    kafka: str


def lookup_orders_by_email(email: str, limit: int = 10) -> List[dict]:
    """Look up orders by user email, ordered by most recent first"""
    conn = get_database_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT o.id, o.order_number, o.status, o.total_amount, o.currency, o.created_at, o.updated_at, o.shipped_at, o.delivered_at,
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
            GROUP BY o.id, o.order_number, o.status, o.total_amount, o.currency, o.created_at, o.updated_at, o.shipped_at, o.delivered_at,
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


def lookup_order_by_number(order_number: str) -> Optional[dict]:
    """Look up an order by order number"""
    conn = get_database_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT o.id, o.order_number, o.status, o.total_amount, o.currency, o.created_at, o.updated_at, o.shipped_at, o.delivered_at,
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
            GROUP BY o.id, o.order_number, o.status, o.total_amount, o.currency, o.created_at, o.updated_at, o.shipped_at, o.delivered_at,
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


async def get_user_profile(user_email: str) -> UserProfileResponse:
    """Get user profile information including addresses and order stats"""
    try:
        # Get user profile
        user_data = lookup_user_by_email(user_email)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get user addresses
        conn = get_database_connection()
        addresses = []
        if conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT id, address, city, state, postal_code, country
                        FROM user_addresses 
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                    """, (user_data['id'],))
                    address_rows = cursor.fetchall()
                    addresses = [dict(addr) for addr in address_rows]
            except Exception as e:
                print(f"Error fetching addresses: {e}")
            finally:
                conn.close()
        
        # Get order statistics
        orders = lookup_orders_by_email(user_email, limit=1000)  # Get all orders for stats
        total_orders = len(orders)
        total_spent = sum(order['total_amount'] for order in orders)
        
        # Format user profile
        profile = UserProfile(
            id=str(user_data['id']),
            email=user_data['email'],
            first_name=user_data.get('first_name'),
            last_name=user_data.get('last_name'),
            phone=user_data.get('phone'),
            is_admin=user_data.get('is_admin', False),
            created_at=user_data['created_at'].isoformat(),
            updated_at=user_data['updated_at'].isoformat()
        )
        
        # Format addresses
        formatted_addresses = [
            UserAddress(
                id=str(addr['id']),
                address=addr['address'],
                city=addr['city'],
                state=addr.get('state'),
                postal_code=addr['postal_code'],
                country=addr['country']
            ) for addr in addresses
        ]
        
        return UserProfileResponse(
            profile=profile,
            addresses=formatted_addresses,
            total_orders=total_orders,
            total_spent=total_spent
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting user profile: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def get_user_orders(user_email: str, page: int = 1, limit: int = 10) -> OrdersResponse:
    """Get user's orders with pagination"""
    try:
        # Validate pagination parameters
        if page < 1:
            page = 1
        if limit < 1 or limit > 100:
            limit = 10
        
        # Get orders
        orders = lookup_orders_by_email(user_email, limit=1000)  # Get all for pagination
        total_count = len(orders)
        
        # Apply pagination
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_orders = orders[start_idx:end_idx]
        
        # Format orders
        formatted_orders = []
        for order in paginated_orders:
            # Format order items
            order_items = []
            if order.get('items'):
                for item in order['items']:
                    order_items.append(OrderItem(
                        product_name=item['product_name'],
                        quantity=item['quantity'],
                        unit_price=float(item['unit_price']),
                        total_price=float(item['total_price'])
                    ))
            
            formatted_orders.append(Order(
                id=str(order['id']),
                order_number=order['order_number'],
                status=order['status'],
                total_amount=float(order['total_amount']),
                currency=order['currency'],
                created_at=order['created_at'].isoformat(),
                updated_at=order['updated_at'].isoformat(),
                shipped_at=order.get('shipped_at').isoformat() if order.get('shipped_at') else None,
                delivered_at=order.get('delivered_at').isoformat() if order.get('delivered_at') else None,
                items=order_items
            ))
        
        return OrdersResponse(
            orders=formatted_orders,
            total_count=total_count,
            page=page,
            limit=limit
        )
        
    except Exception as e:
        print(f"Error getting user orders: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def get_order_details(order_number: str) -> Order:
    """Get detailed information about a specific order"""
    try:
        order_data = lookup_order_by_number(order_number)
        if not order_data:
            raise HTTPException(status_code=404, detail="Order not found")
        
        # Format order items
        order_items = []
        if order_data.get('items'):
            for item in order_data['items']:
                order_items.append(OrderItem(
                    product_name=item['product_name'],
                    quantity=item['quantity'],
                    unit_price=float(item['unit_price']),
                    total_price=float(item['total_price'])
                ))
        
        return Order(
            id=str(order_data['id']),
            order_number=order_data['order_number'],
            status=order_data['status'],
            total_amount=float(order_data['total_amount']),
            currency=order_data['currency'],
            created_at=order_data['created_at'].isoformat(),
            updated_at=order_data['updated_at'].isoformat(),
            shipped_at=order_data.get('shipped_at').isoformat() if order_data.get('shipped_at') else None,
            delivered_at=order_data.get('delivered_at').isoformat() if order_data.get('delivered_at') else None,
            items=order_items
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting order details: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


class CreateOrderItem(BaseModel):
    product_id: str
    quantity: int
    unit_price: float


class CreateOrderRequest(BaseModel):
    user_email: str
    items: List[CreateOrderItem]
    status: str = "pending"
    currency: str = "USD"


async def create_order(request: CreateOrderRequest) -> Order:
    """Create a new order for a user"""
    conn = get_database_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        # Get user by email
        user_data = lookup_user_by_email(request.user_email)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_id = user_data['id']
        
        # Get user's first address (or create a default one)
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT id FROM user_addresses 
                WHERE user_id = %s 
                LIMIT 1
            """, (user_id,))
            address_row = cursor.fetchone()
            address_id = address_row['id'] if address_row else None
        
        # Generate order number
        import uuid
        order_number = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        
        # Calculate total amount
        total_amount = sum(item.unit_price * item.quantity for item in request.items)
        
        # Create order
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                INSERT INTO orders (order_number, user_id, status, total_amount, currency, address_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, order_number, status, total_amount, currency, created_at, updated_at
            """, (order_number, user_id, request.status, total_amount, request.currency, address_id))
            
            order_row = cursor.fetchone()
            order_id = order_row['id']
            
            # Create order items
            for item in request.items:
                cursor.execute("""
                    INSERT INTO order_items (order_id, product_id, quantity, unit_price)
                    VALUES (%s, %s, %s, %s)
                """, (order_id, item.product_id, item.quantity, item.unit_price))
            
            conn.commit()
        
        # Return the created order
        return await get_order_details(order_number)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating order: {e}")
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        conn.close()
