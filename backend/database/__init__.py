"""Database operations module."""
from .connection import get_database_connection
from .user_operations import lookup_user_by_email, update_user_information
from .order_operations import lookup_order_by_number, lookup_orders_by_email, lookup_orders_by_product_name
from .conversations import (
    create_conversation,
    get_conversation,
    update_conversation_context,
    save_query_to_conversation,
    get_conversation_history
)

"""Export all functions so imports are easily accessed elsewhere"""
__all__ = [
    'get_database_connection',
    'lookup_user_by_email',
    'update_user_information',
    'lookup_order_by_number',
    'lookup_orders_by_email',
    'lookup_orders_by_product_name',
    'create_conversation',
    'get_conversation',
    'update_conversation_context',
    'save_query_to_conversation',
    'get_conversation_history'
]

