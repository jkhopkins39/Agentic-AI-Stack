#!/usr/bin/env python3
"""
Complete Agent Configuration and Database Management
Configuration for integrating your existing LangGraph agents with Kafka
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for each agent"""
    name: str
    topics: List[str]
    priority_levels: Dict[str, int]
    max_concurrent_messages: int = 5
    retry_attempts: int = 3
    timeout_seconds: int = 30

@dataclass
class KafkaConfig:
    """Kafka connection and topic configuration"""
    bootstrap_servers: str = 'localhost:9092'
    consumer_group_prefix: str = 'agentic-ai'
    auto_offset_reset: str = 'latest'
    enable_auto_commit: bool = True
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 10000

class AgentIntegrationConfig:
    """Main configuration class for agent-Kafka integration"""
    
    def __init__(self):
        self.kafka_config = KafkaConfig()
        self.agents = self._setup_agent_configs()
        self.database_config = self._setup_database_config()
        
    def _setup_agent_configs(self) -> Dict[str, AgentConfig]:
        """Configure each agent with its Kafka topics and settings"""
        return {
            'message_agent': AgentConfig(
                name='message_agent',
                topics=['customer.query_events'],
                priority_levels={
                    'RESPONSE_REQUIRED': 1,
                    'QUERY_RESOLVED': 2,
                    'STATUS_UPDATE': 3
                },
                max_concurrent_messages=10,
                retry_attempts=3,
                timeout_seconds=30
            ),
            
            'order_agent': AgentConfig(
                name='order_agent', 
                topics=['commerce.order_events'],
                priority_levels={
                    'VALIDATION_REQUIRED': 1,
                    'ORDER_CANCELLED': 2,
                    'ORDER_CONFIRMED': 3,
                    'STATUS_UPDATE': 4
                },
                max_concurrent_messages=15,
                retry_attempts=5,
                timeout_seconds=45
            ),
            
            'email_agent': AgentConfig(
                name='email_agent',
                topics=['commerce.order_validation'],
                priority_levels={
                    'SEND_ORDER_FAILURE': 1,
                    'SEND_ORDER_CONFIRMATION': 2,
                    'SEND_CANCELLATION_NOTICE': 2,
                    'SEND_SHIPMENT_NOTIFICATION': 3,
                    'SEND_DELIVERY_CONFIRMATION': 3
                },
                max_concurrent_messages=20,
                retry_attempts=3,
                timeout_seconds=60
            ),
            
            'policy_agent': AgentConfig(
                name='policy_agent',
                topics=['customer.policy_queries', 'customer.query_events', 'policy.order_events'],
                priority_levels={
                    'POLICY_RESPONSE_REQUIRED': 2,
                    'REFUND_POLICY_CHECK': 1,
                    'LONG_DELIVERY_POLICY': 2,
                    'STANDARD_POLICY_CHECK': 3
                },
                max_concurrent_messages=8,
                retry_attempts=3,
                timeout_seconds=30
            )
        }
    
    def _setup_database_config(self) -> Dict[str, str]:
        """Database connection configuration for agent data retrieval"""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'AgenticAIStackDB'),
            'user': os.getenv('DB_USER', 'AgenticAIStackDB'),
            'password': os.getenv('DB_PASSWORD', '8%w=r?D52Eo2EwcVW:'),
        }

class DatabaseManager:
    """Manage database connections for agents"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.connection = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                cursor_factory=RealDictCursor
            )
            self.connection.autocommit = True
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def find_order_by_correlation_id(self, correlation_id: str) -> Dict[str, Any]:
        """Find order using correlation_id - enhanced version of your find_order function"""
        if not self.connection:
            self.connect()
            
        try:
            with self.connection.cursor() as cursor:
                # Enhanced query with correlation_id and enriched data
                query = """
                SELECT 
                    o.id,
                    o.order_number,
                    o.user_id,
                    o.status,
                    o.total_amount,
                    o.currency,
                    o.correlation_id,
                    o.validation_status,
                    o.validation_errors,
                    o.created_at,
                    o.updated_at,
                    o.shipped_at,
                    o.delivered_at,
                    u.email as user_email,
                    u.first_name,
                    u.last_name,
                    u.phone,
                    ua.address,
                    ua.city,
                    ua.state,
                    ua.postal_code,
                    ua.country
                FROM orders o
                LEFT JOIN users u ON o.user_id = u.id
                LEFT JOIN user_addresses ua ON o.address_id = ua.id
                WHERE o.correlation_id = %s
                """
                
                cursor.execute(query, (correlation_id,))
                order = cursor.fetchone()
                
                if order:
                    # Get order items
                    items_query = """
                    SELECT 
                        oi.id,
                        oi.quantity,
                        oi.unit_price,
                        p.name,
                        p.description,
                        p.price as current_price
                    FROM order_items oi
                    LEFT JOIN products p ON oi.product_id = p.id
                    WHERE oi.order_id = %s
                    """
                    
                    cursor.execute(items_query, (order['id'],))
                    items = cursor.fetchall()
                    
                    # Convert to your existing format
                    result = {
                        'id': str(order['id']),
                        'order_number': order['order_number'],
                        'user_id': str(order['user_id']),
                        'user_email': order['user_email'],
                        'status': order['status'],
                        'total_amount': float(order['total_amount']),
                        'currency': order['currency'],
                        'correlation_id': order['correlation_id'],
                        'validation_status': order['validation_status'],
                        'validation_errors': order['validation_errors'] or [],
                        'created_at': order['created_at'].isoformat() if order['created_at'] else None,
                        'customer_name': f"{order['first_name']} {order['last_name']}" if order['first_name'] else 'Unknown',
                        'shipping_address': f"{order['address']}, {order['city']}, {order['state']} {order['postal_code']}" if order['address'] else 'No address',
                        'items': [
                            {
                                'id': str(item['id']),
                                'name': item['name'],
                                'description': item['description'],
                                'quantity': item['quantity'],
                                'price': float(item['unit_price']),
                                'current_price': float(item['current_price'])
                            }
                            for item in items
                        ]
                    }
                    
                    # Add tracking number for shipped orders
                    if order['status'] == 'shipped':
                        result['tracking_number'] = f"TRACK-{order['order_number']}"
                        result['estimated_delivery'] = 'N/A'  # Could calculate from shipped_at
                    elif order['status'] == 'delivered':
                        result['delivered_at'] = order['delivered_at'].isoformat() if order['delivered_at'] else None
                    elif order['status'] == 'processing':
                        result['estimated_ship_date'] = 'N/A'  # Could calculate business logic
                    
                    return result
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"Error finding order by correlation_id {correlation_id}: {e}")
            return {}
    
    def find_order_by_number(self, order_number: str) -> Dict[str, Any]:
        """Find order by order number"""
        if not self.connection:
            self.connect()
            
        try:
            with self.connection.cursor() as cursor:
                query = """
                SELECT 
                    o.id,
                    o.order_number,
                    o.user_id,
                    o.status,
                    o.total_amount,
                    o.currency,
                    o.correlation_id,
                    o.validation_status,
                    o.validation_errors,
                    o.created_at,
                    o.updated_at,
                    o.shipped_at,
                    o.delivered_at,
                    u.email as user_email,
                    u.first_name,
                    u.last_name,
                    u.phone
                FROM orders o
                LEFT JOIN users u ON o.user_id = u.id
                WHERE UPPER(o.order_number) = UPPER(%s)
                """
                
                cursor.execute(query, (order_number,))
                order = cursor.fetchone()
                
                if order:
                    return {
                        'id': str(order['id']),
                        'order_number': order['order_number'],
                        'user_id': str(order['user_id']),
                        'user_email': order['user_email'],
                        'status': order['status'],
                        'total_amount': float(order['total_amount']),
                        'currency': order['currency'],
                        'correlation_id': order['correlation_id'],
                        'validation_status': order['validation_status'],
                        'validation_errors': order['validation_errors'] or [],
                        'created_at': order['created_at'].isoformat() if order['created_at'] else None,
                        'customer_name': f"{order['first_name']} {order['last_name']}" if order['first_name'] else 'Unknown'
                    }
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"Error finding order by number {order_number}: {e}")
            return {}
    
    def find_queries_by_correlation_id(self, correlation_id: str) -> List[Dict[str, Any]]:
        """Find related queries using correlation_id"""
        if not self.connection:
            self.connect()
            
        try:
            with self.connection.cursor() as cursor:
                query = """
                SELECT 
                    q.id,
                    q.user_id,
                    q.query_text,
                    q.status,
                    q.agent_response,
                    q.correlation_id,
                    q.related_order_id,
                    q.created_at,
                    q.updated_at,
                    u.email as user_email,
                    u.first_name,
                    u.last_name
                FROM queries q
                LEFT JOIN users u ON q.user_id = u.id
                WHERE q.correlation_id = %s OR q.related_order_id IN (
                    SELECT id FROM orders WHERE correlation_id = %s
                )
                ORDER BY q.created_at DESC
                """
                
                cursor.execute(query, (correlation_id, correlation_id))
                queries = cursor.fetchall()
                
                return [
                    {
                        'id': str(query['id']),
                        'query_text': query['query_text'],
                        'status': query['status'],
                        'agent_response': query['agent_response'],
                        'correlation_id': query['correlation_id'],
                        'customer_name': f"{query['first_name']} {query['last_name']}" if query['first_name'] else 'Unknown',
                        'user_email': query['user_email'],
                        'created_at': query['created_at'].isoformat() if query['created_at'] else None,
                        'updated_at': query['updated_at'].isoformat() if query['updated_at'] else None
                    }
                    for query in queries
                ]
                
        except Exception as e:
            logger.error(f"Error finding queries by correlation_id {correlation_id}: {e}")
            return []
    
    def find_orders_by_email(self, email: str) -> List[Dict[str, Any]]:
        """Find orders by user email"""
        if not self.connection:
            self.connect()
            
        try:
            with self.connection.cursor() as cursor:
                query = """
                SELECT 
                    o.id,
                    o.order_number,
                    o.status,
                    o.total_amount,
                    o.currency,
                    o.correlation_id,
                    o.created_at,
                    u.email as user_email,
                    u.first_name,
                    u.last_name
                FROM orders o
                LEFT JOIN users u ON o.user_id = u.id
                WHERE LOWER(u.email) = LOWER(%s)
                ORDER BY o.created_at DESC
                LIMIT 10
                """
                
                cursor.execute(query, (email,))
                orders = cursor.fetchall()
                
                return [
                    {
                        'id': str(order['id']),
                        'order_number': order['order_number'],
                        'status': order['status'],
                        'total_amount': float(order['total_amount']),
                        'currency': order['currency'],
                        'correlation_id': order['correlation_id'],
                        'created_at': order['created_at'].isoformat() if order['created_at'] else None,
                        'user_email': order['user_email'],
                        'customer_name': f"{order['first_name']} {order['last_name']}" if order['first_name'] else 'Unknown'
                    }
                    for order in orders
                ]
                
        except Exception as e:
            logger.error(f"Error finding orders by email {email}: {e}")
            return []
    
    def create_query(self, user_id: str, query_text: str, correlation_id: Optional[str] = None, 
                    related_order_id: Optional[str] = None) -> str:
        """Create a new query record"""
        if not self.connection:
            self.connect()
            
        try:
            with self.connection.cursor() as cursor:
                query_id = str(uuid.uuid4())
                if not correlation_id:
                    correlation_id = str(uuid.uuid4())
                
                insert_query = """
                INSERT INTO queries (id, user_id, query_text, status, correlation_id, related_order_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                now = datetime.now()
                cursor.execute(insert_query, (
                    query_id, user_id, query_text, 'pending', correlation_id, related_order_id, now, now
                ))
                
                logger.info(f"Created query {query_id} with correlation_id {correlation_id}")
                return query_id
                
        except Exception as e:
            logger.error(f"Error creating query: {e}")
            raise
    
    def update_query_response(self, query_id: str, agent_response: str, status: str = 'resolved'):
        """Update query with agent response"""
        if not self.connection:
            self.connect()
            
        try:
            with self.connection.cursor() as cursor:
                update_query = """
                UPDATE queries 
                SET agent_response = %s, status = %s, updated_at = %s
                WHERE id = %s
                """
                
                cursor.execute(update_query, (agent_response, status, datetime.now(), query_id))
                logger.info(f"Updated query {query_id} with response")
                
        except Exception as e:
            logger.error(f"Error updating query response: {e}")
    
    def create_email_notification(self, correlation_id: str, recipient_email: str, 
                                email_type: str, template_data: Dict[str, Any],
                                status: str = 'pending') -> str:
        """Create email notification record"""
        if not self.connection:
            self.connect()
            
        try:
            with self.connection.cursor() as cursor:
                notification_id = str(uuid.uuid4())
                
                insert_query = """
                INSERT INTO email_notifications 
                (id, correlation_id, recipient_email, email_type, template_data, status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                now = datetime.now()
                cursor.execute(insert_query, (
                    notification_id, correlation_id, recipient_email, email_type, 
                    json.dumps(template_data), status, now, now
                ))
                
                logger.info(f"Created email notification {notification_id}")
                return notification_id
                
        except Exception as e:
            logger.error(f"Error creating email notification: {e}")
            raise
    
    def update_email_status(self, notification_id: str, status: str, 
                           sendgrid_message_id: Optional[str] = None,
                           error_message: Optional[str] = None):
        """Update email notification status"""
        if not self.connection:
            self.connect()
            
        try:
            with self.connection.cursor() as cursor:
                update_query = """
                UPDATE email_notifications 
                SET status = %s, sendgrid_message_id = %s, error_message = %s, updated_at = %s
                WHERE id = %s
                """
                
                cursor.execute(update_query, (
                    status, sendgrid_message_id, error_message, datetime.now(), notification_id
                ))
                
                logger.info(f"Updated email notification {notification_id} status to {status}")
                
        except Exception as e:
            logger.error(f"Error updating email status: {e}")
    
    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics from database"""
        if not self.connection:
            self.connect()
            
        try:
            with self.connection.cursor() as cursor:
                # Query metrics
                cursor.execute("""
                SELECT 
                    status,
                    COUNT(*) as query_count,
                    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_response_time_seconds
                FROM queries 
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                GROUP BY status
                """)
                
                query_metrics = cursor.fetchall()
                
                # Email metrics
                cursor.execute("""
                SELECT 
                    email_type,
                    status,
                    COUNT(*) as email_count
                FROM email_notifications 
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                GROUP BY email_type, status
                """)
                
                email_metrics = cursor.fetchall()
                
                # Order metrics  
                cursor.execute("""
                SELECT 
                    status,
                    validation_status,
                    COUNT(*) as order_count,
                    AVG(total_amount) as avg_order_value
                FROM orders 
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                GROUP BY status, validation_status
                """)
                
                order_metrics = cursor.fetchall()
                
                return {
                    'query_metrics': [dict(row) for row in query_metrics],
                    'email_metrics': [dict(row) for row in email_metrics],
                    'order_metrics': [dict(row) for row in order_metrics],
                    'generated_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting agent metrics: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check if database connection is healthy"""
        try:
            if not self.connection:
                self.connect()
                
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result is not None
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

# Utility functions for integration with your existing code
def enhance_existing_functions_with_db(db_manager: DatabaseManager):
    """Enhance your existing find_order and search functions with database integration"""
    
    def enhanced_find_order(order_number: str) -> Dict[str, Any]:
        """Enhanced version of your find_order function"""
        # Try database first
        db_result = db_manager.find_order_by_number(order_number)
        if db_result:
            return db_result
        
        # Fall back to your existing SAMPLE_ORDERS
        from aligned_enhanced_agents import SAMPLE_ORDERS
        return SAMPLE_ORDERS.get(order_number.upper(), {})
    
    def enhanced_search_orders_by_email(email: str) -> List[Dict[str, Any]]:
        """Enhanced version of your search_orders_by_email function"""
        # Try database first
        db_results = db_manager.find_orders_by_email(email)
        if db_results:
            return db_results
        
        # Fall back to your existing SAMPLE_ORDERS
        from aligned_enhanced_agents import SAMPLE_ORDERS
        return [order for order in SAMPLE_ORDERS.values() if order["user_email"].lower() == email.lower()]
    
    def enhanced_find_order_by_correlation_id(correlation_id: str) -> Dict[str, Any]:
        """New function to find orders by correlation_id"""
        return db_manager.find_order_by_correlation_id(correlation_id)
    
    return enhanced_find_order, enhanced_search_orders_by_email, enhanced_find_order_by_correlation_id

# Configuration validation
def validate_configuration() -> Tuple[bool, List[str]]:
    """Validate that all required configuration is present"""
    errors = []
    
    # Check environment variables
    required_env_vars = [
        'DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD',
        'KAFKA_BOOTSTRAP_SERVERS'
    ]
    
    for var in required_env_vars:
        if not os.getenv(var):
            errors.append(f"Missing environment variable: {var}")
    
    # Test database connection
    try:
        config = AgentIntegrationConfig()
        db_manager = DatabaseManager(config.database_config)
        db_manager.connect()
        if not db_manager.health_check():
            errors.append("Database health check failed")
        db_manager.close()
    except Exception as e:
        errors.append(f"Database connection test failed: {e}")
    
    # Test Kafka connection
    try:
        from kafka import KafkaConsumer
        bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers)
        consumer.close()
    except Exception as e:
        errors.append(f"Kafka connection test failed: {e}")
    
    return len(errors) == 0, errors

if __name__ == "__main__":
    # Test the configuration
    print("🔧 Testing Agent Configuration...")
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    # Validate configuration
    is_valid, errors = validate_configuration()
    
    if is_valid:
        print("✅ Configuration is valid!")
        
        # Test database operations
        config = AgentIntegrationConfig()
        db_manager = DatabaseManager(config.database_config)
        
        try:
            db_manager.connect()
            
            # Test basic operations
            print("📊 Testing database operations...")
            
            # Test correlation lookup
            test_correlation = "550e8400-e29b-41d4-a716-446655440001" 
            order = db_manager.find_order_by_correlation_id(test_correlation)
            print(f"✅ Correlation lookup: {len(order)} fields returned")
            
            # Test metrics
            metrics = db_manager.get_agent_metrics()
            print(f"✅ Metrics query: {len(metrics)} metric categories")
            
            print("✅ All database tests passed!")
            
        except Exception as e:
            print(f"❌ Database test failed: {e}")
        finally:
            db_manager.close()
    else:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix these issues before running the agents.")