-- ksqlDB queries for processing data streams for your agentic AI stack
-- Updated with correlation_id support and email agent workflow

-- Create streams from raw Kafka topics (created by Debezium)
CREATE STREAM raw_orders_stream (
    id STRING,
    order_number STRING,
    user_id STRING,
    status STRING,
    total_amount DOUBLE,
    currency STRING,
    address_id STRING,
    correlation_id STRING,
    validation_status STRING,
    validation_errors ARRAY<STRING>,
    created_at BIGINT,
    updated_at BIGINT,
    shipped_at BIGINT,
    delivered_at BIGINT
) WITH (
    KAFKA_TOPIC = 'raw.orders',
    VALUE_FORMAT = 'JSON'
);

CREATE STREAM raw_order_items_stream (
    id STRING,
    order_id STRING,
    product_id STRING,
    quantity INT,
    unit_price DOUBLE,
    correlation_id STRING,
    created_at BIGINT
) WITH (
    KAFKA_TOPIC = 'raw.order_items',
    VALUE_FORMAT = 'JSON'
);

CREATE STREAM raw_queries_stream (
    id STRING,
    user_id STRING,
    query_text STRING,
    status STRING,
    agent_response STRING,
    correlation_id STRING,
    related_order_id STRING,
    created_at BIGINT,
    updated_at BIGINT
) WITH (
    KAFKA_TOPIC = 'raw.queries',
    VALUE_FORMAT = 'JSON'
);

CREATE STREAM raw_users_stream (
    id STRING,
    email STRING,
    first_name STRING,
    last_name STRING,
    phone STRING,
    correlation_id STRING,
    created_at BIGINT,
    updated_at BIGINT
) WITH (
    KAFKA_TOPIC = 'raw.users',
    VALUE_FORMAT = 'JSON'
);

CREATE STREAM raw_products_stream (
    id STRING,
    name STRING,
    description STRING,
    price DOUBLE,
    stock_quantity INT,
    created_at BIGINT,
    updated_at BIGINT
) WITH (
    KAFKA_TOPIC = 'raw.products',
    VALUE_FORMAT = 'JSON'
);

CREATE STREAM raw_email_notifications_stream (
    id STRING,
    correlation_id STRING,
    recipient_email STRING,
    email_type STRING,
    subject STRING,
    template_id STRING,
    template_data STRING,
    sendgrid_message_id STRING,
    status STRING,
    sent_at BIGINT,
    delivered_at BIGINT,
    error_message STRING,
    created_at BIGINT,
    updated_at BIGINT
) WITH (
    KAFKA_TOPIC = 'raw.email_notifications',
    VALUE_FORMAT = 'JSON'
);

-- Create master data reference streams/tables
CREATE TABLE ref_users WITH (
    KAFKA_TOPIC = 'ref.users',
    VALUE_FORMAT = 'JSON'
) AS
SELECT 
    id,
    email,
    first_name,
    last_name,
    phone,
    correlation_id,
    created_at,
    updated_at
FROM raw_users_stream
EMIT CHANGES;

CREATE TABLE ref_products WITH (
    KAFKA_TOPIC = 'ref.products',
    VALUE_FORMAT = 'JSON'
) AS
SELECT 
    id,
    name,
    description,
    price,
    stock_quantity,
    created_at,
    updated_at
FROM raw_products_stream
EMIT CHANGES;

-- Create normalized/enriched relational streams with correlation_id tracking
CREATE STREAM relational_orders WITH (
    KAFKA_TOPIC = 'relational.orders',
    VALUE_FORMAT = 'JSON'
) AS
SELECT 
    o.id,
    o.order_number,
    o.user_id,
    u.email as user_email,
    u.first_name,
    u.last_name,
    o.status,
    o.total_amount,
    o.currency,
    o.address_id,
    o.correlation_id,
    u.correlation_id as user_correlation_id,
    o.validation_status,
    o.validation_errors,
    o.created_at,
    o.updated_at,
    o.shipped_at,
    o.delivered_at
FROM raw_orders_stream o
LEFT JOIN ref_users u ON o.user_id = u.id
EMIT CHANGES;

CREATE STREAM relational_order_items WITH (
    KAFKA_TOPIC = 'relational.order_items',
    VALUE_FORMAT = 'JSON'
) AS
SELECT 
    oi.id,
    oi.order_id,
    oi.product_id,
    p.name as product_name,
    p.description as product_description,
    oi.quantity,
    oi.unit_price,
    p.price as current_price,
    oi.correlation_id,
    oi.created_at
FROM raw_order_items_stream oi
LEFT JOIN ref_products p ON oi.product_id = p.id
EMIT CHANGES;

CREATE STREAM relational_queries WITH (
    KAFKA_TOPIC = 'relational.queries',
    VALUE_FORMAT = 'JSON'
) AS
SELECT 
    q.id,
    q.user_id,
    u.email as user_email,
    u.first_name,
    u.last_name,
    q.query_text,
    q.status,
    q.agent_response,
    q.correlation_id,
    u.correlation_id as user_correlation_id,
    q.related_order_id,
    q.created_at,
    q.updated_at
FROM raw_queries_stream q
LEFT JOIN ref_users u ON q.user_id = u.id
EMIT CHANGES;

-- Create business event streams for agents with correlation_id

-- 1. Commerce order events for Order Agent
CREATE STREAM commerce_order_events WITH (
    KAFKA_TOPIC = 'commerce.order_events',
    VALUE_FORMAT = 'JSON'
) AS
SELECT 
    id as order_id,
    order_number,
    user_id,
    user_email,
    first_name + ' ' + last_name as customer_name,
    status,
    total_amount,
    currency,
    correlation_id,
    user_correlation_id,
    validation_status,
    validation_errors,
    'ORDER_STATUS_CHANGE' as event_type,
    CASE 
        WHEN status = 'pending' AND validation_status = 'pending' THEN 'VALIDATION_REQUIRED'
        WHEN status = 'confirmed' THEN 'ORDER_CONFIRMED'
        WHEN status = 'cancelled' THEN 'ORDER_CANCELLED'
        ELSE 'STATUS_UPDATE'
    END as action_required,
    updated_at as event_timestamp,
    'ORDER_AGENT' as target_agent,
    CASE 
        WHEN status = 'pending' THEN 1
        WHEN status = 'cancelled' THEN 2
        ELSE 3
    END as priority
FROM relational_orders
WHERE status IS NOT NULL
EMIT CHANGES;

-- 2. Order validation events specifically for Email Agent
CREATE STREAM commerce_order_validation WITH (
    KAFKA_TOPIC = 'commerce.order_validation',
    VALUE_FORMAT = 'JSON'
) AS
SELECT 
    id as order_id,
    order_number,
    user_id,
    user_email,
    first_name + ' ' + last_name as customer_name,
    status,
    validation_status,
    validation_errors,
    total_amount,
    currency,
    correlation_id,
    'ORDER_VALIDATION_EVENT' as event_type,
    CASE 
        WHEN validation_status = 'validated' AND status = 'confirmed' THEN 'SEND_ORDER_CONFIRMATION'
        WHEN validation_status = 'failed' THEN 'SEND_ORDER_FAILURE'
        WHEN status = 'shipped' THEN 'SEND_SHIPMENT_NOTIFICATION'
        WHEN status = 'delivered' THEN 'SEND_DELIVERY_CONFIRMATION'
        WHEN status = 'cancelled' THEN 'SEND_CANCELLATION_NOTICE'
        ELSE 'NO_EMAIL_REQUIRED'
    END as email_action,
    updated_at as event_timestamp,
    'EMAIL_AGENT' as target_agent,
    CASE 
        WHEN validation_status = 'failed' THEN 1
        WHEN status IN ('confirmed', 'cancelled') THEN 2
        ELSE 3
    END as priority
FROM relational_orders
WHERE (validation_status IN ('validated', 'failed') OR status IN ('shipped', 'delivered', 'cancelled'))
AND status IS NOT NULL
EMIT CHANGES;

-- 3. Commerce order item events 
CREATE STREAM commerce_order_item_events WITH (
    KAFKA_TOPIC = 'commerce.order_item_events',
    VALUE_FORMAT = 'JSON'
) AS
SELECT 
    id as order_item_id,
    order_id,
    product_id,
    product_name,
    quantity,
    unit_price,
    current_price,
    correlation_id,
    'ORDER_ITEM_CREATED' as event_type,
    created_at as event_timestamp
FROM relational_order_items
EMIT CHANGES;

-- 4. Customer query events for Message Agent
CREATE STREAM customer_query_events WITH (
    KAFKA_TOPIC = 'customer.query_events',
    VALUE_FORMAT = 'JSON'
) AS
SELECT 
    id as query_id,
    user_id,
    user_email,
    first_name + ' ' + last_name as customer_name,
    query_text,
    status,
    agent_response,
    correlation_id,
    user_correlation_id,
    related_order_id,
    'CUSTOMER_QUERY' as event_type,
    CASE 
        WHEN status = 'pending' THEN 'RESPONSE_REQUIRED'
        WHEN status = 'resolved' THEN 'QUERY_RESOLVED'
        ELSE 'STATUS_UPDATE'
    END as action_required,
    created_at as event_timestamp,
    'MESSAGE_AGENT' as target_agent,
    CASE 
        WHEN status = 'pending' THEN 1
        ELSE 2
    END as priority
FROM relational_queries
WHERE status IS NOT NULL
EMIT CHANGES;

-- 5. Customer login events (placeholder - implement when you add login tracking)
CREATE STREAM customer_login_events (
    user_id STRING,
    user_email STRING,
    user_correlation_id STRING,
    login_timestamp BIGINT,
    ip_address STRING,
    user_agent STRING,
    login_method STRING,
    success BOOLEAN,
    failure_reason STRING,
    session_id STRING,
    correlation_id STRING,
    event_type STRING
) WITH (
    KAFKA_TOPIC = 'customer.login_events',
    VALUE_FORMAT = 'JSON'
);

-- 6. SendGrid email output topic for Email Agent
CREATE STREAM sendgrid_email_requests (
    correlation_id STRING,
    recipient_email STRING,
    recipient_name STRING,
    email_type STRING,
    template_id STRING,
    template_data STRUCT<
        customer_name STRING,
        order_number STRING,
        order_total STRING,
        order_items ARRAY<STRUCT<name STRING, quantity INT, price STRING>>,
        tracking_number STRING,
        delivery_address STRING,
        failure_reason STRING
    >,
    priority INT,
    event_timestamp BIGINT,
    source_agent STRING
) WITH (
    KAFKA_TOPIC = 'sendgrid.email_requests',
    VALUE_FORMAT = 'JSON'
);

-- Transform order validation events into SendGrid email requests
CREATE STREAM sendgrid_order_emails WITH (
    KAFKA_TOPIC = 'sendgrid.email_requests',
    VALUE_FORMAT = 'JSON'
) AS
SELECT 
    correlation_id,
    user_email as recipient_email,
    customer_name as recipient_name,
    CASE 
        WHEN email_action = 'SEND_ORDER_CONFIRMATION' THEN 'ORDER_CONFIRMATION'
        WHEN email_action = 'SEND_ORDER_FAILURE' THEN 'ORDER_FAILURE'
        WHEN email_action = 'SEND_SHIPMENT_NOTIFICATION' THEN 'ORDER_SHIPPED'
        WHEN email_action = 'SEND_DELIVERY_CONFIRMATION' THEN 'ORDER_DELIVERED'
        WHEN email_action = 'SEND_CANCELLATION_NOTICE' THEN 'ORDER_CANCELLED'
        ELSE 'UNKNOWN'
    END as email_type,
    CASE 
        WHEN email_action = 'SEND_ORDER_CONFIRMATION' THEN 'order-confirmation-template'
        WHEN email_action = 'SEND_ORDER_FAILURE' THEN 'order-failure-template'
        WHEN email_action = 'SEND_SHIPMENT_NOTIFICATION' THEN 'order-shipped-template'
        WHEN email_action = 'SEND_DELIVERY_CONFIRMATION' THEN 'order-delivered-template'
        WHEN email_action = 'SEND_CANCELLATION_NOTICE' THEN 'order-cancelled-template'
        ELSE 'default-template'
    END as template_id,
    STRUCT(
        customer_name := customer_name,
        order_number := order_number,
        order_total := CAST(total_amount AS STRING) + ' ' + currency,
        order_items := ARRAY[STRUCT(name := 'Placeholder Item', quantity := 1, price := '0.00')], -- Will be enriched by Email Agent
        tracking_number := CASE WHEN status = 'shipped' THEN 'TRACK-' + order_number ELSE '' END,
        delivery_address := '',
        failure_reason := CASE WHEN validation_status = 'failed' THEN ARRAY_JOIN(validation_errors, '; ') ELSE '' END
    ) as template_data,
    priority,
    event_timestamp,
    'EMAIL_AGENT' as source_agent
FROM commerce_order_validation
WHERE email_action != 'NO_EMAIL_REQUIRED'
EMIT CHANGES;

-- Create materialized views for real-time analytics with correlation_id tracking

-- Active orders per user view
CREATE TABLE views_active_orders_per_user WITH (
    KAFKA_TOPIC = 'views.active_orders_per_user',
    VALUE_FORMAT = 'JSON'
) AS
SELECT 
    user_id,
    user_email,
    first_name + ' ' + last_name as customer_name,
    user_correlation_id,
    COUNT(*) as active_order_count,
    SUM(total_amount) as total_active_value,