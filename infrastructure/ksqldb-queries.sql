-- ksqlDB queries for priority-based agentic AI stack
-- Updated to use single ingress topic and priority-segmented agent topics

-- ============================================================================
-- RAW STREAMS (unchanged from your original)
-- ============================================================================

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

-- ============================================================================
-- REFERENCE TABLES (unchanged)
-- ============================================================================

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

-- ============================================================================
-- NORMALIZED RELATIONAL STREAMS (unchanged)
-- ============================================================================

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

-- ============================================================================
-- NEW: SINGLE INGRESS TOPIC
-- Route all events to orchestrator for classification and priority routing
-- ============================================================================

CREATE STREAM system_ingress WITH (
    KAFKA_TOPIC = 'system.ingress',
    VALUE_FORMAT = 'JSON'
) AS
SELECT 
    id as event_id,
    order_number,
    user_id,
    user_email,
    first_name + ' ' + last_name as customer_name,
    'Check order status for ' + order_number as query_text,
    status,
    total_amount,
    currency,
    correlation_id,
    user_correlation_id,
    validation_status,
    validation_errors,
    'ORDER_STATUS_QUERY' as event_type,
    updated_at as event_timestamp,
    CAST(null AS STRING) as session_id
FROM relational_orders
WHERE status IS NOT NULL
EMIT CHANGES;

-- Route customer queries to ingress
INSERT INTO system_ingress
SELECT 
    id as event_id,
    CAST(null AS STRING) as order_number,
    user_id,
    user_email,
    first_name + ' ' + last_name as customer_name,
    query_text,
    status,
    CAST(null AS DOUBLE) as total_amount,
    CAST(null AS STRING) as currency,
    correlation_id,
    user_correlation_id,
    CAST(null AS STRING) as validation_status,
    CAST(null AS ARRAY<STRING>) as validation_errors,
    'CUSTOMER_QUERY' as event_type,
    created_at as event_timestamp,
    CAST(null AS STRING) as session_id
FROM relational_queries
WHERE status IS NOT NULL
EMIT CHANGES;

-- ============================================================================
-- PRIORITY-SEGMENTED AGENT TOPICS
-- Orchestrator will publish to these after classification
-- ============================================================================

-- Order Agent Priority Topics (P1, P2, P3)
CREATE STREAM tasks_order_p1 (
    session_id STRING,
    conversation_id STRING,
    message_type STRING,
    priority INT,
    query_text STRING,
    order_data STRUCT<
        order_id STRING,
        order_number STRING,
        status STRING,
        total_amount DOUBLE,
        currency STRING,
        validation_status STRING
    >,
    state MAP<STRING, STRING>,
    correlation_id STRING,
    event_timestamp BIGINT
) WITH (
    KAFKA_TOPIC = 'tasks.order.p1',
    VALUE_FORMAT = 'JSON',
    PARTITIONS = 3
);

CREATE STREAM tasks_order_p2 (
    session_id STRING,
    conversation_id STRING,
    message_type STRING,
    priority INT,
    query_text STRING,
    order_data STRUCT<
        order_id STRING,
        order_number STRING,
        status STRING,
        total_amount DOUBLE,
        currency STRING,
        validation_status STRING
    >,
    state MAP<STRING, STRING>,
    correlation_id STRING,
    event_timestamp BIGINT
) WITH (
    KAFKA_TOPIC = 'tasks.order.p2',
    VALUE_FORMAT = 'JSON',
    PARTITIONS = 3
);

CREATE STREAM tasks_order_p3 (
    session_id STRING,
    conversation_id STRING,
    message_type STRING,
    priority INT,
    query_text STRING,
    order_data STRUCT<
        order_id STRING,
        order_number STRING,
        status STRING,
        total_amount DOUBLE,
        currency STRING,
        validation_status STRING
    >,
    state MAP<STRING, STRING>,
    correlation_id STRING,
    event_timestamp BIGINT
) WITH (
    KAFKA_TOPIC = 'tasks.order.p3',
    VALUE_FORMAT = 'JSON',
    PARTITIONS = 3
);

-- Email Agent Priority Topics (P1, P2, P3)
CREATE STREAM tasks_email_p1 (
    session_id STRING,
    conversation_id STRING,
    message_type STRING,
    priority INT,
    query_text STRING,
    email_data STRUCT<
        recipient_email STRING,
        email_type STRING,
        template_id STRING,
        order_number STRING
    >,
    state MAP<STRING, STRING>,
    correlation_id STRING,
    event_timestamp BIGINT
) WITH (
    KAFKA_TOPIC = 'tasks.email.p1',
    VALUE_FORMAT = 'JSON',
    PARTITIONS = 3
);

CREATE STREAM tasks_email_p2 (
    session_id STRING,
    conversation_id STRING,
    message_type STRING,
    priority INT,
    query_text STRING,
    email_data STRUCT<
        recipient_email STRING,
        email_type STRING,
        template_id STRING,
        order_number STRING
    >,
    state MAP<STRING, STRING>,
    correlation_id STRING,
    event_timestamp BIGINT
) WITH (
    KAFKA_TOPIC = 'tasks.email.p2',
    VALUE_FORMAT = 'JSON',
    PARTITIONS = 3
);

CREATE STREAM tasks_email_p3 (
    session_id STRING,
    conversation_id STRING,
    message_type STRING,
    priority INT,
    query_text STRING,
    email_data STRUCT<
        recipient_email STRING,
        email_type STRING,
        template_id STRING,
        order_number STRING
    >,
    state MAP<STRING, STRING>,
    correlation_id STRING,
    event_timestamp BIGINT
) WITH (
    KAFKA_TOPIC = 'tasks.email.p3',
    VALUE_FORMAT = 'JSON',
    PARTITIONS = 3
);

-- Policy Agent Priority Topics (P1, P2, P3)
CREATE STREAM tasks_policy_p1 (
    session_id STRING,
    conversation_id STRING,
    message_type STRING,
    priority INT,
    query_text STRING,
    state MAP<STRING, STRING>,
    correlation_id STRING,
    event_timestamp BIGINT
) WITH (
    KAFKA_TOPIC = 'tasks.policy.p1',
    VALUE_FORMAT = 'JSON',
    PARTITIONS = 3
);

CREATE STREAM tasks_policy_p2 (
    session_id STRING,
    conversation_id STRING,
    message_type STRING,
    priority INT,
    query_text STRING,
    state MAP<STRING, STRING>,
    correlation_id STRING,
    event_timestamp BIGINT
) WITH (
    KAFKA_TOPIC = 'tasks.policy.p2',
    VALUE_FORMAT = 'JSON',
    PARTITIONS = 3
);

CREATE STREAM tasks_policy_p3 (
    session_id STRING,
    conversation_id STRING,
    message_type STRING,
    priority INT,
    query_text STRING,
    state MAP<STRING, STRING>,
    correlation_id STRING,
    event_timestamp BIGINT
) WITH (
    KAFKA_TOPIC = 'tasks.policy.p3',
    VALUE_FORMAT = 'JSON',
    PARTITIONS = 3
);

-- Message Agent Priority Topics (P1, P2, P3)
CREATE STREAM tasks_message_p1 (
    session_id STRING,
    conversation_id STRING,
    message_type STRING,
    priority INT,
    query_text STRING,
    state MAP<STRING, STRING>,
    correlation_id STRING,
    event_timestamp BIGINT
) WITH (
    KAFKA_TOPIC = 'tasks.message.p1',
    VALUE_FORMAT = 'JSON',
    PARTITIONS = 3
);

CREATE STREAM tasks_message_p2 (
    session_id STRING,
    conversation_id STRING,
    message_type STRING,
    priority INT,
    query_text STRING,
    state MAP<STRING, STRING>,
    correlation_id STRING,
    event_timestamp BIGINT
) WITH (
    KAFKA_TOPIC = 'tasks.message.p2',
    VALUE_FORMAT = 'JSON',
    PARTITIONS = 3
);

CREATE STREAM tasks_message_p3 (
    session_id STRING,
    conversation_id STRING,
    message_type STRING,
    priority INT,
    query_text STRING,
    state MAP<STRING, STRING>,
    correlation_id STRING,
    event_timestamp BIGINT
) WITH (
    KAFKA_TOPIC = 'tasks.message.p3',
    VALUE_FORMAT = 'JSON',
    PARTITIONS = 3
);

-- ============================================================================
-- AGENT RESPONSE TOPIC (unchanged)
-- ============================================================================

CREATE STREAM agent_responses (
    session_id STRING,
    conversation_id STRING,
    agent_type STRING,
    message STRING,
    status STRING,
    priority INT,
    correlation_id STRING,
    timestamp BIGINT
) WITH (
    KAFKA_TOPIC = 'agent.responses',
    VALUE_FORMAT = 'JSON',
    PARTITIONS = 3
);

-- ============================================================================
-- SENDGRID EMAIL REQUESTS (unchanged from your original)
-- ============================================================================

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

-- ============================================================================
-- ANALYTICS VIEWS (unchanged)
-- ============================================================================

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
    COLLECT_LIST(order_number) as order_numbers,
    COLLECT_LIST(status) as order_statuses
FROM relational_orders
WHERE status IN ('pending', 'confirmed', 'processing', 'shipped')
GROUP BY user_id, user_email, first_name, last_name, user_correlation_id
EMIT CHANGES;

-- Priority distribution monitoring
CREATE TABLE views_priority_distribution WITH (
    KAFKA_TOPIC = 'views.priority_distribution',
    VALUE_FORMAT = 'JSON'
) AS
SELECT 
    agent_type,
    priority,
    COUNT(*) as task_count,
    COLLECT_LIST(session_id) as session_ids
FROM agent_responses
WINDOW TUMBLING (SIZE 1 MINUTE)
GROUP BY agent_type, priority
EMIT CHANGES;

-- Real-time agent performance metrics
CREATE TABLE views_agent_performance WITH (
    KAFKA_TOPIC = 'views.agent_performance',
    VALUE_FORMAT = 'JSON'
) AS
SELECT 
    agent_type,
    COUNT(*) as total_responses,
    COUNT_DISTINCT(session_id) as unique_sessions,
    COLLECT_LIST(STRUCT(priority := priority, timestamp := timestamp)) as response_timeline
FROM agent_responses
WINDOW TUMBLING (SIZE 5 MINUTES)
GROUP BY agent_type
EMIT CHANGES;