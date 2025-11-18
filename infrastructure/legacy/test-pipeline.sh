#!/bin/bash

# Test script to validate the PostgreSQL -> Kafka -> ksqlDB pipeline

echo "Testing PostgreSQL to Kafka pipeline for Agentic AI..."

# Function to insert test data and monitor topics
test_orders_flow() {
    echo "=== Testing Orders Flow ==="
    
    # Insert a test order
    docker exec AgenticAIStackDB psql -U AgenticAIStackDB -d AgenticAIStackDB -c "
    INSERT INTO orders (order_number, user_id, status, total_amount, currency, address_id) 
    VALUES ('TEST-001', (SELECT id FROM users LIMIT 1), 'pending', 1500.00, 'USD', NULL);
    "
    
    echo "Inserted test order. Checking commerce.order_events topic..."
    sleep 5
    
    # Check if order appears in commerce.order_events topic
    timeout 10s docker exec kafka kafka-console-consumer \
        --topic commerce.order_events \
        --bootstrap-server localhost:9092 \
        --from-beginning \
        --max-messages 1
}

test_queries_flow() {
    echo "=== Testing Queries Flow ==="
    
    # Insert a test query
    docker exec AgenticAIStackDB psql -U AgenticAIStackDB -d AgenticAIStackDB -c "
    INSERT INTO queries (user_id, query_text, status) 
    VALUES ((SELECT id FROM users LIMIT 1), 'I need help with my order', 'pending');
    "
    
    echo "Inserted test query. Monitoring customer.query_events topic..."
    sleep 5
    
    # Monitor customer.query_events topic
    timeout 10s docker exec kafka kafka-console-consumer \
        --topic customer.query_events \
        --bootstrap-server localhost:9092 \
        --from-beginning \
        --max-messages 1
}

test_order_status_update() {
    echo "=== Testing Order Status Updates & Email Validation ==="
    
    # Update order status and validation to trigger email notifications
    docker exec AgenticAIStackDB psql -U AgenticAIStackDB -d AgenticAIStackDB -c "
    UPDATE orders 
    SET status = 'confirmed', 
        validation_status = 'validated',
        updated_at = CURRENT_TIMESTAMP 
    WHERE order_number = 'TEST-001';
    "
    
    echo "Updated order status. Checking commerce.order_validation and sendgrid.email_requests topics..."
    sleep 5
    
    # Check commerce.order_validation topic (Email Agent input)
    echo "--- Order Validation Events ---"
    timeout 10s docker exec kafka kafka-console-consumer \
        --topic commerce.order_validation \
        --bootstrap-server localhost:9092 \
        --from-beginning \
        --max-messages 2
    
    echo ""
    echo "--- SendGrid Email Requests ---"
    # Check sendgrid.email_requests topic (Email Agent output)
    timeout 10s docker exec kafka kafka-console-consumer \
        --topic sendgrid.email_requests \
        --bootstrap-server localhost:9092 \
        --from-beginning \
        --max-messages 2
}

test_correlation_tracking() {
    echo "=== Testing Correlation ID Tracking ==="
    
    # Get a correlation_id from an existing order
    CORRELATION_ID=$(docker exec AgenticAIStackDB psql -U AgenticAIStackDB -d AgenticAIStackDB -t -c "
    SELECT correlation_id FROM orders WHERE order_number = 'TEST-001';
    " | xargs)
    
    echo "Tracking correlation_id: $CORRELATION_ID"
    
    # Create a related query with the same correlation_id
    docker exec AgenticAIStackDB psql -U AgenticAIStackDB -d AgenticAIStackDB -c "
    INSERT INTO queries (user_id, query_text, status, correlation_id, related_order_id) 
    VALUES (
        (SELECT user_id FROM orders WHERE order_number = 'TEST-001' LIMIT 1),
        'Where is my order TEST-001?',
        'pending',
        '$CORRELATION_ID',
        (SELECT id FROM orders WHERE order_number = 'TEST-001' LIMIT 1)
    );
    "
    
    echo "Created related query. Checking correlation tracking..."
    sleep 5
    
    # Check correlation tracking view
    timeout 15s docker exec kafka kafka-console-consumer \
        --topic views.correlation_tracking \
        --bootstrap-server localhost:9092 \
        --from-beginning \
        --max-messages 3
}

monitor_all_topics() {
    echo "=== Monitoring All Topics ==="
    
    topics=(
        "customer.query_events" 
        "commerce.order_events" 
        "commerce.order_validation"
        "sendgrid.email_requests"
        "commerce.order_item_events"
        "customer.login_events"
        "ref.users" 
        "ref.products"
        "relational.orders" 
        "relational.queries"
        "views.active_orders_per_user"
        "views.correlation_tracking"
        "views.email_metrics"
    )
    
    for topic in "${topics[@]}"; do
        echo "--- Messages in $topic topic ---"
        timeout 5s docker exec kafka kafka-console-consumer \
            --topic $topic \
            --bootstrap-server localhost:9092 \
            --from-beginning \
            --max-messages 3 2>/dev/null || echo "No messages in $topic"
        echo ""
    done
}

check_ksqldb_streams() {
    echo "=== Checking ksqlDB Streams ==="
    
    # Connect to ksqlDB and show streams
    docker exec ksqldb-server ksql http://localhost:8088 <<EOF
SHOW STREAMS;
SHOW TABLES;
SELECT * FROM orders_stream EMIT CHANGES LIMIT 5;
EOF
}

generate_sample_data() {
    echo "=== Generating Sample Data for Testing ==="
    
    # Create more test users
    docker exec postgres psql -U postgres -d ecommerce -c "
    INSERT INTO users (email, password_hash, first_name, last_name, phone) VALUES
    ('alice.johnson@example.com', '$2b$12$hash', 'Alice', 'Johnson', '555-0125'),
    ('bob.wilson@example.com', '$2b$12$hash', 'Bob', 'Wilson', '555-0126'),
    ('carol.brown@example.com', '$2b$12$hash', 'Carol', 'Brown', '555-0127');
    "
    
    # Create test addresses
    docker exec postgres psql -U postgres -d ecommerce -c "
    INSERT INTO user_addresses (user_id, address, city, state, postal_code, country) 
    SELECT id, '123 Main St', 'Anytown', 'CA', '12345', 'USA' FROM users WHERE email LIKE '%johnson%';
    "
    
    # Create multiple orders
    docker exec postgres psql -U postgres -d ecommerce -c "
    WITH user_ids AS (SELECT id FROM users LIMIT 3)
    INSERT INTO orders (order_number, user_id, status, total_amount, currency) 
    SELECT 
        'ORD-' || generate_random_uuid()::text,
        user_ids.id,
        CASE (random() * 4)::int 
            WHEN 0 THEN 'pending'
            WHEN 1 THEN 'confirmed' 
            WHEN 2 THEN 'shipped'
            ELSE 'delivered'
        END,
        (random() * 2000 + 100)::numeric(10,2),
        'USD'
    FROM user_ids, generate_series(1, 5);
    "
    
    # Create order items
    docker exec postgres psql -U postgres -d ecommerce -c "
    INSERT INTO order_items (order_id, product_id, quantity, unit_price)
    SELECT 
        o.id,
        p.id,
        (random() * 3 + 1)::int,
        p.price
    FROM orders o
    CROSS JOIN products p
    WHERE o.order_number LIKE 'ORD-%'
    LIMIT 10;
    "
    
    # Create queries
    docker exec postgres psql -U postgres -d ecommerce -c "
    INSERT INTO queries (user_id, query_text, status)
    SELECT 
        id,
        CASE (random() * 3)::int
            WHEN 0 THEN 'Where is my order?'
            WHEN 1 THEN 'I want to return this product'
            ELSE 'Can you help me with product recommendations?'
        END,
        'pending'
    FROM users
    LIMIT 5;
    "
    
    echo "Sample data generated successfully!"
}

# Main test execution
echo "Starting pipeline tests..."
echo "================================"

# Generate sample data first
generate_sample_data
sleep 10

# Run tests
test_orders_flow
echo ""

test_queries_flow
echo ""

test_order_status_update
echo ""

test_correlation_tracking
echo ""

monitor_all_topics
echo ""

# Show final status
echo "=== Pipeline Status ==="
echo "Connector Status:"
curl -s http://localhost:8083/connectors/postgres-source-connector/status | jq '.connector.state, .tasks[0].state'

echo ""
echo "Available Topics:"
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092 | grep -E "(customer\.|commerce\.|ref\.|relational\.|views\.|dlq\.|sendgrid\.)"

echo ""
echo "Test completed! Your agentic AI pipeline with correlation tracking is ready."
echo ""
echo "📋 To consume messages for your agents:"
echo "Message Agent:      docker exec kafka kafka-console-consumer --topic customer.query_events --bootstrap-server localhost:9092"
echo "Order Agent:        docker exec kafka kafka-console-consumer --topic commerce.order_events --bootstrap-server localhost:9092"
echo "Email Agent Input:  docker exec kafka kafka-console-consumer --topic commerce.order_validation --bootstrap-server localhost:9092"
echo "Email Agent Output: docker exec kafka kafka-console-consumer --topic sendgrid.email_requests --bootstrap-server localhost:9092"
echo ""
echo "🔗 To monitor correlation tracking:"
echo "Journey Tracking:   docker exec kafka kafka-console-consumer --topic views.correlation_tracking --bootstrap-server localhost:9092"
echo "Email Metrics:      docker exec kafka kafka-console-consumer --topic views.email_metrics --bootstrap-server localhost:9092"
echo ""
echo "📊 To monitor data flows:"
echo "Raw data:           docker exec kafka kafka-console-consumer --topic raw.orders --bootstrap-server localhost:9092"
echo "Enriched data:      docker exec kafka kafka-console-consumer --topic relational.orders --bootstrap-server localhost:9092"
echo "Active orders view: docker exec kafka kafka-console-consumer --topic views.active_orders_per_user --bootstrap-server localhost:9092"
echo ""
echo "🚨 Error monitoring:"
echo "Processing errors:  docker exec kafka kafka-console-consumer --topic dlq.processing_errors --bootstrap-server localhost:9092"
echo "Validation errors:  docker exec kafka kafka-console-consumer --topic dlq.validation_errors --bootstrap-server localhost:9092"
echo "Email failures:     docker exec kafka kafka-console-consumer --topic dlq.email_failures --bootstrap-server localhost:9092" kafka-console-consumer --topic dlq.validation_errors --bootstrap-server localhost:9092"