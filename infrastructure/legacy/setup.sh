#!/bin/bash

# Complete setup script for PostgreSQL + Kafka + ksqlDB integration

echo "Setting up PostgreSQL + Kafka + ksqlDB for Agentic AI Pipeline"

# 1. Start Docker containers
echo "Starting Docker containers..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# 2. Configure PostgreSQL for logical replication
echo "Configuring PostgreSQL for logical replication..."
docker exec -it AgenticAIStackDB psql -U AgenticAIStackDB -d AgenticAIStackDB -c "
-- Create replication slot
SELECT pg_create_logical_replication_slot('debezium_slot', 'pgoutput');

-- Create publication for all tables
CREATE PUBLICATION AgenticAIStack_publication FOR ALL TABLES;

-- Grant necessary permissions
GRANT SELECT ON ALL TABLES IN SCHEMA public TO AgenticAIStackDB;
GRANT USAGE ON SCHEMA public TO AgenticAIStackDB;
"

# 3. Install Debezium PostgreSQL connector
echo "Installing Debezium PostgreSQL connector..."
docker exec -it connect bash -c "
# Download Debezium PostgreSQL connector
cd /usr/share/java
curl -L https://repo1.maven.org/maven2/io/debezium/debezium-connector-postgres/2.4.0.Final/debezium-connector-postgres-2.4.0.Final-plugin.tar.gz | tar xz
"

# Restart Connect to load the new connector
echo "Restarting Kafka Connect..."
docker restart connect
sleep 20

# 4. Create the PostgreSQL source connector
echo "Creating PostgreSQL source connector..."
curl -X POST http://localhost:8083/connectors \
  -H "Content-Type: application/json" \
  -d '{
    "name": "postgres-source-connector",
    "config": {
      "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
      "tasks.max": "1",
      "database.hostname": "AgenticAIStackDB",
      "database.port": "5432",
      "database.user": "AgenticAIStackDB",
      "database.password": "8%w=r?D52Eo2EwcVW:",
      "database.dbname": "AgenticAIStackDB",
      "database.server.name": "ecommerce_db",
      "plugin.name": "pgoutput",
      "slot.name": "debezium_slot",
      "publication.name": "AgenticAIStack_publication",
      "table.include.list": "public.users,public.user_addresses,public.products,public.orders,public.order_items,public.queries",
      "topic.prefix": "AgenticAIStack",
      "key.converter": "org.apache.kafka.connect.json.JsonConverter",
      "key.converter.schemas.enable": "false",
      "value.converter": "org.apache.kafka.connect.json.JsonConverter",
      "value.converter.schemas.enable": "false",
      "transforms": "route",
      "transforms.route.type": "org.apache.kafka.connect.transforms.RegexRouter",
      "transforms.route.regex": "ecommerce_db.public.([^.]+)",
      "transforms.route.replacement": "$1",
      "decimal.handling.mode": "double",
      "time.precision.mode": "adaptive_time_microseconds",
      "include.schema.changes": "false",
      "publication.autocreate.mode": "filtered"
    }
  }'

# 5. Start ksqlDB
echo "Starting ksqlDB..."
docker run -d --name ksqldb-server --network container:kafka \
  -e KSQL_BOOTSTRAP_SERVERS=localhost:9092 \
  -e KSQL_LISTENERS=http://0.0.0.0:8088 \
  confluentinc/ksqldb-server:0.29.0

sleep 15

# 6. Create Kafka topics for agents and data flow
echo "Creating Kafka topics..."

# Business event topics for agents
docker exec kafka kafka-topics --create --topic customer.login_events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic customer.query_events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic commerce.order_events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic commerce.order_validation --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic commerce.order_item_events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# Email agent topics
docker exec kafka kafka-topics --create --topic sendgrid.email_requests --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# Reference data topics
docker exec kafka kafka-topics --create --topic ref.users --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1 --config cleanup.policy=compact
docker exec kafka kafka-topics --create --topic ref.products --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1 --config cleanup.policy=compact

# Relational/enriched topics
docker exec kafka kafka-topics --create --topic relational.orders --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic relational.order_items --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic relational.queries --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# Materialized view topics
docker exec kafka kafka-topics --create --topic views.active_orders_per_user --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic views.correlation_tracking --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic views.email_metrics --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# Dead letter queue topics
docker exec kafka kafka-topics --create --topic dlq.processing_errors --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
docker exec kafka kafka-topics --create --topic dlq.validation_errors --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
docker exec kafka kafka-topics --create --topic dlq.email_failures --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

# 7. Verify setup
echo "Verifying setup..."

echo "Checking Kafka Connect status..."
curl -s http://localhost:8083/connectors/postgres-source-connector/status | jq '.'

echo "Listing available topics..."
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

echo "Checking connector tasks..."
curl -s http://localhost:8083/connectors | jq '.'

echo "Setup complete!"
echo ""
echo "Topic structure for your agentic AI with correlation tracking:"
echo "📋 BUSINESS EVENTS (for agents):"
echo "  - customer.query_events -> Message Agent (Priority 1)"
echo "  - commerce.order_events -> Order Agent (Priority 1-3)"
echo "  - commerce.order_validation -> Email Agent (Priority 1-3)"
echo "  - sendgrid.email_requests -> Email Agent Output to SendGrid"
echo "  - customer.login_events -> Logged events only"
echo ""  
echo "📊 DATA FLOWS:"
echo "  - raw.* -> Raw database changes from Debezium"
echo "  - ref.users, ref.products -> Master data (compacted)"
echo "  - relational.* -> Enriched/normalized data with correlation_id"
echo "  - views.* -> Materialized views for analytics and correlation tracking"
echo "  - dlq.* -> Dead letter queues for error handling"
echo ""
echo "🔗 CORRELATION TRACKING:"
echo "  - All events include correlation_id for end-to-end tracing"
echo "  - views.correlation_tracking -> Complete journey visibility"
echo "  - User, order, query, and email events can be linked"
echo ""
echo "Next steps:"
echo "1. Run ksqlDB queries from the ksqldb_queries.sql file"
echo "2. Test by inserting data into PostgreSQL tables"
echo "3. Monitor agent topics:"
echo "   - Message Agent: docker exec kafka kafka-console-consumer --topic customer.query_events --bootstrap-server localhost:9092"
echo "   - Order Agent: docker exec kafka kafka-console-consumer --topic commerce.order_events --bootstrap-server localhost:9092"
echo "   - Email Agent Input: docker exec kafka kafka-console-consumer --topic commerce.order_validation --bootstrap-server localhost:9092"
echo "   - Email Agent Output: docker exec kafka kafka-console-consumer --topic sendgrid.email_requests --bootstrap-server localhost:9092"
echo "4. Monitor correlation tracking: docker exec kafka kafka-console-consumer --topic views.correlation_tracking --bootstrap-server localhost:9092"