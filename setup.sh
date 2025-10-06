#!/bin/bash

# Complete setup script for PostgreSQL + Kafka + ksqlDB integration
# Updated for priority-segmented topic architecture

echo "Setting up PostgreSQL + Kafka + ksqlDB for Agentic AI Pipeline"

# 1. Start Docker containers
echo "Starting Docker containers..."
docker compose up -d

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
      "table.include.list": "public.users,public.user_addresses,public.products,public.orders,public.order_items,public.queries,public.email_notifications",
      "topic.prefix": "AgenticAIStack",
      "key.converter": "org.apache.kafka.connect.json.JsonConverter",
      "key.converter.schemas.enable": "false",
      "value.converter": "org.apache.kafka.connect.json.JsonConverter",
      "value.converter.schemas.enable": "false",
      "transforms": "route",
      "transforms.route.type": "org.apache.kafka.connect.transforms.RegexRouter",
      "transforms.route.regex": "ecommerce_db.public.([^.]+)",
      "transforms.route.replacement": "raw.$1",
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

# 6. Create Kafka topics for priority-based architecture
echo "Creating Kafka topics..."

# ============================================================================
# CORE INFRASTRUCTURE TOPICS
# ============================================================================

# Single ingress topic (orchestrator consumes from here)
docker exec kafka kafka-topics --create --topic system.ingress --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# Agent response topic
docker exec kafka kafka-topics --create --topic agent.responses --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# ============================================================================
# PRIORITY-SEGMENTED AGENT TOPICS (P1=Critical, P2=High, P3=Normal)
# ============================================================================

# Order Agent Priority Topics
docker exec kafka kafka-topics --create --topic tasks.order.p1 --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic tasks.order.p2 --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic tasks.order.p3 --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# Email Agent Priority Topics
docker exec kafka kafka-topics --create --topic tasks.email.p1 --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic tasks.email.p2 --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic tasks.email.p3 --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# Policy Agent Priority Topics
docker exec kafka kafka-topics --create --topic tasks.policy.p1 --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic tasks.policy.p2 --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic tasks.policy.p3 --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# Message Agent Priority Topics
docker exec kafka kafka-topics --create --topic tasks.message.p1 --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic tasks.message.p2 --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic tasks.message.p3 --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# ============================================================================
# RAW DATA TOPICS (from Debezium CDC)
# ============================================================================

docker exec kafka kafka-topics --create --topic raw.users --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
docker exec kafka kafka-topics --create --topic raw.user_addresses --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
docker exec kafka kafka-topics --create --topic raw.products --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
docker exec kafka kafka-topics --create --topic raw.orders --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic raw.order_items --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic raw.queries --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic raw.email_notifications --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# ============================================================================
# REFERENCE DATA TOPICS (compacted for latest state)
# ============================================================================

docker exec kafka kafka-topics --create --topic ref.users --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1 --config cleanup.policy=compact
docker exec kafka kafka-topics --create --topic ref.products --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1 --config cleanup.policy=compact

# ============================================================================
# RELATIONAL/ENRICHED TOPICS (ksqlDB outputs)
# ============================================================================

docker exec kafka kafka-topics --create --topic relational.orders --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic relational.order_items --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic relational.queries --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# ============================================================================
# EMAIL & EXTERNAL INTEGRATION TOPICS
# ============================================================================

docker exec kafka kafka-topics --create --topic sendgrid.email_requests --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# ============================================================================
# ANALYTICS & MONITORING TOPICS
# ============================================================================

docker exec kafka kafka-topics --create --topic views.active_orders_per_user --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic views.priority_distribution --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic views.agent_performance --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic views.correlation_tracking --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --create --topic views.email_metrics --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# ============================================================================
# DEAD LETTER QUEUE TOPICS
# ============================================================================

docker exec kafka kafka-topics --create --topic dlq.processing_errors --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
docker exec kafka kafka-topics --create --topic dlq.validation_errors --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
docker exec kafka kafka-topics --create --topic dlq.email_failures --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

# 7. Verify setup
echo ""
echo "============================================================================"
echo "VERIFYING SETUP"
echo "============================================================================"

echo ""
echo "Checking Kafka Connect status..."
curl -s http://localhost:8083/connectors/postgres-source-connector/status | jq '.'

echo ""
echo "Listing all Kafka topics..."
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

echo ""
echo "Checking connector tasks..."
curl -s http://localhost:8083/connectors | jq '.'

echo ""
echo "============================================================================"
echo "SETUP COMPLETE"
echo "============================================================================"
echo ""
echo "Topic Architecture Summary:"
echo "  - Ingress: system.ingress (orchestrator consumes)"
echo "  - Order Agent: tasks.order.p1, tasks.order.p2, tasks.order.p3"
echo "  - Email Agent: tasks.email.p1, tasks.email.p2, tasks.email.p3"
echo "  - Policy Agent: tasks.policy.p1, tasks.policy.p2, tasks.policy.p3"
echo "  - Message Agent: tasks.message.p1, tasks.message.p2, tasks.message.p3"
echo "  - Responses: agent.responses"
echo ""
echo "Priority Levels:"
echo "  P1 = Critical (errors, urgent issues)"
echo "  P2 = High (orders, account changes)"
echo "  P3 = Normal (general queries)"
echo ""
echo "Next steps:"
echo "  1. Run ksqlDB queries to set up stream processing"
echo "  2. Start your Python orchestrator/agent application"
echo "  3. Monitor with: docker exec kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic agent.responses"
echo ""