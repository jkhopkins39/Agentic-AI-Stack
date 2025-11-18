#!/bin/bash

# Setup Kafka Topics for Agentic AI Stack
echo "🚀 Setting up Kafka topics for Agentic AI Stack..."

# Wait for Kafka to be ready
echo "⏳ Waiting for Kafka to be ready..."
until docker exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092 > /dev/null 2>&1; do
  echo "Waiting for Kafka..."
  sleep 5
done

echo "✅ Kafka is ready!"

# Create topics
echo "📝 Creating Kafka topics..."

# Customer query events
docker exec kafka kafka-topics --create \
  --topic customer.query_events \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1 \
  --if-not-exists

# Commerce order events
docker exec kafka kafka-topics --create \
  --topic commerce.order_events \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1 \
  --if-not-exists

# Commerce order validation
docker exec kafka kafka-topics --create \
  --topic commerce.order_validation \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1 \
  --if-not-exists

# Customer policy queries
docker exec kafka kafka-topics --create \
  --topic customer.policy_queries \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1 \
  --if-not-exists

# Policy order events
docker exec kafka kafka-topics --create \
  --topic policy.order_events \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1 \
  --if-not-exists

# Agent responses
docker exec kafka kafka-topics --create \
  --topic agent.responses \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1 \
  --if-not-exists

# System events
docker exec kafka kafka-topics --create \
  --topic system.events \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1 \
  --if-not-exists

echo "✅ All Kafka topics created successfully!"

# List topics to verify
echo "📋 Listing all topics:"
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

echo "🎉 Kafka setup complete!"
