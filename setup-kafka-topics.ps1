# Setup Kafka Topics for Agentic AI Stack
Write-Host "🚀 Setting up Kafka topics for Agentic AI Stack..." -ForegroundColor Green

# Wait for Kafka to be ready
Write-Host "⏳ Waiting for Kafka to be ready..." -ForegroundColor Yellow
do {
    try {
        docker exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092 | Out-Null
        $kafkaReady = $true
    }
    catch {
        Write-Host "Waiting for Kafka..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
        $kafkaReady = $false
    }
} while (-not $kafkaReady)

Write-Host "✅ Kafka is ready!" -ForegroundColor Green

# Create topics
Write-Host "📝 Creating Kafka topics..." -ForegroundColor Cyan

# Customer query events
docker exec kafka kafka-topics --create --topic customer.query_events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 --if-not-exists

# Commerce order events
docker exec kafka kafka-topics --create --topic commerce.order_events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 --if-not-exists

# Commerce order validation
docker exec kafka kafka-topics --create --topic commerce.order_validation --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 --if-not-exists

# Customer policy queries
docker exec kafka kafka-topics --create --topic customer.policy_queries --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 --if-not-exists

# Policy order events
docker exec kafka kafka-topics --create --topic policy.order_events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 --if-not-exists

# Agent responses
docker exec kafka kafka-topics --create --topic agent.responses --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 --if-not-exists

# System events
docker exec kafka kafka-topics --create --topic system.events --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 --if-not-exists

Write-Host "✅ All Kafka topics created successfully!" -ForegroundColor Green

# List topics to verify
Write-Host "📋 Listing all topics:" -ForegroundColor Cyan
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

Write-Host "🎉 Kafka setup complete!" -ForegroundColor Green
