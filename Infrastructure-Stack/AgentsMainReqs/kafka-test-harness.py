"""
Kafka Test Harness
==================
This script is used to validate that Kafka is working correctly.
It produces test messages to a topic and then consumes them back.
"""

from kafka import KafkaProducer, KafkaConsumer
import json
import time

TOPIC = "test_topic"
BOOTSTRAP_SERVERS = "localhost:9092"  # adjust if in docker-compose

def run_producer(messages=5):
    """Produce test messages to Kafka."""
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    for i in range(messages):
        msg = {"id": i, "text": f"hello-{i}"}
        producer.send(TOPIC, msg)
        print(f"[Producer] Sent: {msg}")
    producer.flush()
    producer.close()

def run_consumer(timeout=10):
    """Consume messages from Kafka."""
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8"))
    )
    start = time.time()
    print("[Consumer] Waiting for messages...")
    for message in consumer:
        print(f"[Consumer] Received: {message.value}")
        if time.time() - start > timeout:
            break
    consumer.close()

def run_test():
    print("🚀 Running Kafka Test Harness...")
    run_producer(messages=5)
    run_consumer(timeout=10)
    print("✅ Test complete.")

if __name__ == "__main__":
    run_test()
