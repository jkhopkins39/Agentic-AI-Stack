# kafka_producer_burst.py
import time
import argparse
from confluent_kafka import Producer
from faker import Faker
import json

fake = Faker()

def delivery_report(err, msg):
    if err is not None:
        print("Delivery failed:", err)

def run(broker, topic, total_msgs=100000, rate_per_sec=5000):
    p = Producer({"bootstrap.servers": broker})
    interval = 1.0 / rate_per_sec
    sent = 0
    start = time.time()
    try:
        while sent < total_msgs:
            # create realistic event payload
            payload = {
                "event_id": fake.uuid4(),
                "timestamp": fake.iso8601(),
                "customer": {"id": fake.uuid4(), "email": fake.email()},
                "order": {"id": fake.uuid4(), "amount": round(fake.pyfloat(2, 2, 1, 1000), 2)},
                "message": fake.sentence(nb_words=8)
            }
            p.produce(topic, value=json.dumps(payload), callback=delivery_report)
            p.poll(0)  # serve delivery callbacks
            sent += 1
            # throttle to hit approximate rate
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    p.flush()
    elapsed = time.time() - start
    print(f"Sent {sent} messages in {elapsed:.2f}s -> {sent/elapsed:.2f} msg/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default="localhost:9092")
    parser.add_argument("--topic", default="events")
    parser.add_argument("--total", type=int, default=10000)
    parser.add_argument("--rate", type=int, default=1000)
    args = parser.parse_args()
    run(args.broker, args.topic, args.total, args.rate)
