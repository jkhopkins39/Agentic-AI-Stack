import threading
import time
import json
import random
from faker import Faker
from kafka import KafkaProducer

# -----------------------------
# Kafka Producer Setup
# -----------------------------
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],  # make into actual kafka broker
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# shared control flags, admin can pause/resume
control_flags = {
    "login": True,
    "order": True
}


# -----------------------------
# Worker Function
# -----------------------------
# system creates independent threads for each event type and creates dummy data events per event type
def worker(event_type, interval):
    fake = Faker()  # each thread gets its own faker instance

    while True:
        # check if this stream is paused by admin
        if not control_flags[event_type]:
            time.sleep(1)
            continue

        if event_type == "login":
            event = {
                "event_type": "login",
                "user_id": fake.uuid4(),
                "timestamp": fake.date_time_this_year().isoformat(),
                "device": random.choice(["mobile", "desktop"]),
                "location": f"{fake.city()}, {fake.country()}"
            }
            producer.send("logins", event)

        elif event_type == "order":
            event = {
                "event_type": "order",
                "order_id": fake.uuid4(),
                "user_id": fake.uuid4(),
                "timestamp": fake.date_time_this_year().isoformat(),
                "product": fake.word(),
                "quantity": random.randint(1, 5),
                "price": round(random.uniform(10, 100), 2)
            }
            producer.send("orders", event)

        print(f"[{event_type}] Sent: {event}")
        time.sleep(interval)


# -----------------------------
# Admin Control Loop
# -----------------------------
#here is where the admin can control the data-gen stream for the demo
def admin_control():
    while True:
        cmd = input("Admin Command (pause/resume/quit) > ").strip().lower()
        if cmd == "pause":
            stream = input("Which stream? (login/order/all) > ").strip().lower()
            if stream == "all":
                control_flags["login"] = False
                control_flags["order"] = False
            elif stream in control_flags:
                control_flags[stream] = False
            print(f"✅ Paused {stream} stream")
        elif cmd == "resume":
            stream = input("Which stream? (login/order/all) > ").strip().lower()
            if stream == "all":
                control_flags["login"] = True
                control_flags["order"] = True
            elif stream in control_flags:
                control_flags[stream] = True
            print(f"▶️ Resumed {stream} stream")
        elif cmd == "quit":
            print("👋 Shutting down admin control")
            break


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Start generator threads
    threads = [
        threading.Thread(target=worker, args=("login", 2)), #logins
        threading.Thread(target=worker, args=("order", 5)), #orders
        threading.Thread(target=admin_control)
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()
