# locusttester.py
from locust import HttpUser, task, between, events
from faker import Faker
import json
import os

fake = Faker()

class WebsiteUser(HttpUser):
    host = "http://localhost:8000"
    wait_time = between(1, 4)

    def on_start(self):
        # create or login a user
        self.username = fake.user_name()
        self.email = fake.email()
        # attempt to sign up (idempotent if test DB reset between runs)
        resp = self.client.post("/api/signup", json={
            "username": self.username,
            "email": self.email,
            "password": "TestPass123!"
        })
        # login to get token (adjust according to your auth)
        login = self.client.post("/api/login", json={
            "username": self.username,
            "password": "TestPass123!"
        })
        if login.status_code == 200:
            self.token = login.json().get("access_token")
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = None
            self.headers = {}

    @task(4)
    def view_dashboard(self):
        self.client.get("/api/dashboard", headers=self.headers, name="/api/dashboard")

    @task(3)
    def send_message(self):
        # realistic message with order info
        payload = {
            "customer_id": fake.uuid4(),
            "order_id": fake.uuid4(),
            "message": fake.sentence(nb_words=12),
            "email": self.email
        }
        # this endpoint should produce to Kafka as part of your normal flow
        self.client.post("/api/messages", json=payload, headers=self.headers, name="/api/messages")

    @task(1)
    def fetch_history(self):
        self.client.get("/api/messages/history", headers=self.headers, name="/api/messages/history")
