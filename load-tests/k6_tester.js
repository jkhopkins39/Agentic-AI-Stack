// k6_script.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter } from 'k6/metrics';
import { Rate } from 'k6/metrics';

export let errorRate = new Rate("errors");

export let options = {
  stages: [
    { duration: '1m', target: 100 },  // ramp to 100 VUs
    { duration: '3m', target: 500 },  // ramp to 500
    { duration: '5m', target: 1000 }, // stress
    { duration: '2m', target: 0 },    // ramp down
  ],
  thresholds: {
    "http_req_duration": ["p(95) < 1500"],  // p95 < 1.5s
    "errors": ["rate<0.01"],                // errors < 1%
  }
};

const BASE = __ENV.BASE_URL || "http://your-api-host:8000";

function login() {
  let payload = JSON.stringify({
    username: `user_${Math.floor(Math.random()*1000000)}`,
    password: "TestPass123!"
  });
  let res = http.post(`${BASE}/api/login`, payload, { headers: { 'Content-Type': 'application/json' }});
  return res;
}

export default function () {
  // simulate user flow: login -> get dashboard -> post message
  let res = login();
  check(res, { "login 200": (r) => r.status === 200 });
  let token = "";
  if (res.status === 200 && res.json("access_token")) {
    token = res.json("access_token");
  }
  let headers = { headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" }};
  let dash = http.get(`${BASE}/api/dashboard`, headers);
  check(dash, { "dashboard 200": (r) => r.status === 200 });

  let msg = JSON.stringify({
    customer_id: `${Math.random()}`,
    order_id: `${Math.random()}`,
    message: "This is a synthetic test message."
  });
  let post = http.post(`${BASE}/api/messages`, msg, headers);
  check(post, { "message post 200": (r) => r.status === 200 });
  if (post.status !== 200) { errorRate.add(1); }
  sleep(Math.random() * 3); // think time
}
