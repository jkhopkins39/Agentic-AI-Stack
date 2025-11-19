# Backend Testing Guide

Quick guide to test your deployed backend with Kafka and Confluent Cloud.

## Prerequisites

- Your Render backend URL (e.g., `https://your-backend.onrender.com`)
- `curl` installed (or use Postman/Insomnia)

---

## 1. Health Check

Test if the backend is running and connected to Kafka/Database:

```bash
curl https://your-backend.onrender.com/api/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-18T...",
  "database": "healthy",
  "kafka": "healthy"
}
```

**If `kafka: "unhealthy"`**: Check Render logs for Kafka connection errors.

---

## 2. Test Kafka Producer (Send Message to Topic)

Test sending a message to the `system.ingress` topic:

```bash
curl -X POST https://your-backend.onrender.com/publish/ingress \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session-123",
    "user_email": "test@example.com",
    "query_text": "What is my order status?",
    "conversation_id": null
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Published to system.ingress",
  "conversation_id": "uuid-here"
}
```

**Check Render Logs:**
- Should see: `→ Published ingress for test-session-123`
- Should see: `🔵 Orchestrator consumer started on system.ingress`
- Should see: `→ Routed test-session-123 to tasks.order.p1`

---

## 3. Test Full Flow (Message → Kafka → Agent → Response)

### Option A: Using WebSocket (Full End-to-End)

1. **Connect to WebSocket:**
   ```bash
   # Using wscat (install: npm install -g wscat)
   wscat -c wss://your-backend.onrender.com/ws/agent-responses/test-session-123
   ```

2. **In another terminal, send a message:**
   ```bash
   curl -X POST https://your-backend.onrender.com/publish/ingress \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "test-session-123",
       "user_email": "test@example.com",
       "query_text": "What is my order status?",
       "conversation_id": null
     }'
   ```

3. **Watch WebSocket for response:**
   - Should receive JSON with agent response
   - Format: `{"session_id": "...", "agent_type": "...", "response": "..."}`

### Option B: Check Confluent Cloud (Verify Messages in Topics)

1. Go to Confluent Cloud → Your Cluster → Topics
2. Check `system.ingress` topic - should see messages
3. Check `tasks.order.p1` (or other priority topics) - should see routed messages
4. Check `agent.responses` - should see agent responses

---

## 4. Test Different Message Types

### Order Query
```bash
curl -X POST https://your-backend.onrender.com/publish/ingress \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-order-123",
    "user_email": "john.doe@example.com",
    "query_text": "What is the status of order ORD-1001?",
    "conversation_id": null
  }'
```

### Email Query
```bash
curl -X POST https://your-backend.onrender.com/publish/ingress \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-email-123",
    "user_email": "test@example.com",
    "query_text": "Send me an email confirmation",
    "conversation_id": null
  }'
```

### Policy Query
```bash
curl -X POST https://your-backend.onrender.com/publish/ingress \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-policy-123",
    "user_email": "test@example.com",
    "query_text": "What is your return policy?",
    "conversation_id": null
  }'
```

**What to observe:**
1. **Render Logs** - Should see:
   - `→ Published ingress for test-policy-123`
   - `→ Routed test-policy-123 to tasks.policy.p1` (or p2/p3 based on priority)
   - `→ Policy agent: Consumed test-policy-123 from tasks.policy.p1`
   - `✓ Found X relevant chunks for policy query` (RAG system finding policy info)
   - `→ Published agent response for test-policy-123`

2. **Confluent Cloud Topics:**
   - `system.ingress` - Your original message
   - `tasks.policy.p1` (or p2/p3) - Routed message to policy agent
   - `agent.responses` - Policy agent's response

3. **WebSocket** (if connected):
   - Should receive response with `"agent_type": "POLICY_AGENT"`

**Other Policy Questions to Test:**
- "How long do I have to return an item?"
- "Can I return this product?"
- "What are your shipping policies?"
- "Do you accept returns?"
- "How do returns work?"

---

## 5. Monitor Render Logs

Watch your Render service logs in real-time:

1. Go to Render Dashboard → Your Backend Service → Logs
2. Look for:
   - `✓ Kafka producer started`
   - `🔵 Orchestrator consumer started on system.ingress`
   - `→ Published ingress for ...`
   - `→ Routed ... to tasks.*.p*`
   - `→ Order agent: Consumed ... from tasks.order.p1`
   - `→ Published agent response for ...`

---

## 6. Verify Kafka Topics in Confluent Cloud

1. **Go to Confluent Cloud** → Your Cluster → Topics
2. **Check these topics have messages:**
   - `system.ingress` - Incoming messages
   - `tasks.order.p1`, `tasks.order.p2`, `tasks.order.p3` - Order agent queues
   - `tasks.email.p1`, `tasks.email.p2`, `tasks.email.p3` - Email agent queues
   - `tasks.policy.p1`, `tasks.policy.p2`, `tasks.policy.p3` - Policy agent queues
   - `agent.responses` - Agent responses

3. **View messages:**
   - Click on a topic → "Messages" tab
   - Should see JSON messages with your test data

---

## 7. Test Database Connection

```bash
curl https://your-backend.onrender.com/api/user/profile?user_email=john.doe@example.com
```

**Expected:** User profile data (if user exists in database)

---

## 8. Quick Test Script

Save this as `test_backend.sh`:

```bash
#!/bin/bash

BACKEND_URL="https://your-backend.onrender.com"

echo "1. Testing Health Check..."
curl -s "$BACKEND_URL/api/health" | jq '.'

echo -e "\n2. Testing Kafka Producer..."
curl -s -X POST "$BACKEND_URL/publish/ingress" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-'$(date +%s)'",
    "user_email": "test@example.com",
    "query_text": "Hello, this is a test message",
    "conversation_id": null
  }' | jq '.'

echo -e "\n✅ Test complete! Check Render logs and Confluent Cloud topics."
```

**Run it:**
```bash
chmod +x test_backend.sh
./test_backend.sh
```

---

## Troubleshooting

### Health Check Shows `kafka: "unhealthy"`
- Check Render logs for Kafka connection errors
- Verify `KAFKA_BOOTSTRAP_SERVERS` is set correctly
- Verify `KAFKA_SASL_USERNAME` and `KAFKA_SASL_PASSWORD` are set
- Check Confluent Cloud cluster is running

### Messages Not Appearing in Topics
- Check Render logs for routing/orchestrator messages
- Verify topics exist in Confluent Cloud
- Check API key has WRITE permissions on topics

### TopicAuthorizationFailedError
**Error**: `TopicAuthorizationFailedError: system.ingress`

**Fix**: Your API key needs topic-level permissions. In Confluent Cloud:
1. Go to **Security** → **API Keys** → Select your API key
2. Click **"Edit ACLs"** or **"Manage ACLs"**
3. Add these ACLs for **ALL your topics**:
   - **Resource Type**: Topic
   - **Resource Name**: `system.ingress` (or use `*` for all topics)
   - **Operation**: READ, WRITE, DESCRIBE
   - **Permission**: ALLOW
4. Repeat for all topics:
   - `system.ingress`
   - `agent.responses`
   - `tasks.order.p1`, `tasks.order.p2`, `tasks.order.p3`
   - `tasks.email.p1`, `tasks.email.p2`, `tasks.email.p3`
   - `tasks.policy.p1`, `tasks.policy.p2`, `tasks.policy.p3`
   - `tasks.message.p1`, `tasks.message.p2`, `tasks.message.p3`

**Quick Fix**: Use `*` as resource name to grant permissions to all topics at once.

### No Agent Responses
- Check Render logs for agent consumer messages
- Verify agents are processing messages
- Check `agent.responses` topic in Confluent Cloud

### WebSocket Not Connecting
- Verify WebSocket is enabled in Render service settings
- Use `wss://` (not `ws://`) for HTTPS backends
- Check CORS settings if connecting from browser

---

## Next Steps

Once basic tests pass:
1. Test with your frontend application
2. Monitor Confluent Cloud metrics
3. Check Render service metrics
4. Test error handling (invalid messages, etc.)

