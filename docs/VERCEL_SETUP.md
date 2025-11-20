# Vercel Frontend Deployment

Quick guide to deploy the frontend to Vercel and connect it to your Render backend.

## Prerequisites

- ✅ Backend deployed on Render (with Kafka + Database working)
- ✅ GitHub repository connected to Vercel
- ✅ Render backend URL (e.g., `https://your-backend.onrender.com`)

---

## Deployment Steps

### 1. Connect Repository
1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click **"Add New Project"**
3. Import your GitHub repository
4. Select the repository

### 2. Configure Project Settings

**Root Directory:**
```
frontend/
```

**Build Command:**
```
npm run build
```
(Default - Vite handles this automatically)

**Output Directory:**
```
dist
```
(Default - Vite's output directory)

**Install Command:**
```
npm install --legacy-peer-deps
```
(Use `--legacy-peer-deps` to handle React 19 compatibility issues)

### 3. Environment Variables

Add this **one** environment variable:

**Variable Name:** `VITE_API_URL`  
**Value:** `https://your-backend.onrender.com`

**Important:**
- Replace `your-backend.onrender.com` with your actual Render backend URL
- **No trailing slash** (e.g., `https://backend.onrender.com` not `https://backend.onrender.com/`)
- Must start with `https://` (Vercel uses HTTPS)

### 4. Deploy

Click **"Deploy"** - Vercel will:
1. Install dependencies (`npm install`)
2. Build the app (`npm run build`)
3. Deploy to a Vercel URL

---

## How It Works

### Message Flow:
1. **User types in frontend** → Sends to `/publish/ingress` endpoint
2. **Backend receives** → Publishes to `system.ingress` Kafka topic
3. **Orchestrator routes** → Routes to priority topics (`tasks.*.p*`)
4. **Agents process** → Order/Email/Policy/Message agents handle the request
5. **Database queries** → Agents query Render PostgreSQL as needed
6. **Response published** → Agent response sent to `agent.responses` topic
7. **WebSocket delivers** → Frontend receives response via WebSocket
8. **User sees response** → Message appears in chat UI

### WebSocket Connection:
- Frontend automatically converts `https://` → `wss://` for WebSocket
- Connects to: `wss://your-backend.onrender.com/ws/agent-responses/{session_id}`
- No additional configuration needed

---

## Testing After Deployment

1. **Open your Vercel URL** (e.g., `https://your-app.vercel.app`)
2. **Login** with test user (e.g., `john.doe@example.com` / `password123`)
3. **Send a message** like:
   - "What is my order status?" (routes to Order agent)
   - "What is your return policy?" (routes to Policy agent)
   - "Send me an email" (routes to Email agent)
4. **Verify response** appears in chat
5. **Check Render logs** to see Kafka message flow
6. **Check Confluent Cloud** to see messages in topics

---

## Troubleshooting

### Frontend can't connect to backend
- Verify `VITE_API_URL` is set correctly (no trailing slash)
- Check backend is running on Render
- Check CORS settings in backend (should allow Vercel domain)

### WebSocket not connecting
- Verify backend URL uses `https://` (Vercel requires HTTPS)
- Check Render WebSocket is enabled
- Check browser console for WebSocket errors

### Messages not appearing
- Check Render logs for Kafka errors
- Verify Confluent Cloud topics exist
- Check API key has proper ACLs (see `TESTING.md`)

---

## Quick Checklist

- [ ] Repository connected to Vercel
- [ ] Root directory set to `frontend/`
- [ ] `VITE_API_URL` environment variable set (your Render backend URL)
- [ ] Deployed successfully
- [ ] Can access Vercel URL
- [ ] Can login
- [ ] Can send messages
- [ ] Receive responses via WebSocket
- [ ] Messages flow through Kafka (check Render logs)
- [ ] Database queries work (check Render logs)

---

## Environment Variable Format

```
VITE_API_URL=https://agentic-ai-stack.onrender.com
```

**Note:** Vite automatically makes `VITE_*` variables available to your app at build time.

