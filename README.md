# Multi-Agent LLM Customer Service Stack

### Backend (please edit as needed)
- **Python** with FastAPI
- **LangGraph** for agent orchestration
- **Anthropic Claude** as the LLM provider
- **PostgreSQL** for data persistence
- **Redis** for session management and caching
- **Apache Kafka** for event streaming and real-time processing

### Frontend (please edit as needed)
- **React 19** with TypeScript
- **Vite 7** for build tooling
- **TailwindCSS** for styling
- **Axios** for API communication

### Infrastructure
- **Docker** & **Docker Compose** for containerization
- **Node.js 20** runtime environment
- **Apache Kafka** with Zookeeper for event streaming

## Quick Start

### 1. Clone the Repository
```bash
git clone 
cd multi-agent-llm-frontend
```

### 2. Environment Setup
Create a `.env` file in the root directory:
```bash
POSTGRES_DB=agent_system
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password123
# Get your API keys from:
# OpenAI: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-proj-your-actual-openai-key-here

# Anthropic: https://console.anthropic.com/settings/keys  
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key-here

# SendGrid: https://app.sendgrid.com/settings/api_keys
SENDGRID_API_KEY=SG.your-sendgrid-api-key-here

```

### 3. Backend Setup (Docker)
```bash
# Start all services including Kafka
docker-compose up -d

# Wait for services to be ready, then setup Kafka topics
# On Windows:
.\setup-kafka-topics.ps1

# On Linux/Mac:
./setup-kafka-topics.sh
```

The backend will be available at: http://localhost:8000

### 4. Kafka Integration
The system now includes **Apache Kafka** for event streaming:

**Kafka Topics:**
- `customer.query_events` - Customer queries and interactions
- `commerce.order_events` - Order-related events and updates
- `commerce.order_validation` - Order validation events
- `customer.policy_queries` - Policy-related queries
- `policy.order_events` - Policy and order cross-references
- `agent.responses` - Agent response events
- `system.events` - System-wide events

**Event Tracking:**
- Every user interaction gets a `correlation_id` for end-to-end tracking
- All agent responses are published to Kafka topics
- Events include priority levels (1=urgent, 2=high, 3=medium, 4=low)

### 5. Frontend Setup (Local Development) in a separate terminal

 **Note**: Frontend Docker configuration is currently under development. Please use local development setup.

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Install TailwindCSS Vite plugin
npm install @tailwindcss/vite

# Start development server
npm run dev
```

The frontend will be available at: http://localhost:5173 in your browser

## Access Points

- **Frontend Application**: http://localhost:5173
- **Backend API**: http://localhost:8000

## Note that docker compose up will have to be running in one window, or docker compose up -d, and then npm run dev will have to be running as well. If you use the -d flag then you can run npm run dev in the same terminal window.
