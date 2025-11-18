# Agentic AI Customer Service Stack

A production-ready multi-agent LLM system for intelligent customer service, built with event-driven architecture using Kafka, LangGraph agents, and real-time WebSocket communication.

READ THIS TO RUN IT

## 🏗️ Architecture

### Backend Stack
- **Python 3.11** with FastAPI for REST APIs and WebSocket support
- **LangGraph** for multi-agent orchestration and workflow management
- **Anthropic Claude & OpenAI GPT** for intelligent agent responses
- **Apache Kafka** for event streaming and message routing with priority queues
- **PostgreSQL** for persistent data storage (users, orders, conversations)
- **Redis** for session management and caching
- **Apache Kafka** for event streaming and real-time processing

### Frontend Stack
- **React 19** with TypeScript
- **Vite 7** for blazing-fast development and build tooling
- **TailwindCSS 4** for modern styling
- **Axios** for HTTP requests and WebSocket communication

### Infrastructure
- **Docker & Docker Compose** for containerized deployment
- **Node.js 20** runtime environment
- **Apache Kafka** with Zookeeper for event streaming

## Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Environment Setup

Your `.env` file should already be configured. If not, ensure it contains:

```env
# Database Configuration
DB_HOST=AgenticAIStackDB
DB_PORT=5432
DB_NAME=AgenticAIStackDB
DB_USER=AgenticAIStackDB
DB_PASSWORD=your-password-here

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# API Keys (Required)
ANTHROPIC_API_KEY=your-anthropic-key-here
OPENAI_API_KEY=your-openai-key-here
SENDGRID_API_KEY=your-sendgrid-key-here

# Session Configuration
SESSION_EMAIL=your-email@example.com
```

### 3. Run Setup Script

The setup script initializes Kafka topics, configures Debezium connectors, and sets up the database:

```bash
./setup.sh
```

### 4. Build and Start Backend

```bash
# Start all services including Kafka
docker-compose up -d

# Wait for services to be ready, then setup Kafka topics
# On Windows:
.\setup-kafka-topics.ps1

# On Linux/Mac:
./setup-kafka-topics.sh
```

**Note:** This will start all services including Kafka, PostgreSQL, Redis, and the backend API.

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

### 5. Start Frontend (in a separate terminal)

```bash
cd frontend
npm run dev
```

The frontend will be available at: **http://localhost:5173**

## 🎯 Access Points

- **Frontend Application**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **Kafka UI (if configured)**: http://localhost:8080
- **ksqlDB Server**: http://localhost:8088

## 🧠 Agent System

The stack includes 4 specialized agents:

1. **Order Agent** - Handles order inquiries, tracking, and status updates
2. **Email Agent** - Manages email notifications and receipt delivery
3. **Policy Agent** - Uses RAG to answer policy-related questions
4. **Message Agent** - Handles general queries and information updates

### Priority-Based Routing

Messages are routed to priority queues (P1/P2/P3) based on urgency:
- **P1 (Critical)**: Urgent issues, errors, system failures
- **P2 (High)**: Orders, account changes, important requests
- **P3 (Normal)**: General inquiries, policy questions

## 📁 Project Structure

```
.
├── main.py                 # FastAPI backend with agent logic
├── docker-compose.yml      # Container orchestration
├── Dockerfile             # Backend container definition
├── setup.sh               # Infrastructure setup script
├── policy.txt             # Policy document for RAG
├── schema.sql             # Database schema
├── frontend/              # React frontend
│   ├── src/              # Source files
│   ├── package.json      # Frontend dependencies
│   └── vite.config.js    # Vite configuration
└── README.md             # This file
```

## 🔧 Development

### Running in Development Mode

For hot-reload during development:

```bash
# Terminal 1: Backend with auto-reload
docker-compose up backend

# Terminal 2: Frontend with Vite HMR
cd frontend
npm run dev
```

### Running in Background

To run services in detached mode:

```bash
docker-compose up -d backend
cd frontend && npm run dev
```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f kafka
```

## 🧪 Testing

The frontend includes Vitest for testing:

```bash
cd frontend
npm run test          # Run tests
npm run test:ui       # Run tests with UI
```

## 🛑 Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

## 📊 Kafka Topics

The system uses the following topic structure:

- `system.ingress` - Entry point for all customer queries
- `tasks.{agent}.p{1,2,3}` - Priority queues for each agent type
- `agent.responses` - Agent responses to frontend

## 🔐 Security Notes

- **Never commit API keys** - Keep your `.env` file out of version control
- **Update default passwords** - Change all default database passwords
- **Use environment-specific configs** - Different settings for dev/staging/prod

## 🐛 Troubleshooting

### Backend won't start
- Check if ports 8000, 5432, 6379, 9092 are available
- Verify API keys are set in `.env`
- Run `docker-compose logs backend` for error details

### Frontend won't connect
- Ensure backend is running on port 8000
- Check `VITE_API_URL` in frontend environment
- Verify CORS settings in `main.py`

### Kafka connection issues
- Wait for Kafka to fully initialize (~30 seconds)
- Check `docker-compose logs kafka`
- Verify topics are created: `docker exec kafka kafka-topics --list --bootstrap-server localhost:9092`

## 📝 License

[Your License Here]

## 👥 Contributors

[Your Team/Contributors Here]

## 📧 Support

For issues and questions: agenticaistack@gmail.com
