# 🤖 Multi-Agent LLM Customer Service Stack

### Backend (please edit as needed)
- **Python** with FastAPI
- **LangGraph** for agent orchestration
- **Anthropic Claude** as the LLM provider
- **PostgreSQL** for data persistence
- **Redis** for session management and caching

### Frontend (please edit as needed)
- **React 19** with TypeScript
- **Vite 7** for build tooling
- **TailwindCSS** for styling
- **Axios** for API communication

### Infrastructure
- **Docker** & **Docker Compose** for containerization
- **Node.js 20** runtime environment

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
ANTHROPIC_API_KEY=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
OPENAI_API_KEY=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

```

### 3. Backend Setup (Docker)
```bash
# Start backend services (database, redis, and backend API)
docker-compose up
```

The backend will be available at: http://localhost:8000

### 4. Frontend Setup (Local Development) in a separate terminal

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

## Usage

```

### Full Docker Stack (when frontend Docker is fixed)
```bash
# Start all services
docker-compose up

# Or run in background
docker-compose up -d

# View logs
docker-compose logs -f
```
