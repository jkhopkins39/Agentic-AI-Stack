# Infrastructure Directory

This directory contains infrastructure configuration, deployment scripts, and setup files.

## Structure

- **`kafka/`** - Kafka topic setup scripts
  - `setup-kafka-topics.sh` - Unix/Mac script to create Kafka topics
  - `setup-kafka-topics.ps1` - Windows PowerShell script to create Kafka topics

- **`database/`** - Database schemas and configuration
  - `schema.sql` - Main database schema
  - `postgres-schema.sql` - PostgreSQL-specific schema
  - `pg_hba.conf` - PostgreSQL host-based authentication config
  - `postgresql.conf` - PostgreSQL configuration
  - `postgres-connector.json` - Debezium PostgreSQL connector config

- **`deploy/`** - Deployment configurations (to be added)
  - Future: `render.yaml` - Render deployment config
  - Future: `vercel.json` - Vercel deployment config

- **`docker-compose.yml`** - Local development Docker Compose configuration

- **`ksqldb-queries.sql`** - KSQL DB query definitions

- **`setup.sh`** - Infrastructure setup script

- **`legacy/`** - Legacy infrastructure files (from Infrastructure-Stack)

## Usage

### Local Development
```bash
# Start all services
cd infrastructure
docker-compose up -d

# Create Kafka topics
./kafka/setup-kafka-topics.sh
```

### Production Deployment
See `../docs/SYSTEM_ASSESSMENT.md` for deployment strategies.

