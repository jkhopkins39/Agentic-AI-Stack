## Directory Structure
```
LangGraph/
├── main.py                    # Entry point
├── utils/                     # Util funcs
│   ├── __init__.py
│   ├── validation.py          # Email/phone validation, input sanitization
│   ├── logging.py             # Query and Kafka message logging
│   └── kafka_retry.py         # Kafka retry logic with exp backoff
├── rag/                       # RAG
│   ├── __init__.py
│   ├── document_processor.py  # Doc loading, splitting, Chroma storage
│   └── query.py               # RAG query functions
├── database/                  # DB Ops
│   ├── __init__.py
│   ├── connection.py          # PostgreSQL connection management
│   ├── user_operations.py     # User lookup and updates
│   ├── order_operations.py    # Order queries (by number, email, product)
│   └── conversations.py       # Conversation tracking and history
├── notifications/             # Multi-channel notifications
│   ├── __init__.py
│   ├── email_notifications.py # Email sending
│   ├── sms_notifications.py   # SMS sending
│   ├── preferences.py         # User notif preferences
│   └── multi_channel.py       # Unified notif dispatcher
├── agents/                    # LangGraph
│   ├── __init__.py
│   ├── models.py              # Pydantic models for structured output
│   └── handlers.py            # Agent functions (classifier, order,etc)
└── graph/                     # LangGraph config
    ├── __init__.py
    └── graph_config.py        # Graph setup and chatbot runner
```


## Environment Variables Required
Ensure there exists an .env file in the project root with:

ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key

DB_HOST=localhost
DB_PORT=5432
DB_NAME=AgenticAIStackDB
DB_USER=AgenticAIStackDB
DB_PASSWORD=your_password

SENDGRID_API_KEY=your_sendgrid_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=+1234567890

USER_EMAIL=user@example.com
