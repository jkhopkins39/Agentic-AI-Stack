# Production Readiness Improvements

## Overview
This document details the security, validation, and monitoring improvements implemented to make the LangGraph agent system production-ready. All changes focus on robustness, security, and evaluation metrics.

## Changes Implemented

### 1. Unclassified Query Logging for Evaluation Metrics

Purpose: Track queries that the classifier struggles with to improve the model and provide evaluation metrics.

Implementation:
- Created log_unclassified_query() function that logs queries classified as "Message" (general/unclear)
- Logs include timestamp, session_id, conversation_id, query text, and attempted classification
- Logs stored in logs/unclassified_queries.json for analysis

Usage:
```python
# Automatically called when a query is classified as "Message"
log_unclassified_query(
    query="What's the weather like?",
    session_id="abc123",
    conversation_id="xyz789",
    attempted_classification="Message"
)
```

Log Format:
```json
{
  "timestamp": "2025-10-28T14:30:00",
  "session_id": "abc123",
  "conversation_id": "xyz789",
  "query": "What's the weather like?",
  "attempted_classification": "Message"
}
```

---

### 2. Enhanced RAG Query Validation
#### Low Relevance Score (< 0.7):
```python
# provides answer but acknowledges uncertainty
response = (
    "I found some potentially relevant information, but I'm not entirely confident "
    "it directly addresses your question. Here's what I found:\n\n"
    + generated_response +
    "\n\nIf this doesn't fully answer your question, please contact support@agenticaistack.com"
)
```

### 3. Email Validation & Hardcoded Email Removal
#### New Validation Function:
```python
def validate_email(email: str) -> tuple[bool, str]:
    """
    Validate email address format
    Returns: (is_valid, error_message)
    """
    - Checks if email exists
    - Validates length (max 320 chars per RFC 5321)
    - Validates format using regex
    - Returns clear error messages
```
#### Email Regex Pattern:
```python
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
```

### 4. Kafka Retry Logic with Exp Backoff
#### New Function:
```python
def publish_to_kafka_with_retry(
    producer,
    topic: str,
    message: dict,
    max_retries: int = 3
) -> bool
```

#### Retry Strategy:
1. Exponential Backoff: 1s → 2s → 4s → 8s
2. Max Retries: Configurable (default: 3)
3. Retry Tracking: Counts stored in message metadata
4. Failure Logging: Failed messages logged for manual retry

#### Example Usage:
```python
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

message = {
    "event": "order_created",
    "order_id": "ORD-123",
    "timestamp": datetime.now().isoformat()
}

success = publish_to_kafka_with_retry(producer, 'orders', message)

if success:
    print("Message published successfully")
else:
    print("Message failed after all retries - check logs/failed_kafka_messages.json")
```

#### Failed Message Logging:
```json
{
  "timestamp": "2025-10-28T14:30:00",
  "message": {
    "event": "order_created",
    "order_id": "ORD-123",
    "retry_count": 3
  },
  "error": "Connection timeout"
}
```

### 5. Input Sanitization & SQL Injection Prevention
#### New Func:
```python
def sanitize_input(input_string: str, max_length: int = 500) -> tuple[str, bool]:
    """
    Sanitize user input to prevent SQL injection and XSS attacks
    Returns: (sanitized_string, is_potentially_malicious)
    """
```

#### Security Blacklist Patterns:
```python
SQL_INJECTION_PATTERNS = [
    r"(\bor\b|\band\b).*[=<>]",  # OR/AND with comparisons
    r"(union|select|insert|update|delete|drop|create|alter|exec|execute)\s",
    r"[;'\"\\]",  # Common SQL injection characters
    r"--",  # SQL comments
    r"/\*.*\*/",  # SQL block comments
    r"xp_cmdshell",  # Command execution
    r"script>",  # XSS attempts
]
```

### New Log Files
```
logs/
  ├── unclassified_queries.json    # Queries for evaluation metrics
  └── failed_kafka_messages.json   # Failed Kafka publishes for retry
```


## Testing

### Test Email Validation:
```python
# Valid emails
validate_email("user@example.com")  # (True, "")
validate_email("john.doe+tag@company.co.uk")  # (True, "")

# Invalid emails
validate_email("not-an-email")  # (False, "Invalid email format...")
validate_email("@example.com")  # (False, "Invalid email format...")
validate_email("user@")  # (False, "Invalid email format...")
```

### Test Input Sanitization:
```python
# Clean input
sanitize_input("laptop")  # ("laptop", False)

# Malicious input
sanitize_input("laptop' OR 1=1--")  # ("laptop OR 11", True)
sanitize_input("<script>alert('XSS')</script>")  # ("scriptalertXSSscript", False)
```

### Test RAG Query:
```python
# Test with irrelevant query
query_rag("What's the weather today?")
# Returns helpful message with suggestions

# Test with partially relevant query (score 0.6)
query_rag("Can I return something?")
# Returns answer with confidence disclaimer
```

### Test Kafka Retry:
```python
# Simulate failure
def failing_producer_send(topic, value):
    raise Exception("Connection timeout")

# Will retry 3 times with exponential backoff
# Then log to failed_kafka_messages.json
```

---



### 1. Update Kafka Publishers:
```python
# Old way
producer.send('topic', message)

# New way with retry
publish_to_kafka_with_retry(producer, 'topic', message, max_retries=3)
```

### 4. Review Unclassified Queries:
```bash
# Periodically review logs
cat logs/unclassified_queries.json | jq '.[] | .query'
```
