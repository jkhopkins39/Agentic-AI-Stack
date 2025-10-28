"""Logging functions for metrics and monitoring."""
import os
import json
from datetime import datetime

# Logging configuration
UNCLASSIFIED_QUERIES_LOG = "logs/unclassified_queries.json"
FAILED_KAFKA_LOG = "logs/failed_kafka_messages.json"

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)


def log_unclassified_query(query: str, session_id: str, conversation_id: str, attempted_classification: str = None):
    """Log unclassified queries for evaluation metrics and model improvement"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "conversation_id": conversation_id,
        "query": query,
        "attempted_classification": attempted_classification,
    }
    
    try:
        # Load existing logs
        if os.path.exists(UNCLASSIFIED_QUERIES_LOG):
            with open(UNCLASSIFIED_QUERIES_LOG, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Append new log
        logs.append(log_entry)
        
        # Save back to file
        with open(UNCLASSIFIED_QUERIES_LOG, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"[METRICS] Logged unclassified query for evaluation")
    except Exception as e:
        print(f"Error logging unclassified query: {e}")


def log_failed_kafka_message(message: dict, error: str):
    """Log failed Kafka messages for retry and monitoring"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "message": message,
        "error": str(error),
        "retry_count": message.get("retry_count", 0)
    }
    
    try:
        if os.path.exists(FAILED_KAFKA_LOG):
            with open(FAILED_KAFKA_LOG, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        with open(FAILED_KAFKA_LOG, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"[KAFKA] Logged failed message for retry")
    except Exception as e:
        print(f"Error logging failed Kafka message: {e}")

