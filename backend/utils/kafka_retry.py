import time
from .logging import log_failed_kafka_message

#Kafka retry logic with exp backoff
def publish_to_kafka_with_retry(producer, topic: str, message: dict, max_retries: int = 3) -> bool:
    retry_count = message.get("retry_count", 0)
    base_delay = 1 
    #Loop through max retries
    for attempt in range(max_retries):
        try:
            future = producer.send(topic, value=message)
            future.get(timeout=10)
            if retry_count > 0:
                print(f"[KAFKA] Message successfully published after {retry_count} previous failures")
            
            return True
            
        except Exception as e:
            retry_count += 1
            message["retry_count"] = retry_count
            
            if attempt < max_retries - 1:
                # Calculate delay: 1s, 2s, 4s, so om
                delay = base_delay * (2 ** attempt)
                print(f"[KAFKA] Publish failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"[KAFKA] Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"[KAFKA] Failed to publish after {max_retries} attempts: {e}")
                log_failed_kafka_message(message, str(e))
                return False
    
    return False

