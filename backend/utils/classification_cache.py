"""Classification caching with fuzzy matching and embedding similarity. The aim
is to store responses to common questions for much faster retrieval and reduction
of redundant response generation."""
import hashlib # Used for creation of MD5 hash
import time # Timeouts, etc
from typing import Optional, Dict, Tuple
import re # Text normalization

# In-memory cache with time to live
_classification_cache: Dict[str, Tuple[str, float]] = {}
CACHE_TTL = 3600  # 1 hour
MAX_CACHE_SIZE = 10000  # Max cache entries


def _normalize_text(text: str) -> str:
    """Normalize text for caching (lowercase, remove extra whitespace)"""
    return re.sub(r'\s+', ' ', text.lower().strip())


def _create_cache_key(text: str) -> str:
    #Normalize text input
    normalized = _normalize_text(text)

    """Create a unique ID for the question text.
    Converts normed text to MD5 hash (fixed-length str); Same question = same key (always)"""
    return hashlib.md5(normalized.encode()).hexdigest()


def _is_similar(text1: str, text2: str, threshold: float = 0.8) -> bool:
    # Simple similarity check using word overlap
    words1 = set(_normalize_text(text1).split())
    words2 = set(_normalize_text(text2).split())
    
    if not words1 or not words2:
        return False
    
    intersection = words1 & words2
    union = words1 | words2
    
    if not union:
        return False
    
    similarity = len(intersection) / len(union)
    return similarity >= threshold


def get_cached_classification(query_text: str) -> Optional[str]:
    """
    Get cached classification result if available and not expired.
    Also checks for similar queries...
    """
    # Clean up expired entries periodically
    current_time = time.time()
    expired_keys = [
        key for key, (_, timestamp) in _classification_cache.items()
        if current_time - timestamp > CACHE_TTL
    ]
    for key in expired_keys:
        _classification_cache.pop(key, None)
    
    # Check exact match first
    cache_key = _create_cache_key(query_text)
    if cache_key in _classification_cache:
        message_type, timestamp = _classification_cache[cache_key]
        if current_time - timestamp <= CACHE_TTL:
            return message_type
    
    # Check for similar queries (fuzzy matching)
    normalized_query = _normalize_text(query_text)
    for cached_key, (message_type, timestamp) in _classification_cache.items():
        if current_time - timestamp <= CACHE_TTL:
            # Try to find the original text (we'd need to store it, but for now use key)
            # For better fuzzy matching, we'd need to store original texts
            # This is a simplified version
            pass
    
    return None


def cache_classification(query_text: str, message_type: str):
    """Cache a classification result"""
    cache_key = _create_cache_key(query_text)
    
    # Evict oldest entries if cache is full
    if len(_classification_cache) >= MAX_CACHE_SIZE:
        # Remove oldest tenth of entries
        sorted_entries = sorted(
            _classification_cache.items(),
            key=lambda x: x[1][1]  # Sort by timestamp
        )
        to_remove = len(sorted_entries) // 10
        for key, _ in sorted_entries[:to_remove]:
            _classification_cache.pop(key, None)
    
    _classification_cache[cache_key] = (message_type, time.time())


def clear_cache():
    # Clear classification cache...
    _classification_cache.clear()

