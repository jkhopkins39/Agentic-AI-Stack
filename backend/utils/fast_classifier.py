"""Fast rule-based classifier for common patterns.
The aim here is to reduce the amount of classification done
by the LLM, shifting the burden to classical scripting in order
to reduce token usage (monetary cost) and latency (time cost)"""
import re
from typing import Optional, Literal

MessageType = Literal["Order", "Email", "Policy", "Message", "Change Information", "Order Receipt"]


def fast_classify(query_text: str) -> Optional[MessageType]:
    """
    Fast rule-based classifier for common patterns.
    Returns None if classification is ambiguous and needs LLM.
    """
    query_lower = query_text.lower()
    
    # Policy keywords (high confidence)
    policy_keywords = [
        r'\breturn\s+policy\b',
        r'\breturn\s+policy\?',
        r'\breturn\s+timeframe',
        r'\breturn\s+process',
        r'\breturn\s+window',
        r'\bhow\s+long.*return',
        r'\bcan\s+i\s+return',
        r'\bdo\s+you\s+accept\s+returns',
        r'\brefund\s+policy',
        r'\bexchange\s+policy',
        r'\bwarranty\s+policy',
        r'\bshipping\s+policy',
        r'\bdelivery\s+policy',
        r'\bcompany\s+policy',
        r'\bstore\s+policy',
        r'\bterms\s+of\s+service',
        r'\breturn\s+procedure',
        r'\bhow\s+do\s+returns\s+work',
    ]
    
    # Order keywords (high confidence)
    order_keywords = [
        r'\border\s+status',
        r'\border\s+number',
        r'\btracking\s+number',
        r'\bshipping\s+status',
        r'\bwhere\s+is\s+my\s+order',
        r'\bwhere\s+is\s+my\s+package',
        r'\border\s+history',
        r'\border\s+receipt',
        r'\bmy\s+order',
        r'\bmy\s+orders',
        r'\border\s+tracking',
        r'\btrack\s+order',
    ]
    
    # Email keywords
    email_keywords = [
        r'\bsend\s+.*\s+email',
        r'\bemail\s+me',
        r'\bvia\s+email',
        r'\bemail\s+it',
        r'\bsend\s+email',
    ]
    
    # Change Information keywords
    change_info_keywords = [
        r'\bchange\s+my\s+(name|email|phone|address)',
        r'\bupdate\s+my\s+(name|email|phone|address)',
        r'\bmodify\s+my\s+(name|email|phone|address)',
        r'\bchange\s+(name|email|phone|address)',
        r'\bupdate\s+(name|email|phone|address)',
    ]
    
    # Order Receipt keywords
    receipt_keywords = [
        r'\breceipt\s+for\s+order',
        r'\border\s+receipt',
        r'\bget\s+receipt',
        r'\bshow\s+receipt',
        r'\bdownload\s+receipt',
        r'\bmy\s+receipt',
    ]
    
    # Check policy keywords (high confidence)
    for pattern in policy_keywords:
        if re.search(pattern, query_lower):
            return "Policy"
    
    # Check receipt keywords (before general order keywords)
    for pattern in receipt_keywords:
        if re.search(pattern, query_lower):
            return "Order Receipt"
    
    # Check order keywords
    for pattern in order_keywords:
        if re.search(pattern, query_lower):
            return "Order"
    
    # Check email keywords
    for pattern in email_keywords:
        if re.search(pattern, query_lower):
            return "Email"
    
    # Check change information keywords
    for pattern in change_info_keywords:
        if re.search(pattern, query_lower):
            return "Change Information"
    
    # If no clear pattern matches, return None to use LLM
    return None

