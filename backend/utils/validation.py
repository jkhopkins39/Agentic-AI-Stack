import re

# Input validation patterns
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
PHONE_E164_REGEX = re.compile(r'^\+[1-9]\d{1,14}$')

#Basic blacklist, more to come
SQL_INJECTION_PATTERNS = [
    r"(\bor\b|\band\b).*[=<>]",  # OR/AND with comparisons // lets an attacker change query logic
    r"(union|select|insert|update|delete|drop|create|alter|exec|execute)\s", #may be used to extract data or run commands
    r"[;'\"\\]",  # Common SQL injection characters that may be used to kill queries
    r"--",  # SQL comments that may be used to bypass filters
    r"/\*.*\*/",  # ^^^
    r"xp_cmdshell",  #to run commands
    r"script>",  # XSS attempts that may be used to inject scripts
]


def validate_email(email: str) -> tuple[bool, str]:
    if not email or not isinstance(email, str):
        return False, "Email address is required"
    
    email = email.strip()
    
    if len(email) > 320:  # RFC 5321 max length
        return False, "Email address is too long"
    
    if not EMAIL_REGEX.match(email):
        return False, "Invalid email format. Please use format: user@example.com"
    
    return True, ""

#Validate phone number in E.164 format
def validate_phone(phone: str) -> tuple[bool, str]:
    if not phone or not isinstance(phone, str):
        return False, "Phone number is required"
    
    phone = phone.strip()
    
    if not PHONE_E164_REGEX.match(phone):
        return False, "Invalid phone format. Please use E.164 format: +1234567890"
    
    return True, ""

#Returns: (sanitized_string, is_potentially_malicious)
def sanitize_input(input_string: str, max_length: int = 500) -> tuple[str, bool]:
    if not input_string:
        return "", False
    
    # Trim to max length to prevent DOS attacks
    input_string = input_string[:max_length]
    
    # Check for SQL injection patterns
    is_malicious = False
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, input_string, re.IGNORECASE):
            is_malicious = True
            print(f"[SECURITY] Potential SQL injection attempt detected: {pattern}")
            break
    
    # Remove potentially dangerous characters while preserving legitimate use
    # Allow letters, numbers, spaces, basic punctuation
    sanitized = re.sub(r'[^\w\s@.\-+,()]', '', input_string)
    
    return sanitized, is_malicious

