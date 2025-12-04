# Try to import email notifications, but don't fail if resend is not installed
try:
    from .email_notifications import send_information_change_email, send_order_receipt_email, format_order_receipt
    EMAIL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Email notifications not available: {e}")
    EMAIL_AVAILABLE = False
    send_information_change_email = None
    send_order_receipt_email = None
    format_order_receipt = None

# Try to import SMS notifications, but don't fail if twilio is not installed
try:
    from .sms_notifications import send_sms_notification, format_order_receipt_sms
    SMS_AVAILABLE = True
except ImportError:
    SMS_AVAILABLE = False
    send_sms_notification = None
    format_order_receipt_sms = None

# Try to import other notification modules
try:
    from .preferences import get_user_notification_preference, set_user_notification_preference
except ImportError:
    get_user_notification_preference = None
    set_user_notification_preference = None

try:
    from .multi_channel import send_notification
except ImportError:
    send_notification = None

#Export all functions so imports are easily accessed elsewhere
__all__ = [
    'send_information_change_email',
    'send_order_receipt_email',
    'format_order_receipt',
    'send_sms_notification',
    'format_order_receipt_sms',
    'get_user_notification_preference',
    'set_user_notification_preference',
    'send_notification',
    'SMS_AVAILABLE',
    'EMAIL_AVAILABLE'
]

