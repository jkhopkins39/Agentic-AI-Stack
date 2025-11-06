from .email_notifications import send_information_change_email, send_order_receipt_email, format_order_receipt
from .sms_notifications import send_sms_notification, format_order_receipt_sms
from .preferences import get_user_notification_preference, set_user_notification_preference
from .multi_channel import send_notification

#Export all functions so imports are easily accessed elsewhere
__all__ = [
    'send_information_change_email',
    'send_order_receipt_email',
    'format_order_receipt',
    'send_sms_notification',
    'format_order_receipt_sms',
    'get_user_notification_preference',
    'set_user_notification_preference',
    'send_notification'
]

