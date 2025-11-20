import os
from .email_notifications import send_order_receipt_email, send_information_change_email
from .sms_notifications import send_sms_notification, format_order_receipt_sms
from .preferences import get_user_notification_preference

# Session email from environment or default. Change system email
DEFAULT_SYSTEM_EMAIL = "agenticstack@commerceconductor.com"
SESSION_EMAIL = os.getenv('USER_EMAIL', DEFAULT_SYSTEM_EMAIL)

"""Checks preferences and sends notification via email or SMS. WIP"""
def send_notification(notification_type: str, content: dict, user_data: dict, delivery_method: str = None) -> bool:
    if not user_data:
        print("No user data provided for notification")
        return False
    
    """Get user preference if delivery method not specified. Will update once SMS is implemented."""
    if not delivery_method and user_data.get('id'):
        preference = get_user_notification_preference(user_data['id'])
        if preference:
            delivery_method = preference.get('preferred_method', 'email')
        else:
            delivery_method = 'email'  # Default to email
    elif not delivery_method:
        delivery_method = 'email'
    
    email_sent = False
    sms_sent = False
    
    recipient_email = user_data.get('email', SESSION_EMAIL)
    recipient_phone = user_data.get('phone')
    
    # Send email if requested
    if delivery_method in ['email', 'both']:
        if notification_type == 'order_receipt':
            email_sent = send_order_receipt_email(content, recipient_email)
        elif notification_type == 'info_change':
            email_sent = send_information_change_email(content.get('changes_made', []), recipient_email)
    
    # Send SMS if requested and phone number available
    if delivery_method in ['sms', 'both'] and recipient_phone:
        if notification_type == 'order_receipt':
            sms_content = format_order_receipt_sms(content)
            sms_sent = send_sms_notification(recipient_phone, sms_content)
        elif notification_type == 'info_change':
            changes_text = ", ".join(content.get('changes_made', []))
            sms_content = f"Account Update: Your {changes_text} has been changed. If you didn't make this change, contact agenticaistack@gmail.com"
            sms_sent = send_sms_notification(recipient_phone, sms_content)
    
    # Return True if at least one method succeeded
    if delivery_method == 'both':
        return email_sent or sms_sent
    elif delivery_method == 'email':
        return email_sent
    elif delivery_method == 'sms':
        return sms_sent
    
    return False

