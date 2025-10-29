import os
from twilio.rest import Client

"""Todo: Implement SMS notification w/ twilio"""
SMS_AVAILABLE = False

#Format order data 4 SMS
def format_order_receipt_sms(order_data: dict) -> str:
    if not order_data:
        return "No order data available."
    
    # SMS messages should be concise
    items_count = len(order_data.get('items', []))
    message = f"Order Receipt\n"
    message += f"Order: {order_data['order_number']}\n"
    message += f"Date: {order_data['created_at'].strftime('%m/%d/%Y') if order_data['created_at'] else 'N/A'}\n"
    message += f"Items: {items_count}\n"
    message += f"Total: ${order_data['total_amount']:.2f}\n"
    message += f"Status: {order_data['status'].title()}\n"
    message += f"Questions? Contact support at agenticaistack@gmail.com"
    return message


def send_sms_notification(phone_number: str, message_content: str) -> bool:
    """Todo: Send SMS notification using Twilio"""
    if not SMS_AVAILABLE:
        print(f"SMS not available. Message would have been sent to: {phone_number}")
        print(f"Message content: {message_content}")
        return False
    
    # Twilio configuration
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')
    
    if not all([account_sid, auth_token, twilio_phone_number]):
        print("Twilio credentials not found in environment variables")
        print(f"SMS would have been sent to: {phone_number}")
        print(f"Message content: {message_content}")
        return False
    
    #creates client with credentials, pass parameters to create message
    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=message_content,
            from_=twilio_phone_number,
            to=phone_number
        )
        print(f"SMS sent successfully to {phone_number}. SID: {message.sid}")
        return True
    except Exception as e:
        print(f"Error sending SMS: {e}")
        print(f"SMS would have been sent to: {phone_number}")
        print(f"Message content: {message_content}")
        return False

