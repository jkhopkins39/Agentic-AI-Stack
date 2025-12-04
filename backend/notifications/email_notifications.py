import os

# Try to import resend, but don't fail if not installed
try:
    import resend
    resend.api_key = os.environ.get("RESEND_API_KEY", "")
    RESEND_AVAILABLE = True
except ImportError:
    print("⚠️ Resend package not installed - email sending disabled")
    resend = None
    RESEND_AVAILABLE = False

"""Change default system email"""
DEFAULT_SYSTEM_EMAIL = "agenticstack@commerceconductor.com"
FROM_EMAIL = "Commerce Conductor <agenticstack@commerceconductor.com>"


def format_order_receipt(order_data: dict) -> str:
    """Format order data into a readable receipt"""
    if not order_data:
        return "No order data available."
    
    receipt = f"""
ORDER RECEIPT
=============

Order Number: {order_data['order_number']}
Customer: {order_data['first_name']} {order_data['last_name']}
Email: {order_data['email']}
Order Date: {order_data['created_at'].strftime('%Y-%m-%d %H:%M:%S') if order_data['created_at'] else 'N/A'}
Status: {order_data['status'].title()}

ITEMS:
------"""
    
    total = 0
    if order_data.get('items'):
        for item in order_data['items']:
            item_total = float(item['total_price'])
            total += item_total
            receipt += f"""
• {item['product_name']}
  Quantity: {item['quantity']}
  Unit Price: ${item['unit_price']:.2f}
  Total: ${item_total:.2f}"""
    
    receipt += f"""

------
TOTAL: ${order_data['total_amount']:.2f} {order_data['currency']}
======
"""
    
    return receipt


async def send_information_change_email(changes_made: list, recipient_email: str = None):
    """Send information change notification email using Resend API"""
    # Use default system email if recipient not provided
    if not recipient_email:
        recipient_email = DEFAULT_SYSTEM_EMAIL
    
    if not recipient_email:
        print("⚠️ No recipient email provided for information change notification")
        return False
    
    # Check if Resend is available and API key is configured
    if not RESEND_AVAILABLE or not resend or not resend.api_key:
        print("⚠️ Resend not available or RESEND_API_KEY not configured - email sending disabled")
        return False
    
    # Format the changes
    changes_text = ", ".join(changes_made)
    
    # Create HTML content
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                Account Information Updated
            </h2>
            
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0;">
                <p style="margin: 0; font-size: 16px;">
                    <strong>Your {changes_text} has been successfully updated.</strong>
                </p>
            </div>
            
            <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <p style="margin: 0; color: #856404;">
                    <strong>Security Notice:</strong> If you did not make this change, contact us at agenticstack@commerceconductor.com.
                </p>
            </div>
            
            <div style="margin: 30px 0;">
                <p><strong>Need Help?</strong></p>
                <p>If this change was not made by you, please contact our support team immediately:</p>
                <p style="background-color: #e3f2fd; padding: 10px; border-radius: 3px;">
                    <a href="mailto:agenticstack@commerceconductor.com" style="color: #1976d2;">agenticstack@commerceconductor.com</a>
                </p>
            </div>
            
            <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666;">
                <p>This is an automated message from Commerce Conductor.</p>
                <p>Please do not reply to this email.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    try:
        params: resend.Emails.SendParams = {
            "from": FROM_EMAIL,
            "to": [recipient_email],
            "subject": "Account Information Updated - Commerce Conductor",
            "html": html_content,
        }
        
        email = resend.Emails.send(params)
        print(f"✅ Information change email sent to {recipient_email}: {email}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send information change email: {e}")
        return False


async def send_order_receipt_email(order_data: dict, recipient_email: str = None):
    """Send order receipt email using Resend API"""
    # Use default system email if recipient not provided
    if not recipient_email:
        recipient_email = DEFAULT_SYSTEM_EMAIL
    
    if not recipient_email:
        print("⚠️ No recipient email provided for order receipt")
        print("Order receipt content:", format_order_receipt(order_data))
        return False
    
    # Check if Resend is available and API key is configured
    if not RESEND_AVAILABLE or not resend or not resend.api_key:
        print("⚠️ Resend not available or RESEND_API_KEY not configured - email sending disabled")
        return False
    
    # Format the order receipt
    receipt_content = format_order_receipt(order_data)
    
    # Create HTML content
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                🛒 Order Receipt - Commerce Conductor
            </h2>
            <pre style="font-family: 'Courier New', monospace; background-color: #f8f9fa; padding: 20px; border-radius: 5px; white-space: pre-wrap; font-size: 14px;">{receipt_content}</pre>
            
            <div style="margin-top: 30px; padding: 20px; background-color: #e8f5e9; border-radius: 5px;">
                <p style="margin: 0; color: #2e7d32;">
                    <strong>Thank you for your order!</strong>
                </p>
                <p style="margin: 10px 0 0 0; color: #555;">
                    If you have any questions about your order, please contact our customer support.
                </p>
            </div>
            
            <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666;">
                <p>This is an automated message from Commerce Conductor.</p>
                <p>Contact us: <a href="mailto:agenticstack@commerceconductor.com">agenticstack@commerceconductor.com</a></p>
            </div>
        </div>
    </body>
    </html>
    """
    
    try:
        params: resend.Emails.SendParams = {
            "from": FROM_EMAIL,
            "to": [recipient_email],
            "subject": f"Order Receipt #{order_data.get('order_number', 'N/A')} - Commerce Conductor",
            "html": html_content,
        }
        
        email = resend.Emails.send(params)
        print(f"✅ Order receipt email sent to {recipient_email}: {email}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send order receipt email: {e}")
        return False


async def send_generic_email(recipient_email: str, subject: str, html_content: str):
    """Send a generic email using Resend API"""
    if not recipient_email:
        print("⚠️ No recipient email provided")
        return False
    
    # Check if Resend is available and API key is configured
    if not RESEND_AVAILABLE or not resend or not resend.api_key:
        print("⚠️ Resend not available or RESEND_API_KEY not configured - email sending disabled")
        return False
    
    try:
        params: resend.Emails.SendParams = {
            "from": FROM_EMAIL,
            "to": [recipient_email],
            "subject": subject,
            "html": html_content,
        }
        
        email = resend.Emails.send(params)
        print(f"✅ Email sent to {recipient_email}: {email}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False
