import os
from notifications.mailersend import send_email_via_mailersend

"""Change default system email"""
DEFAULT_SYSTEM_EMAIL = "agenticstack@commerceconductor.com"


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
    """Send information change notification email using MailerSend API"""
    # Use default system email if recipient not provided
    if not recipient_email:
        recipient_email = DEFAULT_SYSTEM_EMAIL
    
    if not recipient_email:
        print("⚠️ No recipient email provided for information change notification")
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
                    <strong>Security Notice:</strong> If you did not make this change, contact us at agenticaistack@gmail.com.
                </p>
            </div>
            
            <div style="margin: 30px 0;">
                <p><strong>Need Help?</strong></p>
                <p>If this change was not made by you, please contact our support team immediately:</p>
                <p style="background-color: #e3f2fd; padding: 10px; border-radius: 3px;">
                    <a href="mailto:agenticaistack@gmail.com" style="color: #1976d2;">agenticaistack@gmail.com</a>
                </p>
            </div>
            
            <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666;">
                <p>This is an automated message from your Agentic AI Stack system.</p>
                <p>Please do not reply to this email.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Create plain text version
    text_content = f"""
Account Information Updated

Your {changes_text} has been successfully updated.

SECURITY NOTICE: If you did not make this change, please contact us immediately.

Need Help?
If this change was not made by you, please contact our support team:
Email: agenticaistack@gmail.com

This is an automated message from your Agentic AI Stack system.
Please do not reply to this email.
    """
    
    # Send via MailerSend API
    return await send_email_via_mailersend(
        to_email=recipient_email,
        subject="Account Information Changed - Agentic AI Stack",
        html_content=html_content,
        text_content=text_content
    )


async def send_order_receipt_email(order_data: dict, recipient_email: str = None):
    """Send order receipt email using MailerSend API"""
    # Use default system email if recipient not provided
    if not recipient_email:
        recipient_email = DEFAULT_SYSTEM_EMAIL
    
    if not recipient_email:
        print("⚠️ No recipient email provided for order receipt")
        print("Order receipt content:", format_order_receipt(order_data))
        return False
    
    # Format the order receipt
    receipt_content = format_order_receipt(order_data)
    
    # Create HTML content
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                Order Receipt
            </h2>
            <pre style="font-family: 'Courier New', monospace; background-color: #f8f9fa; padding: 20px; border-radius: 5px; white-space: pre-wrap; font-size: 14px;">{receipt_content}</pre>
            <br><br>
            <p>If you have any questions, please contact our customer support.</p>
        </div>
    </body>
    </html>
    """
    
    # Create plain text version
    text_content = f"Order Receipt\n\n{receipt_content}\n\nThank you for your business!\nIf you have any questions, please contact our customer support."
    
    # Send via MailerSend API
    return await send_email_via_mailersend(
        to_email=recipient_email,
        subject=f"Order Receipt - {order_data['order_number']}",
        html_content=html_content,
        text_content=text_content
    )

