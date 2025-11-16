#!/usr/bin/env python3
"""
Test script for MailerSend API
Run this to verify your MailerSend configuration works
"""
import os
import sys
from dotenv import load_dotenv
from mailersend import emails

# Load environment variables
load_dotenv()

# Get configuration
API_KEY = os.getenv('MAILERSEND_API_KEY')
FROM_EMAIL = os.getenv('MAILERSEND_FROM_EMAIL', 'agenticstack@commerceconductor.com')
FROM_NAME = os.getenv('MAILERSEND_FROM_NAME', 'Agentic AI Stack')
TO_EMAIL = os.getenv('TEST_EMAIL', input("Enter recipient email address: ").strip())

if not API_KEY:
    print("❌ ERROR: MAILERSEND_API_KEY not found in environment variables")
    print("   Make sure it's set in your .env file or environment")
    sys.exit(1)

if not TO_EMAIL:
    print("❌ ERROR: No recipient email provided")
    sys.exit(1)

print("=" * 80)
print("MailerSend API Test")
print("=" * 80)
print(f"API Key: {'*' * (len(API_KEY) - 10)}{API_KEY[-10:]} (length: {len(API_KEY)})")
print(f"From: {FROM_EMAIL} ({FROM_NAME})")
print(f"To: {TO_EMAIL}")
print("=" * 80)
print()

try:
    # Initialize MailerSend client
    print("📧 Initializing MailerSend client...")
    mailer = emails.NewEmail(API_KEY)
    
    # Prepare email data
    mail_body = {}
    
    # Set from email
    print("📧 Setting from email...")
    mail_from = {
        "name": FROM_NAME,
        "email": FROM_EMAIL
    }
    mailer.set_mail_from(mail_from, mail_body)
    
    # Set recipient
    print("📧 Setting recipient...")
    recipients = [{
        "name": TO_EMAIL.split('@')[0],
        "email": TO_EMAIL
    }]
    mailer.set_mail_to(recipients, mail_body)
    
    # Set subject and content
    print("📧 Setting subject and content...")
    subject = "Test Email from MailerSend API"
    html_content = """
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                Test Email
            </h2>
            <p>This is a test email sent from the MailerSend API.</p>
            <p>If you received this email, the API is working correctly!</p>
            <p><strong>Timestamp:</strong> {}</p>
        </div>
    </body>
    </html>
    """.format(__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    text_content = "This is a test email sent from the MailerSend API.\n\nIf you received this email, the API is working correctly!"
    
    mailer.set_subject(subject, mail_body)
    mailer.set_html_content(html_content, mail_body)
    mailer.set_plaintext_content(text_content, mail_body)
    
    # Print mail_body for debugging
    print("\n📧 Email payload:")
    print(f"   {mail_body}")
    print()
    
    # Send email
    print("📧 Sending email...")
    response = mailer.send(mail_body)
    
    print()
    print("=" * 80)
    if response:
        print("✅ SUCCESS: Email sent!")
        print(f"   Response: {response}")
        if hasattr(response, 'message_id'):
            print(f"   Message ID: {response.message_id}")
        print()
        print("⚠️  Note: If you don't receive the email, check:")
        print("   1. Your spam/junk folder")
        print("   2. MailerSend Activity page: https://app.mailersend.com/activity")
        print("   3. Domain verification status: https://app.mailersend.com/domains")
        print("   4. Trial account restrictions (can only send to admin email)")
    else:
        print("❌ FAILED: No response received from MailerSend")
    print("=" * 80)
    
except Exception as e:
    print()
    print("=" * 80)
    print("❌ ERROR: Failed to send email")
    print("=" * 80)
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print()
    print("Full traceback:")
    import traceback
    traceback.print_exc()
    print("=" * 80)
    sys.exit(1)

