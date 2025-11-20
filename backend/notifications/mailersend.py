"""
MailerSend API integration using official Python SDK
"""
import os
import asyncio
from typing import Optional
from dotenv import load_dotenv
from mailersend import emails

# Try to load .env file (useful for local development)
# In Docker, env vars should be passed directly
load_dotenv()

# Debug: Check if env var is loaded
MAILERSEND_API_KEY = os.getenv('MAILERSEND_API_KEY')
print(f"🔑 MailerSend: API Key check at module load", flush=True)
print(f"   MAILERSEND_API_KEY present: {'Yes' if MAILERSEND_API_KEY else 'No'}", flush=True)
if MAILERSEND_API_KEY:
    print(f"   MAILERSEND_API_KEY length: {len(MAILERSEND_API_KEY)}", flush=True)
    print(f"   MAILERSEND_API_KEY starts with: {MAILERSEND_API_KEY[:10]}...", flush=True)
else:
    print(f"   ⚠️ MAILERSEND_API_KEY not found in environment", flush=True)
    print(f"   All env vars with 'MAIL' or 'SEND': {[k for k in os.environ.keys() if 'MAIL' in k.upper() or 'SEND' in k.upper()]}", flush=True)

# From email must be from a verified domain in MailerSend
# Verify your domain at: https://app.mailersend.com/domains
DEFAULT_FROM_EMAIL = os.getenv('MAILERSEND_FROM_EMAIL', 'agenticstack@commerceconductor.com')
DEFAULT_FROM_NAME = os.getenv('MAILERSEND_FROM_NAME', 'Agentic AI Stack')


def _send_email_sync(
    to_email: str,
    subject: str,
    html_content: str,
    text_content: Optional[str] = None,
    from_email: Optional[str] = None,
    from_name: Optional[str] = None
) -> bool:
    """
    Synchronous email sending function (runs in executor)
    """
    try:
        # Initialize MailerSend client
        mailer = emails.NewEmail(MAILERSEND_API_KEY)
        
        # Use text content if provided, otherwise extract from HTML
        if not text_content:
            import re
            text_content = re.sub(r'<[^>]+>', '', html_content)
            text_content = text_content.strip()
        
        # Prepare email data
        mail_body = {}
        
        # Set from email
        mail_from = {
            "name": from_name or DEFAULT_FROM_NAME,
            "email": from_email or DEFAULT_FROM_EMAIL
        }
        mailer.set_mail_from(mail_from, mail_body)
        
        # Set recipient
        recipients = [{
            "name": to_email.split('@')[0],  # Use email prefix as name
            "email": to_email
        }]
        mailer.set_mail_to(recipients, mail_body)
        
        # Set subject and content
        mailer.set_subject(subject, mail_body)
        mailer.set_html_content(html_content, mail_body)
        mailer.set_plaintext_content(text_content, mail_body)
        
        # Send email
        response = mailer.send(mail_body)
        
        if response:
            print(f"✓ Email sent successfully via MailerSend to {to_email}", flush=True)
            if hasattr(response, 'message_id'):
                print(f"   Message ID: {response.message_id}", flush=True)
            return True
        else:
            print(f"⚠️ MailerSend: No response received", flush=True)
            return False
            
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ Error sending email via MailerSend: {error_msg}", flush=True)
        print(f"   Error type: {type(e).__name__}", flush=True)
        
        # Provide helpful guidance for common errors
        if 'Trial accounts' in error_msg or 'administrator' in error_msg.lower():
            print(f"   💡 TRIAL ACCOUNT RESTRICTION:", flush=True)
            print(f"      MailerSend trial accounts can ONLY send emails to the administrator's email", flush=True)
            print(f"      (the email you used to sign up for MailerSend)", flush=True)
            print(f"      To send to any email address, you need to:", flush=True)
            print(f"      1. Verify a domain in MailerSend: https://app.mailersend.com/domains", flush=True)
            print(f"      2. Set MAILERSEND_FROM_EMAIL to an email from that verified domain", flush=True)
            print(f"      OR upgrade from trial account to a paid plan", flush=True)
        
        if 'verified' in error_msg.lower() or 'domain' in error_msg.lower():
            domain = (from_email or DEFAULT_FROM_EMAIL).split('@')[1] if '@' in (from_email or DEFAULT_FROM_EMAIL) else 'unknown'
            print(f"   💡 DOMAIN VERIFICATION REQUIRED:", flush=True)
            print(f"      The domain '{domain}' must be verified in your MailerSend account", flush=True)
            print(f"      Steps to fix:", flush=True)
            print(f"      1. Go to: https://app.mailersend.com/domains", flush=True)
            print(f"      2. Add and verify the domain '{domain}'", flush=True)
            print(f"      3. Set MAILERSEND_FROM_EMAIL in .env to an email from that domain", flush=True)
            print(f"         Example: MAILERSEND_FROM_EMAIL=agenticstack@{domain}", flush=True)
        
        import traceback
        traceback.print_exc()
        return False


async def send_email_via_mailersend(
    to_email: str,
    subject: str,
    html_content: str,
    text_content: Optional[str] = None,
    from_email: Optional[str] = None,
    from_name: Optional[str] = None
) -> bool:
    """
    Send email using MailerSend official Python SDK (async wrapper)
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        html_content: HTML email body
        text_content: Plain text email body (optional, will use HTML if not provided)
        from_email: Sender email (defaults to DEFAULT_FROM_EMAIL)
        from_name: Sender name (defaults to DEFAULT_FROM_NAME)
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    # Check if API key is configured
    if not MAILERSEND_API_KEY:
        print("⚠️ MAILERSEND_API_KEY not configured. Cannot send email.", flush=True)
        print(f"   Email would have been sent to: {to_email}", flush=True)
        print(f"   Subject: {subject}", flush=True)
        print(f"   Check your .env file or environment variables for MAILERSEND_API_KEY", flush=True)
        return False
    
    # Check if API key looks valid (should be a string, not empty)
    if not isinstance(MAILERSEND_API_KEY, str) or len(MAILERSEND_API_KEY.strip()) == 0:
        print("⚠️ MAILERSEND_API_KEY is empty or invalid. Cannot send email.", flush=True)
        return False
    
    if not to_email:
        print("⚠️ No recipient email provided", flush=True)
        return False
    
    from_email = from_email or DEFAULT_FROM_EMAIL
    from_name = from_name or DEFAULT_FROM_NAME
    
    print(f"📧 MailerSend: Preparing to send email", flush=True)
    print(f"   From: {from_email} ({from_name})", flush=True)
    print(f"   To: {to_email}", flush=True)
    print(f"   Subject: {subject}", flush=True)
    print(f"   API Key present: {'Yes' if MAILERSEND_API_KEY else 'No'} (length: {len(MAILERSEND_API_KEY) if MAILERSEND_API_KEY else 0})", flush=True)
    
    # Run the synchronous SDK call in an executor to keep it non-blocking
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _send_email_sync,
            to_email,
            subject,
            html_content,
            text_content,
            from_email,
            from_name
        )
        return result
    except Exception as e:
        print(f"⚠️ Error in email executor: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False
