"""
Gmail API sending (service account + domain-wide delegation).

Temporarily unused — platform posting is disabled via GMAIL_POSTING_ENABLED
in services/platform_parser.py. Re-enable when credentials are configured.
"""

import os
import base64
from email.mime.text   import MIMEText
from google.oauth2      import service_account
from googleapiclient   import discovery
from services.logger   import get_logger

log = get_logger(__name__)


def send_email(
    to:      str,
    subject: str,
    body:    str,
) -> tuple[bool, str | None]:
    """Send an email via Gmail API using a service account. Returns (ok, error_message)."""
    creds_path = os.getenv("GMAIL_SERVICE_ACCOUNT_FILE", "gmail_credentials.json")
    sender     = os.getenv("GMAIL_SENDER_EMAIL", "").strip()

    if not sender:
        return False, "Gmail not configured — set GMAIL_SENDER_EMAIL in .env"
    if not os.path.isfile(creds_path):
        return False, (
            f"Gmail credentials file not found: {creds_path}. "
            "Download a Google service account JSON key and set GMAIL_SERVICE_ACCOUNT_FILE."
        )

    try:
        creds = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/gmail.send"],
        ).with_subject(sender)

        service = discovery.build("gmail", "v1", credentials=creds)

        message            = MIMEText(body)
        message["to"]      = to
        message["subject"] = subject
        raw                = base64.urlsafe_b64encode(message.as_bytes()).decode()

        service.users().messages().send(
            userId="me",
            body={"raw": raw},
        ).execute()

        log.info(f"email sent to {to}")
        return True, None

    except Exception as e:
        log.error(f"Gmail send failed: {e}")
        return False, f"Gmail send failed: {e}"