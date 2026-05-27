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
) -> bool:
    """Send an email via Gmail API using a service account."""
    try:
        creds = service_account.Credentials.from_service_account_file(
            os.getenv("GMAIL_SERVICE_ACCOUNT_FILE", "gmail_credentials.json"),
            scopes=["https://www.googleapis.com/auth/gmail.send"],
        ).with_subject(os.getenv("GMAIL_SENDER_EMAIL", ""))

        service = discovery.build("gmail", "v1", credentials=creds)

        message          = MIMEText(body)
        message["to"]    = to
        message["subject"] = subject
        raw              = base64.urlsafe_b64encode(
            message.as_bytes()
        ).decode()

        service.users().messages().send(
            userId="me",
            body={"raw": raw},
        ).execute()

        log.info(f"email sent to {to}")
        return True

    except Exception as e:
        log.error(f"Gmail send failed: {e}")
        return False