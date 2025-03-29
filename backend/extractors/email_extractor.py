import email
from bs4 import BeautifulSoup
import tempfile
import os

class EmailExtractor:
    def extract_text_from_email(self, email_bytes):
        """Extract text from EML files"""
        try:
            # Parse email message
            msg = email.message_from_bytes(email_bytes)
            
            # Extract email metadata
            metadata = [
                f"From: {msg['from']}",
                f"To: {msg['to']}",
                f"Subject: {msg['subject']}",
                f"Date: {msg['date']}",
                "\nContent:\n"
            ]
            
            # Extract email body
            body = []
            
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body.append(part.get_payload(decode=True).decode())
                    elif part.get_content_type() == "text/html":
                        # Convert HTML to plain text
                        html = part.get_payload(decode=True).decode()
                        soup = BeautifulSoup(html, 'html.parser')
                        body.append(soup.get_text())
            else:
                content = msg.get_payload(decode=True).decode()
                if msg.get_content_type() == "text/html":
                    soup = BeautifulSoup(content, 'html.parser')
                    body.append(soup.get_text())
                else:
                    body.append(content)
            
            # Combine metadata and body
            return '\n'.join(metadata + body)
            
        except Exception as e:
            raise Exception(f"Error extracting text from email: {str(e)}")
