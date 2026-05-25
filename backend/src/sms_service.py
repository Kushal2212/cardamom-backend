from twilio.rest import Client
import os

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)


def send_sms(to, message):
    try:
        msg = client.messages.create(
            body=message,
            from_=TWILIO_PHONE,
            to=to
        )
        return {"status": "sent", "sid": msg.sid}
    except Exception as e:
        return {"status": "error", "error": str(e)}