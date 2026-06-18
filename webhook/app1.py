import sys
import os
from flask import Flask, request, jsonify
import requests

# ================= PATH =================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.append(PROJECT_ROOT)

from chatbot.main_ai import ai_chat

# ================= CONFIG =================
VERIFY_TOKEN = "deepika_verify_token"
WHATSAPP_TOKEN = "YOUR_ACCESS_TOKEN"
PHONE_NUMBER_ID = "YOUR_PHONE_NUMBER_ID"

app = Flask(__name__)


# ✅ HOME ROUTE (FIX NOT FOUND ERROR)
@app.route("/")
def home():
    return "✅ WhatsApp AI Healthcare Webhook Running"


# ================= SEND MESSAGE =================
def send_whatsapp_message(phone, message):

    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "text",
        "text": {"body": message}
    }

    requests.post(url, headers=headers, json=payload)


# ================= VERIFY =================
@app.route("/webhook", methods=["GET"])
def verify():

    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if token == VERIFY_TOKEN:
        return challenge

    return "Verification failed", 403


# ================= RECEIVE MESSAGE =================
@app.route("/webhook", methods=["POST"])
def receive():

    data = request.get_json()
    print("Incoming:", data)

    try:
        msg = data["entry"][0]["changes"][0]["value"]["messages"][0]

        phone = msg["from"]
        text = msg["text"]["body"]

        reply = ai_chat(text)

        send_whatsapp_message(phone, reply)

    except Exception as e:
        print("Error:", e)

    return jsonify({"status": "ok"})


# ================= RUN =================
if __name__ == "__main__":
    app.run(port=5000, debug=True)