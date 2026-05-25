from flask import Blueprint, request, jsonify
from backend.database.mongo import contacts
from datetime import datetime

contact_bp = Blueprint("contact", __name__, url_prefix="/api")


# ═══════════════════════════════════════
# SAVE CONTACT MESSAGE
# ═══════════════════════════════════════
@contact_bp.route("/contact", methods=["POST"])
def contact():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    name = data.get("name", "").strip()
    email = data.get("email", "").strip()
    subject = data.get("subject", "").strip()
    message = data.get("message", "").strip()

    if not name or not email or not message:
        return jsonify({
            "error": "Name, email and message are required"
        }), 400

    contact_doc = {
        "name": name,
        "email": email,
        "subject": subject or "No subject",
        "message": message,
        "is_read": False,
        "created_at": datetime.utcnow()
    }

    contacts.insert_one(contact_doc)

    return jsonify({
        "success": True,
        "message": "Message received successfully"
    }), 201