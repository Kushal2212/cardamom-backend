from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime
import os

from database.mongo import users, newsletter

newsletter_bp = Blueprint("newsletter", __name__, url_prefix="/api/newsletter")


# ───────────────────────────────────────
# Helpers
# ───────────────────────────────────────

def to_oid(id_str):
    try:
        from bson import ObjectId
        return ObjectId(id_str)
    except Exception:
        return None


def check_admin():
    user_id = get_jwt_identity()
    oid = to_oid(user_id)
    user = users.find_one({"_id": oid if oid else user_id})
    return user if user and user.get("is_admin") else None


# ───────────────────────────────────────
# Subscribe
# ───────────────────────────────────────

@newsletter_bp.route("/subscribe", methods=["POST"])
def subscribe():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()

    if not email or "@" not in email:
        return jsonify({"error": "Invalid email"}), 400

    existing = newsletter.find_one({"email": email})
    if existing:
        if existing.get("is_active"):
            return jsonify({"message": "Already subscribed"}), 200
        newsletter.update_one({"email": email}, {"$set": {"is_active": True}})
        return jsonify({"message": "Resubscribed successfully"}), 200

    newsletter.insert_one({
        "email": email,
        "is_active": True,
        "created_at": datetime.utcnow(),
    })
    return jsonify({"message": "Subscribed successfully"}), 201


# ───────────────────────────────────────
# Unsubscribe
# ───────────────────────────────────────

@newsletter_bp.route("/unsubscribe", methods=["POST"])
def unsubscribe():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()

    if not email:
        return jsonify({"error": "Email required"}), 400

    newsletter.update_one({"email": email}, {"$set": {"is_active": False}})
    return jsonify({"message": "Unsubscribed successfully"}), 200


# ───────────────────────────────────────
# Get Subscribers (Admin)
# ───────────────────────────────────────

@newsletter_bp.route("/subscribers", methods=["GET"])
@jwt_required()
def get_subscribers():
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 10))
    active = request.args.get("active")

    query = {}
    if active == "true":
        query["is_active"] = True
    elif active == "false":
        query["is_active"] = False

    skip = (page - 1) * limit
    total = newsletter.count_documents(query)
    active_count = newsletter.count_documents({"is_active": True})

    data = []
    for sub in newsletter.find(query).skip(skip).limit(limit):
        sub["_id"] = str(sub["_id"])
        data.append(sub)

    return jsonify({
        "subscribers": data,
        "total": total,
        "total_active": active_count,
        "page": page,
        "pages": (total + limit - 1) // limit,
    })


# ───────────────────────────────────────
# Delete Subscriber (Admin)
# ───────────────────────────────────────

@newsletter_bp.route("/subscribers/<email>", methods=["DELETE"])
@jwt_required()
def delete_subscriber(email):
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    newsletter.delete_one({"email": email})
    return jsonify({"message": "Subscriber deleted"})


# ───────────────────────────────────────
# Send Newsletter (Admin)
# ───────────────────────────────────────

@newsletter_bp.route("/send", methods=["POST"])
@jwt_required()
def send_newsletter():
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    data = request.get_json() or {}
    subject = data.get("subject", "")
    title = data.get("title", "")
    body = data.get("body", "")

    if not subject or not body:
        return jsonify({"error": "Subject and body required"}), 400

    subscribers = list(newsletter.find({"is_active": True}))
    sent = len(subscribers)

    if os.environ.get("MAIL_SERVER"):
        try:
            from flask_mail import Mail, Message
            from flask import current_app

            mail = Mail(current_app)
            for sub in subscribers:
                try:
                    msg = Message(
                        subject=subject,
                        recipients=[sub["email"]],
                        html=_build_email_html(title or subject, body),
                    )
                    mail.send(msg)
                except Exception:
                    pass
        except ImportError:
            pass

    return jsonify({"message": f"Newsletter sent to {sent} subscribers", "sent": sent})


# ───────────────────────────────────────
# Email Template
# ───────────────────────────────────────

def _build_email_html(title, body):
    return f"""
    <div style="font-family:Arial;max-width:600px;margin:auto;background:#1a3a2a;color:#fff;padding:30px;border-radius:10px">
        <h2 style="color:#a8d5b0">{title}</h2>
        <p style="line-height:1.6">{body}</p>
        <hr style="border-color:#2e5a45">
        <p style="font-size:12px;color:#aaa">Cardamom Disease Detection System</p>
    </div>
    """