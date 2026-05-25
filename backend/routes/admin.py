from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime, timedelta
from bson import ObjectId
from backend.database.mongo import users, predictions, contacts

admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")


# ───────────────────────────────────────
# Helpers
# ───────────────────────────────────────

def to_oid(id_str):
    """Convert string to ObjectId; return None if invalid."""
    try:
        return ObjectId(id_str)
    except Exception:
        return None


def find_by_id(collection, id_str):
    oid = to_oid(id_str)
    return collection.find_one({"_id": oid if oid else id_str})


def check_admin():
    user_id = get_jwt_identity()
    user = find_by_id(users, user_id)
    return user if user and user.get("is_admin") else None


def serialize(doc):
    if not doc:
        return doc

    if "_id" in doc:
        doc["_id"] = str(doc["_id"])

    if "created_at" in doc and doc["created_at"]:
        doc["created_at"] = doc["created_at"].isoformat()

    return doc


# ───────────────────────────────────────
# Debug (remove in production)
# ───────────────────────────────────────

@admin_bp.route("/whoami", methods=["GET"])
@jwt_required()
def whoami():
    user_id = get_jwt_identity()
    user = find_by_id(users, user_id)
    return jsonify({
        "jwt_identity": user_id,
        "found": user is not None,
        "is_admin": user.get("is_admin") if user else None,
    })


# ───────────────────────────────────────
# Stats
# ───────────────────────────────────────

@admin_bp.route("/stats", methods=["GET"])
@jwt_required()
def stats():
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    total_users = users.count_documents({})
    total_predictions = predictions.count_documents({})

    disease_counts = {
        item["_id"]: item["count"]
        for item in predictions.aggregate([
            {"$group": {"_id": "$disease", "count": {"$sum": 1}}}
        ])
    }

    daily = []
    for i in range(6, -1, -1):
        day = datetime.utcnow() - timedelta(days=i)
        start = datetime(day.year, day.month, day.day)
        end = start + timedelta(days=1)
        daily.append({
            "date": start.strftime("%b %d"),
            "count": predictions.count_documents({
                "created_at": {"$gte": start, "$lt": end}
            }),
        })

    week_ago = datetime.utcnow() - timedelta(days=7)
    avg_result = list(predictions.aggregate([
        {"$group": {"_id": None, "avg": {"$avg": "$confidence"}}}
    ]))

    return jsonify({
        "total_users": total_users,
        "total_predictions": total_predictions,
        "new_users_week": users.count_documents({"created_at": {"$gte": week_ago}}),
        "new_preds_week": predictions.count_documents({"created_at": {"$gte": week_ago}}),
        "avg_confidence": round(avg_result[0]["avg"], 2) if avg_result else 0,
        "disease_counts": disease_counts,
        "daily_predictions": daily,
    })


# ───────────────────────────────────────
# Users
# ───────────────────────────────────────

@admin_bp.route("/users", methods=["GET"])
@jwt_required()
def get_users():
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 8))
    search = request.args.get("search", "").strip()

    query = {}
    if search:
        query["$or"] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"email": {"$regex": search, "$options": "i"}},
        ]

    skip = (page - 1) * limit
    total = users.count_documents(query)

    user_list = []
    for u in users.find(query).skip(skip).limit(limit):
        u = serialize(u)
        u["prediction_count"] = predictions.count_documents(
            {"user_id": u["_id"]})
        user_list.append(u)

    return jsonify({
        "users": user_list,
        "total": total,
        "page": page,
        "pages": (total + limit - 1) // limit,  # ceiling division
    })


@admin_bp.route("/users/<uid>", methods=["DELETE"])
@jwt_required()
def delete_user(uid):
    admin = check_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    if uid == str(admin["_id"]):
        return jsonify({"error": "Cannot delete yourself"}), 400

    oid = to_oid(uid)
    users.delete_one({"_id": oid if oid else uid})
    predictions.delete_many({"user_id": uid})
    return jsonify({"message": "User deleted successfully"})


@admin_bp.route("/users/<uid>/make-admin", methods=["POST"])
@jwt_required()
def make_admin(uid):
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    oid = to_oid(uid)
    users.update_one({"_id": oid if oid else uid},
                     {"$set": {"is_admin": True}})
    return jsonify({"message": "User promoted to admin"})


@admin_bp.route("/users/<uid>/revoke-admin", methods=["POST"])
@jwt_required()
def revoke_admin(uid):
    admin = check_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    if uid == str(admin["_id"]):
        return jsonify({"error": "Cannot revoke yourself"}), 400

    oid = to_oid(uid)
    users.update_one({"_id": oid if oid else uid},
                     {"$set": {"is_admin": False}})
    return jsonify({"message": "Admin role revoked"})


# ───────────────────────────────────────
# Predictions
# ───────────────────────────────────────

@admin_bp.route("/predictions", methods=["GET"])
@jwt_required()
def get_predictions():
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 10))
    disease = request.args.get("disease")

    query = {}
    if disease:
        query["disease"] = disease

    skip = (page - 1) * limit
    total = predictions.count_documents(query)

    data = []
    for p in predictions.find(query).skip(skip).limit(limit):
        p = serialize(p)
        user = find_by_id(users, p.get("user_id"))
        p["user_name"] = user["name"] if user else "—"
        p["user_email"] = user["email"] if user else "—"
        data.append(p)

    return jsonify({
        "predictions": data,
        "total": total,
        "page": page,
        "pages": (total + limit - 1) // limit,
    })


@admin_bp.route("/predictions/<pid>", methods=["DELETE"])
@jwt_required()
def delete_prediction(pid):
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    oid = to_oid(pid)
    predictions.delete_one({"_id": oid if oid else pid})
    return jsonify({"message": "Prediction deleted"})


# ───────────────────────────────────────
# Contact Messages
# ───────────────────────────────────────

@admin_bp.route("/messages", methods=["GET"])
@jwt_required()
def get_messages():
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    msgs = [serialize(m) for m in contacts.find()]
    unread = contacts.count_documents({"is_read": False})
    return jsonify({"messages": msgs, "unread": unread})


@admin_bp.route("/messages/<mid>/read", methods=["POST"])
@jwt_required()
def mark_read(mid):
    if not check_admin():
        return jsonify({"error": "Admin access required"}), 403

    oid = to_oid(mid)
    contacts.update_one({"_id": oid if oid else mid},
                        {"$set": {"is_read": True}})
    return jsonify({"message": "Marked as read"})


# ───────────────────────────────────────
# Public: Save Contact
# ───────────────────────────────────────

@admin_bp.route("/contact", methods=["POST"])
def save_contact():
    data = request.get_json()
    contacts.insert_one({
        "name": data.get("name"),
        "email": data.get("email"),
        "subject": data.get("subject", "No subject"),
        "message": data.get("message"),
        "is_read": False,
        "created_at": datetime.utcnow(),
    })
    return jsonify({"message": "Message received"})
