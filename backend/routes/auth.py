from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity

from backend.database.mongo import users
from backend.extensions import bcrypt

import validators
from email_validator import validate_email, EmailNotValidError
from password_strength import PasswordPolicy

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

# ── Password policy ─────────────────────────────
policy = PasswordPolicy.from_names(
    length=8,
    uppercase=1,
    numbers=1,
    special=1
)

# ═══════════════════════════════════════
# REGISTER
# ═══════════════════════════════════════
@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    # ── Validation functions ───────────────
    def validate_name(name):
        return len(name) >= 3 and name.replace(" ", "").isalpha()

    def validate_password(password):
        errors = policy.test(password)
        return len(errors) == 0

    def check_email(email):
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False

    # ── Validation checks ───────────────────
    if not validate_name(name):
        return jsonify({"error": "Name must be at least 3 letters"}), 400

    if not check_email(email):
        return jsonify({"error": "Invalid email address"}), 400

    if not validate_password(password):
        return jsonify({
            "error": "Password must be 8+ chars with uppercase, number, special char"
        }), 400

    # ── Check duplicate user ────────────────
    if users.find_one({"email": email}):
        return jsonify({"error": "Email already exists"}), 409

    # ── Hash password ───────────────────────
    hashed = bcrypt.generate_password_hash(password).decode("utf-8")

    # ── Create user ─────────────────────────
    user = {
        "name": name,
        "email": email,
        "password": hashed,
        "is_admin": False
    }

    result = users.insert_one(user)

    token = create_access_token(identity=str(result.inserted_id))

    return jsonify({
        "message": "Account created successfully",
        "token": token,
        "user": {
            "id": str(result.inserted_id),
            "name": name,
            "email": email,
            "is_admin": False
        }
    }), 201


# ═══════════════════════════════════════
# LOGIN
# ═══════════════════════════════════════
@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    user = users.find_one({"email": email})

    if not user or not bcrypt.check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid email or password"}), 401

    token = create_access_token(identity=str(user["_id"]))

    return jsonify({
        "message": "Login successful",
        "token": token,
        "user": {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
            "is_admin": user.get("is_admin", False)
        }
    }), 200


# ═══════════════════════════════════════
# PROFILE
# ═══════════════════════════════════════
@auth_bp.route("/profile", methods=["GET"])
@jwt_required()
def profile():
    user_id = get_jwt_identity()

    user = users.find_one({"_id": user_id})

    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "user": {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
            "is_admin": user.get("is_admin", False)
        }
    }), 200


# ═══════════════════════════════════════
# LOGOUT
# ═══════════════════════════════════════
@auth_bp.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    return jsonify({
        "message": "Logged out. Delete token on client side."
    }), 200