# top of app.py
from database.mongo import users
from extensions import bcrypt, jwt
from flask_cors import CORS
from flask_jwt_extended import get_jwt_identity, jwt_required
from datetime import timedelta
from flask import Flask, jsonify
from bson import ObjectId
import os
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_app():
    app = Flask(__name__, static_folder=None)

    try:
        from load_models import ensure_models
        ensure_models()
    except Exception as e:
        print("Model load error:", e)

    CORS(app, resources={
        r"/api/*": {
            "origins": [
                "http://127.0.0.1:5500",
                "http://localhost:5500",
                "http://127.0.0.1:5000",
                "http://localhost:5000",
            ]
        }
    })

    app.config["SECRET_KEY"] = os.environ.get(
        "SECRET_KEY", "a8bcf6bc5ffd994ac6dd154b54298185dfa72c8a37f072610ca5bf4c17b89a40")
    app.config["JWT_SECRET_KEY"] = os.environ.get(
        "JWT_SECRET_KEY", "7e5668b66e65f66f9eab2331962d72e45e8448ebc249a457a9f122d5218c2de5")
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=7)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    app.config["MAIL_SERVER"] = os.environ.get("MAIL_SERVER", "")
    app.config["MAIL_PORT"] = int(os.environ.get("MAIL_PORT", 587))
    app.config["MAIL_USE_TLS"] = True
    app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME", "")
    app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD", "")
    app.config["MAIL_DEFAULT_SENDER"] = os.environ.get(
        "MAIL_USERNAME", "noreply@cardamomdx.com")

    bcrypt.init_app(app)
    jwt.init_app(app)

    # ── Core blueprints ──────────────────────────────────────────────────
    from routes.auth import auth_bp
    from routes.predict_routes import predict_bp
    from routes.admin import admin_bp
    from routes.weather import weather_bp
    from routes.newsletter_routes import newsletter_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(weather_bp)
    app.register_blueprint(newsletter_bp)

    # ── Optional contact routes ──────────────────────────────────────────
    try:
        from backend.routes.contact_routes import contact_bp
        app.register_blueprint(contact_bp)
    except ImportError:
        print("⚠️ Contact routes not loaded")

    # ── SMS system ───────────────────────────────────────────────────────
    try:
        from backend.routes.sms_alert_system import sms_bp, start_scheduler
        app.register_blueprint(sms_bp)
        start_scheduler(app)
        print("✅ SMS system loaded")
    except ImportError:
        print("⚠️ SMS system not loaded")
    except Exception as e:
        print(f"⚠️ SMS system error: {e}")

    # ── Utility routes ───────────────────────────────────────────────────
    @app.route("/api/test-token", methods=["GET"])
    @jwt_required()
    def test_token():
        user_id = get_jwt_identity()
        return jsonify({"user_id": user_id, "message": "Token works!"})



    from flask import jsonify, session

    @app.route("/api/profile", methods=["GET"])
    @jwt_required()
    def get_profile():

        user_id = get_jwt_identity()

        user = users.find_one(
            {"_id": ObjectId(user_id)},
            {"_id": 0, "password": 0}
    )

        if not user:
            return jsonify({"error": "User not found"}), 404

        return jsonify(user), 200

    @app.route("/static/uploads/<path:filename>")
    def serve_uploads(filename):
        uploads_dir = os.path.join(BASE_DIR, "static", "uploads")
        return send_from_directory(uploads_dir, filename)

    print("✅ Database tables ready")
    return app


if __name__ == "__main__":
    app = create_app()
    print("\n🌿 Cardamom Disease Detection System")
    print("   Web:  http://127.0.0.1:5000")
    print("   API:  http://127.0.0.1:5000/api\n")
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
