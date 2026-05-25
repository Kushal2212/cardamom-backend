import os
import uuid
from datetime import datetime

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename

from backend.database.mongo import predictions

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from backend.src.predict import predict as ml_predict, _load_resources  # ML model


predict_bp = Blueprint("predict", __name__, url_prefix="/api")

ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}


# ═══════════════════════════════════════
# Helper
# ═══════════════════════════════════════
def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ═══════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════
@predict_bp.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    user_id = get_jwt_identity()

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "" or not allowed(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # Save file
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"

    upload_dir = os.path.join(current_app.root_path, "static", "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    filepath = os.path.join(upload_dir, secure_filename(filename))
    file.save(filepath)

    # ── ML Prediction ─────────────────────────────
    try:
        result = ml_predict(filepath)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if "error" in result:
        return jsonify(result), 200

    if "disease" not in result:
        return jsonify({"error": "Invalid model output"}), 500

    # ── GradCAM (optional) ────────────────────────
    gradcam_url = None
    try:
        from backend.src.gradcam import generate_gradcam_pp_auto

        model, _, index_to_class = _load_resources()

        class_idx = None
        for idx, cls in index_to_class.items():
            if cls == result["disease"]:
                class_idx = idx
                break

        if class_idx is not None:
            cam_name = f"{uuid.uuid4().hex}_gradcam.jpg"
            cam_path = os.path.join(upload_dir, cam_name)

            ok = generate_gradcam_pp_auto(filepath, model, class_idx, cam_path)

            if ok:
                gradcam_url = f"/static/uploads/{cam_name}"

    except Exception as e:
        print("GradCAM skipped:", e)

    # ── Save to MongoDB ───────────────────────────
    doc = {
        "user_id": user_id,
        "disease": result["disease"],
        "confidence": result["confidence"],
        "severity": result["severity"],
        "nepali_name": result.get("nepali"),
        "recommendation": result.get("recommendation"),
        "image_filename": filename,
        "created_at": datetime.utcnow()
    }

    inserted = predictions.insert_one(doc)

    return jsonify({
        "prediction_id": str(inserted.inserted_id),
        "disease": result["disease"],
        "confidence": result["confidence"],
        "severity": result["severity"],
        "nepali": result.get("nepali"),
        "description": result.get("description"),
        "recommendation": result.get("recommendation"),
        "image_url": f"/static/uploads/{filename}",
        "gradcam_url": gradcam_url,
        "saved": True
    })


# ═══════════════════════════════════════
# HISTORY
# ═══════════════════════════════════════
@predict_bp.route("/history", methods=["GET"])
@jwt_required()
def history():
    user_id = get_jwt_identity()

    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 10))

    skip = (page - 1) * limit

    cursor = predictions.find({"user_id": user_id}) \
        .sort("created_at", -1) \
        .skip(skip) \
        .limit(limit)

    data = []
    for p in cursor:
        p["_id"] = str(p["_id"])
        data.append(p)

    total = predictions.count_documents({"user_id": user_id})

    return jsonify({
        "predictions": data,
        "total": total,
        "page": page,
        "pages": (total // limit) + 1
    })


# ═══════════════════════════════════════
# HISTORY DETAIL
# ═══════════════════════════════════════
@predict_bp.route("/history/<pid>", methods=["GET"])
@jwt_required()
def history_detail(pid):
    user_id = get_jwt_identity()

    record = predictions.find_one({
        "_id": pid,
        "user_id": user_id
    })

    if not record:
        return jsonify({"error": "Not found"}), 404

    record["_id"] = str(record["_id"])

    return jsonify({"prediction": record})


# ═══════════════════════════════════════
# DELETE PREDICTION
# ═══════════════════════════════════════
@predict_bp.route("/history/<pid>", methods=["DELETE"])
@jwt_required()
def delete_prediction(pid):
    user_id = get_jwt_identity()

    record = predictions.find_one({
        "_id": pid,
        "user_id": user_id
    })

    if not record:
        return jsonify({"error": "Not found"}), 404

    # delete image
    if record.get("image_filename"):
        img_path = os.path.join(
            current_app.root_path,
            "static",
            "uploads",
            record["image_filename"]
        )
        if os.path.exists(img_path):
            os.remove(img_path)

    predictions.delete_one({"_id": pid})

    return jsonify({"message": "Deleted"})


# ═══════════════════════════════════════
# STATS
# ═══════════════════════════════════════
@predict_bp.route("/stats", methods=["GET"])
@jwt_required()
def stats():
    user_id = get_jwt_identity()

    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": "$disease", "count": {"$sum": 1}}}
    ]

    result = predictions.aggregate(pipeline)

    counts = {r["_id"]: r["count"] for r in result}

    total = predictions.count_documents({"user_id": user_id})

    return jsonify({
        "total_predictions": total,
        "disease_counts": counts
    })