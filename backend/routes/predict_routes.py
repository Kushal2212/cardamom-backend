import os
import uuid
from pymongo import response
import requests
import base64
from datetime import datetime

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename

from database.mongo import predictions

predict_bp = Blueprint("predict", __name__, url_prefix="/api")

ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}
HF_API_URL = os.environ.get(
    "HF_API_URL", "https://kushal2212-cardamom-model.hf.space")


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def call_huggingface(image_path):
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    # ✅ Send as plain string, not dict
    payload = {
        "data": [
            f"data:{mime};base64,{image_b64}",
            "EfficientNet"
        ]
    }

    response = requests.post(
        f"{HF_API_URL}/gradio_api/call/predict",
        json=payload,
        timeout=60
    )
    response.raise_for_status()

    event_id = response.json().get("event_id")
    result_response = requests.get(
        f"{HF_API_URL}/gradio_api/call/predict/{event_id}",
        timeout=60
    )
    for line in result_response.text.split("\n"):
        if line.startswith("data: "):
            import json
            data = json.loads(line[6:])
            return data[0] if data else {}
    return {}


@predict_bp.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    user_id = get_jwt_identity()

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    upload_dir = os.path.join(current_app.root_path, "static", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, secure_filename(filename))
    file.save(filepath)

    try:
        result = call_huggingface(filepath)
    except requests.exceptions.Timeout:
        return jsonify({"error": "Model server timed out. Try again."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if "error" in result:
        return jsonify(result), 200

    if "disease" not in result:
        return jsonify({"error": "Invalid model output"}), 500

    doc = {
        "user_id":          user_id,
        "disease":          result["disease"],
        "confidence":       result["confidence"],
        "severity":         result.get("severity"),
        "nepali_name":      result.get("nepali"),
        "recommendation":   result.get("recommendation"),
        "image_filename":   filename,
        "all_predictions":  result.get("all_predictions", []),
        "model_details":    result.get("model_details", {}),
        "created_at":       datetime.utcnow()
    }
    inserted = predictions.insert_one(doc)

    return jsonify({
        "prediction_id":    str(inserted.inserted_id),
        "disease":          result["disease"],
        "confidence":       result["confidence"],
        "confidence_level": result.get("confidence_level"),
        "confidence_label": result.get("confidence_label"),
        "low_confidence":   result.get("low_confidence", False),
        "severity":         result.get("severity"),
        "nepali":           result.get("nepali"),
        "description":      result.get("description"),
        "recommendation":   result.get("recommendation"),
        "image_url":        f"/static/uploads/{filename}",
        "gradcam_url":      None,
        "all_predictions":  result.get("all_predictions", []),
        "saved":            True
    })
