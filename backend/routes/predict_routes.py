import os
import uuid
import requests
import base64
from datetime import datetime
import json

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename

from database.mongo import predictions

predict_bp = Blueprint("predict", __name__, url_prefix="/api")

ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}
HF_API_URL = os.environ.get("HF_API_URL", "https://kushal2212-cardamom-model.hf.space")

DISEASE_INFO = {
    "healthy": {
        "nepali": "स्वस्थ",
        "description": "बोट स्वस्थ अवस्थामा छ।",
        "recommendation": "नियमित हेरचाह जारी राख्नुहोस्।",
        "severity": "कुनै जोखिम छैन"
    },
    "chhirke": {
        "nepali": "छिर्के रोग",
        "description": "छिर्के भाइरसजन्य रोग हो।",
        "recommendation": "संक्रमित बोट तुरुन्त उखेलेर नष्ट गर्नुहोस्।",
        "severity": "उच्च जोखिम"
    },
    "leaf_blight": {
        "nepali": "पात झुल्सा रोग",
        "description": "पात झुल्सा ढुसीजन्य रोग हो।",
        "recommendation": "म्यान्कोजेब छर्नुहोस्। संक्रमित पात काटेर नष्ट गर्नुहोस्।",
        "severity": "मध्यम जोखिम"
    }
}


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def call_huggingface(image_path):
    # Step 1: Upload image to HF Space
    with open(image_path, "rb") as f:
        upload_response = requests.post(
            f"{HF_API_URL}/gradio_api/upload",
            files={"files": f},
            timeout=60
        )
    upload_response.raise_for_status()
    uploaded_path = upload_response.json()[0]

    # Step 2: Call predict
    payload = {
        "data": [
            {"path": uploaded_path},
            "EfficientNet"
        ]
    }
    response = requests.post(
        f"{HF_API_URL}/gradio_api/call/predict",
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    event_id = response.json()["event_id"]

    # Step 3: Get result
    result_response = requests.get(
        f"{HF_API_URL}/gradio_api/call/predict/{event_id}",
        timeout=60
    )
    for line in result_response.text.split("\n"):
        if line.startswith("data: "):
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
        hf_result = call_huggingface(filepath)
    except requests.exceptions.Timeout:
        return jsonify({"error": "Model server timed out. Try again."}), 504
    except Exception as e:
        import traceback
        print("PREDICT ERROR:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

    # Map HF response → internal format
    disease = hf_result.get("predicted_class") or hf_result.get("disease")
    confidence = hf_result.get("confidence", 0)

    if not disease:
        return jsonify({"error": "Invalid model output", "raw": hf_result}), 500

    info = DISEASE_INFO.get(disease, {
        "nepali": disease,
        "description": "No information available.",
        "recommendation": "Consult an agricultural expert.",
        "severity": "Unknown"
    })

    doc = {
        "user_id":        user_id,
        "disease":        disease,
        "confidence":     confidence,
        "severity":       info["severity"],
        "nepali_name":    info["nepali"],
        "recommendation": info["recommendation"],
        "image_filename": filename,
        "created_at":     datetime.utcnow()
    }
    inserted = predictions.insert_one(doc)

    return jsonify({
        "prediction_id":  str(inserted.inserted_id),
        "disease":        disease,
        "confidence":     confidence,
        "severity":       info["severity"],
        "nepali":         info["nepali"],
        "description":    info["description"],
        "recommendation": info["recommendation"],
        "image_url":      f"/static/uploads/{filename}",
        "gradcam_url":    None,
        "saved":          True
    }), 200