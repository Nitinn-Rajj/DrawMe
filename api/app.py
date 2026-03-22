"""
DrawMe - Flask API Server (v2 — Improved)
Receives drawing data from the frontend, preprocesses it,
and returns CNN predictions. Includes debug endpoint for preprocessing visualization.
"""

import os
import io
import json
import base64
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow import keras

from utils import preprocess_canvas_image

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model", "saved")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Try .keras first (new format), then fall back to .h5
MODEL_PATH_KERAS = os.path.join(MODEL_DIR, "drawme_model.keras")
MODEL_PATH_H5 = os.path.join(MODEL_DIR, "drawme_model.h5")
CATEGORIES_PATH = os.path.join(MODEL_DIR, "categories.json")

# ─── App Setup ───────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# ─── Load Model & Categories ────────────────────────────────────────────────

print("🧠 Loading model...")
model = None

if os.path.exists(MODEL_PATH_KERAS):
    model = keras.models.load_model(MODEL_PATH_KERAS)
    print(f"  ✓ Model loaded from: {MODEL_PATH_KERAS}")
elif os.path.exists(MODEL_PATH_H5):
    model = keras.models.load_model(MODEL_PATH_H5)
    print(f"  ✓ Model loaded from: {MODEL_PATH_H5}")
else:
    print(f"  ✗ No model found in: {MODEL_DIR}")
    print("    Train the model first: python model/train.py")

if os.path.exists(CATEGORIES_PATH):
    with open(CATEGORIES_PATH, "r") as f:
        CATEGORIES = json.load(f)
    print(f"  ✓ Categories loaded: {CATEGORIES}")
else:
    CATEGORIES = [
        "cloud", "sun", "tree", "car", "fish",
        "cat", "dog", "house", "star", "flower",
        "bird", "bicycle", "guitar", "moon", "hat"
    ]
    print("  ⚠ Using default categories")


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    """Serve static frontend files."""
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Receive a Base64-encoded drawing and return predictions.

    Request JSON:
        { "image": "data:image/png;base64,iVBOR..." }

    Response JSON:
        {
            "success": true,
            "predictions": [
                { "class": "cat", "confidence": 0.87 },
                ...
            ],
            "debug_image": "data:image/png;base64,..."  (if debug=true in request)
        }
    """
    if model is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded. Run 'python model/train.py' first."
        }), 503

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({
            "success": False,
            "error": "No image data provided. Send JSON with 'image' key."
        }), 400

    try:
        debug_mode = data.get("debug", False)

        # Preprocess the canvas image
        processed_image = preprocess_canvas_image(data["image"], debug=debug_mode)

        # Run prediction
        predictions = model.predict(processed_image, verbose=0)
        probs = predictions[0]

        # Build sorted results
        results = []
        for idx, prob in enumerate(probs):
            results.append({
                "class": CATEGORIES[idx],
                "confidence": round(float(prob), 4)
            })

        # Sort by confidence descending
        results.sort(key=lambda x: x["confidence"], reverse=True)

        response = {
            "success": True,
            "predictions": results
        }

        # If debug mode, include the preprocessed image as base64
        if debug_mode:
            from PIL import Image
            # Convert the processed image back to a visible PNG
            debug_img = (processed_image[0, :, :, 0] * 255).astype("uint8")
            pil_img = Image.fromarray(debug_img, mode="L")
            # Scale up for visibility
            pil_img = pil_img.resize((280, 280), Image.NEAREST)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            buffer.seek(0)
            debug_b64 = base64.b64encode(buffer.read()).decode("utf-8")
            response["debug_image"] = f"data:image/png;base64,{debug_b64}"

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }), 500


@app.route("/api/categories", methods=["GET"])
def get_categories():
    """Return the list of categories the model can recognize."""
    return jsonify({
        "success": True,
        "categories": CATEGORIES
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "num_categories": len(CATEGORIES)
    })


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  🎨  DrawMe API Server (v2)")
    print("=" * 50)
    print(f"  Frontend: http://localhost:5050")
    print(f"  API:      http://localhost:5050/api/predict")
    print("=" * 50 + "\n")

    app.run(host="0.0.0.0", port=5050, debug=True)
