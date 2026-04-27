"""
DrawMe - Flask API Server (v3 — Multiplayer Edition)
Receives drawing data from the frontend, preprocesses it,
and returns CNN predictions. Includes real-time multiplayer game via Socket.IO.
"""

import os
import io
import json
import base64
import time
import ctypes
import glob
import site
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import eventlet
eventlet.monkey_patch()


def preload_cuda_libraries():
    """Preload NVIDIA pip-package shared libraries before importing TensorFlow."""
    lib_dirs = []
    for root in site.getsitepackages() + [site.getusersitepackages()]:
        lib_dirs.extend(glob.glob(os.path.join(root, "nvidia", "*", "lib")))

    loaded_count = 0
    for lib_dir in sorted(set(lib_dirs)):
        if not os.path.isdir(lib_dir):
            continue
        for so_path in sorted(glob.glob(os.path.join(lib_dir, "lib*.so*"))):
            try:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
                loaded_count += 1
            except OSError:
                pass

    if loaded_count:
        print(f"[Runtime] Preloaded {loaded_count} CUDA shared libraries")


preload_cuda_libraries()

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from tensorflow import keras

from utils import preprocess_canvas_image
from game import RoomManager, CONFIDENCE_THRESHOLD

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model", "saved")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Load only best-checkpoint artifacts.
MODEL_PATH_KERAS = os.path.join(MODEL_DIR, "drawme_best.keras")
MODEL_PATH_H5 = os.path.join(MODEL_DIR, "drawme_best.h5")
CATEGORIES_PATH = os.path.join(MODEL_DIR, "categories.json")

# ─── App Setup ───────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet",
                    ping_timeout=30, ping_interval=10)

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

# ─── Game Room Manager ───────────────────────────────────────────────────────

room_manager = RoomManager(categories=CATEGORIES)

# ─── HTTP Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/game")
def game_page():
    """Serve the multiplayer game page."""
    return send_from_directory(FRONTEND_DIR, "game.html")


@app.route("/<path:filename>")
def serve_static(filename):
    """Serve static frontend files."""
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Receive a Base64-encoded drawing and return predictions.
    (Original single-player endpoint — preserved for backward compat)
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
            debug_img = (processed_image[0, :, :, 0] * 255).astype("uint8")
            pil_img = Image.fromarray(debug_img, mode="L")
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


# ─── Helper: Run prediction on image data ────────────────────────────────────

def run_prediction(image_data: str) -> list:
    """Run the model on a base64 image and return sorted predictions."""
    if model is None:
        return []

    try:
        processed = preprocess_canvas_image(image_data, debug=False)
        preds = model.predict(processed, verbose=0)
        probs = preds[0]

        results = []
        for idx, prob in enumerate(probs):
            results.append({
                "class": CATEGORIES[idx],
                "confidence": round(float(prob), 4)
            })
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
    except Exception as e:
        print(f"  ✗ Prediction error: {e}")
        return []


# ─── Socket.IO Events ───────────────────────────────────────────────────────

@socketio.on("connect")
def handle_connect():
    print(f"  ⚡ Client connected: {request.sid}")


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    print(f"  ⚡ Client disconnected: {sid}")

    room = room_manager.get_player_room(sid)
    if room:
        room_id = room.room_id
        room_manager.leave_room(sid)

        if room.state == "playing":
            # Opponent wins by default
            opponent_id = room.get_opponent_id(sid)
            if opponent_id:
                room.winner = opponent_id
                room.end_game()
                emit("player_disconnected", {
                    "player_id": sid,
                    "message": "Your opponent disconnected. You win!"
                }, room=room_id)
                emit("game_end", room.get_results(), room=room_id)
        else:
            emit("player_left", {"player_id": sid}, room=room_id)


@socketio.on("create_room")
def handle_create_room(data):
    sid = request.sid
    player_name = data.get("player_name", "Player 1")
    num_rounds = min(max(int(data.get("num_rounds", 5)), 1), 10)
    timer_duration = min(max(int(data.get("timer_duration", 60)), 30), 180)

    room = room_manager.create_room(
        player_id=sid,
        player_name=player_name,
        num_rounds=num_rounds,
        timer_duration=timer_duration,
    )

    join_room(room.room_id)

    emit("room_created", {
        "room_id": room.room_id,
        "player_id": sid,
        "player_name": player_name,
        "num_rounds": num_rounds,
        "timer_duration": timer_duration,
    })

    print(f"  🏠 Room {room.room_id} created by {player_name} ({sid})")


@socketio.on("join_room")
def handle_join_room(data):
    sid = request.sid
    room_id = data.get("room_id", "").strip().upper()
    player_name = data.get("player_name", "Player 2")

    if not room_id:
        emit("error", {"message": "Room code is required."})
        return

    room = room_manager.join_room(
        player_id=sid,
        player_name=player_name,
        room_id=room_id,
    )

    if room is None:
        emit("error", {"message": f"Room '{room_id}' not found or is full."})
        return

    join_room(room_id)

    # Notify everyone in the room
    emit("player_joined", {
        "room_id": room_id,
        "player_id": sid,
        "player_name": player_name,
        "players": {pid: p.to_dict() for pid, p in room.players.items()},
        "is_full": room.is_full,
    }, room=room_id)

    print(f"  🏠 {player_name} ({sid}) joined room {room_id}")


@socketio.on("player_ready")
def handle_player_ready(data):
    sid = request.sid
    room = room_manager.get_player_room(sid)

    if room is None:
        emit("error", {"message": "You are not in a room."})
        return

    all_ready = room.set_ready(sid)

    emit("ready_update", {
        "player_id": sid,
        "all_ready": all_ready,
        "players": {pid: p.to_dict() for pid, p in room.players.items()},
    }, room=room.room_id)

    if all_ready:
        # Start the game!
        words = room.start_game()
        if words:
            emit("game_start", {
                "words": words,
                "num_rounds": room.num_rounds,
                "timer_duration": room.timer_duration,
                "players": {pid: p.to_dict() for pid, p in room.players.items()},
            }, room=room.room_id)

            # Start server-controlled timer
            socketio.start_background_task(
                target=game_timer_loop,
                room_id=room.room_id,
            )

            print(f"  🎮 Game started in room {room.room_id}: {words}")


@socketio.on("predict_frame")
def handle_predict_frame(data):
    """Handle a drawing frame prediction request during the game."""
    sid = request.sid
    image_data = data.get("image", "")

    if not image_data:
        return

    room = room_manager.get_player_room(sid)
    if room is None or room.state != "playing":
        return

    # Run ML prediction
    predictions = run_prediction(image_data)
    if not predictions:
        return

    top = predictions[0]

    # Check if prediction matches target
    result = room.check_prediction(sid, top["class"], top["confidence"])

    # Send prediction result back to the player
    emit("prediction_result", {
        "top_prediction": top["class"],
        "confidence": top["confidence"],
        "top3": predictions[:3],
        "matched": result.get("matched", False),
    })

    if result.get("matched"):
        # Notify player of round completion
        player = room.players[sid]
        emit("round_complete", {
            "round_index": result["round_index"],
            "time_taken": result["time_taken"],
            "confidence": result["confidence"],
            "completed": player.completed,
            "next_word": room.words[player.current_index] if player.current_index < room.num_rounds else None,
            "all_done": result.get("all_done", False),
        })

        # Broadcast progress to all players
        emit("progress_update", {
            "players": {pid: p.to_dict() for pid, p in room.players.items()},
        }, room=room.room_id)

        # If player finished all rounds, game might be over
        if result.get("all_done") and room.state == "finished":
            emit("game_end", room.get_results(), room=room.room_id)


@socketio.on("leave_room")
def handle_leave_room(data):
    sid = request.sid
    room = room_manager.get_player_room(sid)

    if room:
        room_id = room.room_id
        leave_room(room_id)

        if room.state == "playing":
            opponent_id = room.get_opponent_id(sid)
            if opponent_id:
                room.winner = opponent_id
                room.end_game()
                emit("player_disconnected", {
                    "player_id": sid,
                    "message": "Your opponent left the game. You win!"
                }, room=room_id)
                emit("game_end", room.get_results(), room=room_id)

        room_manager.leave_room(sid)
        emit("left_room", {"room_id": room_id})


# ─── Game Timer Background Task ──────────────────────────────────────────────

def game_timer_loop(room_id: str):
    """Server-controlled timer that ticks every second."""
    while True:
        eventlet.sleep(1)

        room = room_manager.get_room(room_id)
        if room is None or room.state != "playing":
            break

        remaining = room.tick_timer()

        socketio.emit("timer_update", {
            "time_remaining": remaining,
        }, room=room_id)

        if remaining <= 0:
            socketio.emit("game_end", room.get_results(), room=room_id)
            break


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  🎨  DrawMe API Server (v3 — Multiplayer)")
    print("=" * 50)
    print(f"  Frontend:    http://localhost:5050")
    print(f"  Multiplayer: http://localhost:5050/game")
    print(f"  API:         http://localhost:5050/api/predict")
    print("=" * 50 + "\n")

    socketio.run(app, host="0.0.0.0", port=5050, debug=True, use_reloader=False)
