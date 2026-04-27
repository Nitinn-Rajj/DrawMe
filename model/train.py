"""
DrawMe - CNN Training Script (v2 — Improved)
Trains a Convolutional Neural Network on the Quick, Draw! dataset.

Improvements over v1:
    - BatchNormalization for training stability
    - 3 convolutional blocks (32 → 64 → 128) for better feature extraction
    - Data augmentation (rotation, shift, zoom) for robustness
    - More training data (20,000 samples per class)
    - Learning rate reduction on plateau
    - Early stopping to prevent overfitting
    - Larger dense layer (256 units)
"""

import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ─── Configuration ───────────────────────────────────────────────────────────

CATEGORIES = [
    "cloud", "sun", "tree", "car", "fish",
    "cat", "dog", "house", "star", "flower",
    "bird", "bicycle", "guitar", "moon", "hat"
]

SAMPLES_PER_CLASS = 20000   # Increased from 10K for better generalization
IMG_SIZE = 28
NUM_CLASSES = len(CATEGORIES)
EPOCHS = 30
BATCH_SIZE = 256
TEST_SPLIT = 0.2
RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")


# ─── Data Loading & Preprocessing ───────────────────────────────────────────

def load_data():
    """
    Load .npy files for each category, balance, normalize, and reshape.

    Preprocessing steps:
        1. Class Balancing: Take exactly SAMPLES_PER_CLASS from each category
        2. Normalization: Scale pixel values from [0, 255] to [0, 1]
        3. Reshaping: Reshape from flat 784-vector to (28, 28, 1) image tensor
    """
    X_all = []
    y_all = []

    print(f"\n📦 Loading data from: {DATA_DIR}")
    print(f"   Samples per class: {SAMPLES_PER_CLASS}")
    print(f"   Categories: {NUM_CLASSES}\n")

    for idx, category in enumerate(CATEGORIES):
        filepath = os.path.join(DATA_DIR, f"{category}.npy")

        if not os.path.exists(filepath):
            print(f"  ✗ Missing: {category}.npy — run download_data.py first!")
            sys.exit(1)

        # Load the full numpy array (shape: [N, 784])
        data = np.load(filepath)
        available = data.shape[0]
        use_count = min(SAMPLES_PER_CLASS, available)
        print(f"  [{idx + 1:2d}/{NUM_CLASSES}] {category:12s} — {available:>7,} available, using {use_count:,}")

        # Class balancing: randomly sample SAMPLES_PER_CLASS
        if available >= SAMPLES_PER_CLASS:
            indices = np.random.RandomState(RANDOM_SEED).choice(available, SAMPLES_PER_CLASS, replace=False)
            data = data[indices]
        else:
            data = data[:available]

        X_all.append(data)
        y_all.append(np.full(data.shape[0], idx))

    # Concatenate all categories
    X = np.concatenate(X_all, axis=0).astype("float32")
    y = np.concatenate(y_all, axis=0)

    # Shuffle the combined dataset
    shuffle_idx = np.random.RandomState(RANDOM_SEED).permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    # Normalize pixel values: [0, 255] → [0.0, 1.0]
    X = X / 255.0

    # Reshape: (N, 784) → (N, 28, 28, 1)
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    print(f"\n  Total samples: {X.shape[0]:,}")
    print(f"  Image shape:   {X.shape[1:]}")
    print(f"  Labels shape:  {y.shape}")

    return X, y


# ─── Data Augmentation ──────────────────────────────────────────────────────

def create_data_augmentation():
    """
    Create a data augmentation pipeline.

    This is critical for sketch recognition because users draw differently
    than the training data — slightly rotated, shifted, or scaled.
    These augmentations make the model robust to these real-world variations.
    """
    return keras.Sequential([
        layers.RandomRotation(
            factor=0.08,          # ±29° (0.08 * 360)
            fill_mode="constant",
            fill_value=0.0
        ),
        layers.RandomTranslation(
            height_factor=0.1,    # ±10% vertical shift
            width_factor=0.1,     # ±10% horizontal shift
            fill_mode="constant",
            fill_value=0.0
        ),
        layers.RandomZoom(
            height_factor=(-0.1, 0.1),  # ±10% zoom
            fill_mode="constant",
            fill_value=0.0
        ),
    ], name="data_augmentation")


# ─── Model Definition ───────────────────────────────────────────────────────

def build_model():
    """
    Build an improved CNN architecture with BatchNorm and 3 conv blocks.

    Architecture:
        [Data Augmentation] (training only)
        Conv2D(32, 3×3)  → BatchNorm → ReLU → MaxPool(2×2)
        Conv2D(64, 3×3)  → BatchNorm → ReLU → MaxPool(2×2)
        Conv2D(128, 3×3) → BatchNorm → ReLU
        GlobalAveragePooling2D
        Dense(256) → BatchNorm → ReLU → Dropout(0.4)
        Dense(15, Softmax)

    Key improvements:
        - BatchNorm stabilizes training and allows faster convergence
        - 3rd conv block captures more complex patterns
        - GlobalAveragePooling reduces overfitting vs Flatten
        - Larger Dense(256) for more representational capacity
        - Higher dropout (0.4) for better regularization
    """
    data_augmentation = create_data_augmentation()

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="input")

    # Apply data augmentation during training only
    x = data_augmentation(inputs)

    # --- Block 1: Basic strokes (edges, lines) ---
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # --- Block 2: Complex shapes (loops, curves, corners) ---
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # --- Block 3: High-level features (object parts) ---
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)

    # --- Classification Head ---
    # GlobalAveragePooling is better than Flatten for generalization
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(256, name="dense1")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.Dropout(0.4, name="dropout")(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="DrawMe_CNN_v2")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ─── Training ───────────────────────────────────────────────────────────────

def train():
    """Train the model with callbacks for better training dynamics."""
    print("=" * 60)
    print("  🎨  DrawMe — CNN Training Pipeline (v2)")
    print("=" * 60)

    # 1. Load & preprocess data
    X, y = load_data()

    # 2. Train/Test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\n  Train set: {X_train.shape[0]:,} samples")
    print(f"  Test set:  {X_test.shape[0]:,} samples")

    # 3. Build model
    print("\n🧠 Building model...\n")
    model = build_model()
    model.summary()

    # 4. Setup callbacks
    cb_list = [
        # Reduce learning rate when validation loss plateaus
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,           # Halve the LR
            patience=3,           # Wait 3 epochs
            min_lr=1e-6,
            verbose=1
        ),
        # Stop training if no improvement
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,           # Wait 7 epochs
            restore_best_weights=True,
            verbose=1
        ),
    ]

    # 5. Train
    print(f"\n🚀 Training for up to {EPOCHS} epochs (with early stopping)...\n")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=cb_list,
        verbose=1
    )

    # 6. Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    # 7. Save model
    os.makedirs(SAVE_DIR, exist_ok=True)

    model_path = os.path.join(SAVE_DIR, "drawme_model.keras")
    model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")

    # Also save as .h5 for backwards compatibility
    model_path_h5 = os.path.join(SAVE_DIR, "drawme_model.h5")
    model.save(model_path_h5)
    print(f"💾 Model saved (h5) to: {model_path_h5}")

    # Save category mapping
    categories_path = os.path.join(SAVE_DIR, "categories.json")
    with open(categories_path, "w") as f:
        json.dump(CATEGORIES, f, indent=2)
    print(f"📋 Categories saved to: {categories_path}")

    # Save training history
    history_path = os.path.join(SAVE_DIR, "training_history.json")
    with open(history_path, "w") as f:
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        json.dump(history_dict, f, indent=2)
    print(f"📈 Training history saved to: {history_path}")

    print("\n" + "=" * 60)
    print("  ✅  Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    train()
