"""DrawMe training pipeline with profile-based config and GPU support."""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers

from config import load_training_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train DrawMe CNN with selectable config profiles.")
    parser.add_argument(
        "--profile",
        default="small_machine",
        choices=["small_machine", "big_machine"],
        help="Built-in training profile to start from.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON config path to override profile values.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and runtime setup without starting training.",
    )
    return parser.parse_args()


def setup_runtime(cfg):
    runtime_cfg = cfg["runtime"]
    seed = int(runtime_cfg["seed"])

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if runtime_cfg.get("deterministic", False):
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    gpus = tf.config.list_physical_devices("GPU")
    use_gpu = bool(runtime_cfg["gpu"].get("enable", True)) and len(gpus) > 0

    if use_gpu and runtime_cfg["gpu"].get("memory_growth", True):
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:
                print(f"[WARN] Could not enable memory growth for {gpu}: {exc}")

    if use_gpu and runtime_cfg.get("mixed_precision", False):
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")
    else:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("float32")

    return {
        "gpu_available": len(gpus) > 0,
        "gpu_enabled": use_gpu,
        "gpu_count": len(gpus),
        "mixed_precision": bool(use_gpu and runtime_cfg.get("mixed_precision", False)),
        "seed": seed,
    }


def load_data(cfg):
    data_cfg = cfg["data"]
    categories = data_cfg["categories"]
    data_dir = cfg["paths"]["data_dir"]
    samples_per_class = int(data_cfg["samples_per_class"])
    img_size = int(data_cfg["img_size"])
    seed = int(cfg["runtime"]["seed"])
    rng = np.random.default_rng(seed)

    x_all = []
    y_all = []

    print(f"\n[Data] Loading from: {data_dir}")
    print(f"[Data] Samples per class target: {samples_per_class}")
    print(f"[Data] Categories: {len(categories)}\n")

    for idx, category in enumerate(categories):
        filepath = os.path.join(data_dir, f"{category}.npy")
        if not os.path.exists(filepath):
            print(f"[ERR] Missing file: {filepath}")
            print("[ERR] Run: python model/download_data.py")
            sys.exit(1)

        arr = np.load(filepath)
        available = arr.shape[0]
        use_count = min(samples_per_class, available)
        print(
            f"  [{idx + 1:2d}/{len(categories)}] {category:12s} "
            f"available={available:>7,} using={use_count:,}"
        )

        if available > use_count:
            indices = rng.choice(available, use_count, replace=False)
            arr = arr[indices]
        else:
            arr = arr[:use_count]

        x_all.append(arr)
        y_all.append(np.full(arr.shape[0], idx))

    x = np.concatenate(x_all, axis=0).astype("float32")
    y = np.concatenate(y_all, axis=0)

    permutation = rng.permutation(len(x))
    x = x[permutation]
    y = y[permutation]

    x /= 255.0
    x = x.reshape(-1, img_size, img_size, 1)

    print(f"\n[Data] Total samples: {x.shape[0]:,}")
    print(f"[Data] Image shape: {x.shape[1:]}")

    return x, y


def create_data_augmentation(cfg):
    aug_cfg = cfg["augmentation"]
    if not aug_cfg.get("enabled", True):
        return None

    return keras.Sequential(
        [
            layers.RandomRotation(
                factor=float(aug_cfg["rotation"]),
                fill_mode="constant",
                fill_value=0.0,
            ),
            layers.RandomTranslation(
                height_factor=float(aug_cfg["translation_height"]),
                width_factor=float(aug_cfg["translation_width"]),
                fill_mode="constant",
                fill_value=0.0,
            ),
            layers.RandomZoom(
                height_factor=(
                    -float(aug_cfg["zoom"]),
                    float(aug_cfg["zoom"]),
                ),
                fill_mode="constant",
                fill_value=0.0,
            ),
        ],
        name="data_augmentation",
    )


def build_model(cfg):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    num_classes = len(data_cfg["categories"])
    input_shape = (int(data_cfg["img_size"]), int(data_cfg["img_size"]), 1)

    inputs = keras.Input(shape=input_shape, name="input")
    x = inputs

    augmentation = create_data_augmentation(cfg)
    if augmentation is not None:
        x = augmentation(x)

    conv_filters = model_cfg["conv_filters"]
    kernel_size = tuple(model_cfg.get("kernel_size", [3, 3]))

    for idx, filters in enumerate(conv_filters, start=1):
        x = layers.Conv2D(
            int(filters),
            kernel_size,
            padding="same",
            use_bias=False,
            name=f"conv{idx}",
        )(x)
        x = layers.BatchNormalization(name=f"bn{idx}")(x)
        x = layers.Activation("relu", name=f"relu{idx}")(x)
        if idx <= len(conv_filters) - 1:
            x = layers.MaxPooling2D((2, 2), name=f"pool{idx}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)

    dense_units = model_cfg["dense_units"]
    dropout = float(model_cfg["dropout"])
    for idx, units in enumerate(dense_units, start=1):
        x = layers.Dense(int(units), use_bias=False, name=f"dense{idx}")(x)
        x = layers.BatchNormalization(name=f"dense_bn{idx}")(x)
        x = layers.Activation("relu", name=f"dense_relu{idx}")(x)
        x = layers.Dropout(dropout, name=f"dropout{idx}")(x)

    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=model_cfg.get("name", "DrawMe_CNN"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(model_cfg["learning_rate"])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_callbacks(cfg, save_dir):
    tr_cfg = cfg["training"]
    cb = [callbacks.TerminateOnNaN()]

    cb.append(
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(tr_cfg["lr_factor"]),
            patience=int(tr_cfg["lr_patience"]),
            min_lr=float(tr_cfg["min_lr"]),
            verbose=1,
        )
    )
    cb.append(
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=int(tr_cfg["early_stopping_patience"]),
            restore_best_weights=True,
            verbose=1,
        )
    )
    cb.append(
        callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, "drawme_best.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )
    )
    return cb


def save_artifacts(cfg, runtime_info, history, model, test_metrics):
    save_dir = cfg["paths"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    keras_path = os.path.join(save_dir, "drawme_model.keras")
    h5_path = os.path.join(save_dir, "drawme_model.h5")
    categories_path = os.path.join(save_dir, "categories.json")
    history_path = os.path.join(save_dir, "training_history.json")
    metadata_path = os.path.join(save_dir, "training_metadata.json")
    config_out_path = os.path.join(save_dir, "resolved_config.json")

    model.save(keras_path)
    model.save(h5_path)

    with open(categories_path, "w", encoding="utf-8") as f:
        json.dump(cfg["data"]["categories"], f, indent=2)

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)

    metadata = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "runtime": runtime_info,
        "dataset": {
            "categories": len(cfg["data"]["categories"]),
            "samples_per_class": int(cfg["data"]["samples_per_class"]),
            "img_size": int(cfg["data"]["img_size"]),
        },
        "model": cfg["model"],
        "training": cfg["training"],
        "evaluation": {
            "test_loss": float(test_metrics[0]),
            "test_accuracy": float(test_metrics[1]),
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(config_out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[Save] Model (.keras): {keras_path}")
    print(f"[Save] Model (.h5): {h5_path}")
    print(f"[Save] Categories: {categories_path}")
    print(f"[Save] History: {history_path}")
    print(f"[Save] Metadata: {metadata_path}")
    print(f"[Save] Resolved config: {config_out_path}")


def train(cfg):
    runtime_info = setup_runtime(cfg)

    print("=" * 72)
    print("DrawMe Training Pipeline")
    print("=" * 72)
    print(f"[Runtime] GPU available: {runtime_info['gpu_available']}")
    print(f"[Runtime] GPU enabled:   {runtime_info['gpu_enabled']}")
    print(f"[Runtime] GPU count:     {runtime_info['gpu_count']}")
    print(f"[Runtime] Mixed precision: {runtime_info['mixed_precision']}")
    print(f"[Runtime] Seed: {runtime_info['seed']}")

    x, y = load_data(cfg)

    tr_cfg = cfg["training"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(tr_cfg["test_split"]),
        random_state=int(cfg["runtime"]["seed"]),
        stratify=y,
    )
    print(f"\n[Split] Train: {x_train.shape[0]:,} samples")
    print(f"[Split] Test:  {x_test.shape[0]:,} samples")

    model = build_model(cfg)
    print("\n[Model] Summary")
    model.summary()

    save_dir = cfg["paths"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    cb_list = build_callbacks(cfg, save_dir)

    print(f"\n[Train] Starting training for up to {tr_cfg['epochs']} epochs")
    history = model.fit(
        x_train,
        y_train,
        epochs=int(tr_cfg["epochs"]),
        batch_size=int(tr_cfg["batch_size"]),
        validation_split=float(tr_cfg["validation_split"]),
        callbacks=cb_list,
        verbose=1,
    )

    print("\n[Eval] Running test evaluation...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[Eval] Test loss: {test_loss:.4f}")
    print(f"[Eval] Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    save_artifacts(cfg, runtime_info, history, model, (test_loss, test_acc))

    print("\n" + "=" * 72)
    print("Training complete")
    print("=" * 72)


def main():
    args = parse_args()
    cfg, cfg_sources = load_training_config(profile=args.profile, override_path=args.config)

    print("[Config] Loaded from:")
    for src in cfg_sources:
        print(f"  - {src}")

    if args.dry_run:
        runtime_info = setup_runtime(cfg)
        print("\n[Dry Run] Runtime setup complete:")
        print(json.dumps(runtime_info, indent=2))
        return

    train(cfg)


if __name__ == "__main__":
    main()
