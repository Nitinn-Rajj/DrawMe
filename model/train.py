"""DrawMe training pipeline with profile-based config and GPU support."""

import argparse
import ctypes
import glob
import json
import os
import random
import site
import sys
import time
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def preload_cuda_libraries():
    """Preload NVIDIA pip-package shared libraries so TensorFlow can use GPU."""
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
                # Some optional libs may not load on every platform; ignore safely.
                pass

    if loaded_count:
        print(f"[Runtime] Preloaded {loaded_count} CUDA shared libraries")


preload_cuda_libraries()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers

from config import load_training_config


class EpochTimingCallback(callbacks.Callback):
    """Collect per-epoch timing so throughput can be tracked between runs."""

    def on_train_begin(self, logs=None):
        self.epoch_durations_sec = []

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        duration = float(time.perf_counter() - self._epoch_start)
        self.epoch_durations_sec.append(duration)


class NumpyBatchSequence(keras.utils.Sequence):
    """Memory-safe batch loader that avoids giant tensor materialization."""

    def __init__(self, x, y, indices, batch_size, shuffle=False, seed=42):
        super().__init__()
        self.x = x
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(int(seed))
        self.order = self.indices.copy()
        if self.shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return int(np.ceil(len(self.order) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.order))
        batch_idx = self.order[start:end]
        return self.x[batch_idx], self.y[batch_idx]

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DrawMe CNN with selectable config profiles.")
    parser.add_argument(
        "--profile",
        default="default",
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
    safe_cap = int(data_cfg.get("safe_samples_per_class_cap", 0))
    if safe_cap > 0 and samples_per_class > safe_cap:
        print(
            f"[WARN] samples_per_class={samples_per_class:,} is too high for stable training; "
            f"capping to {safe_cap:,}."
        )
        samples_per_class = safe_cap
        cfg["data"]["samples_per_class"] = samples_per_class

    img_size = int(data_cfg["img_size"])
    seed = int(cfg["runtime"]["seed"])
    rng = np.random.default_rng(seed)
    use_mixed_precision = bool(cfg["runtime"].get("mixed_precision", False))

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

    x_dtype = "float16" if use_mixed_precision else "float32"
    x = np.concatenate(x_all, axis=0).astype(x_dtype)
    y = np.concatenate(y_all, axis=0).astype("int32")

    permutation = rng.permutation(len(x))
    x = x[permutation]
    y = y[permutation]

    x /= np.array(255.0, dtype=x.dtype)
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
    timing_cb = EpochTimingCallback()
    cb = [callbacks.TerminateOnNaN(), timing_cb]

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
    return cb, timing_cb


def save_artifacts(cfg, runtime_info, history, model, test_metrics, epoch_durations_sec):
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
        "timing": {
            "epochs_ran": len(epoch_durations_sec),
            "epoch_durations_sec": [float(v) for v in epoch_durations_sec],
            "total_training_sec": float(np.sum(epoch_durations_sec)) if epoch_durations_sec else 0.0,
            "avg_epoch_sec": float(np.mean(epoch_durations_sec)) if epoch_durations_sec else 0.0,
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
    all_indices = np.arange(len(x), dtype=np.int64)
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=float(tr_cfg["test_split"]),
        random_state=int(cfg["runtime"]["seed"]),
        stratify=y,
    )

    train_fit_indices, val_indices = train_test_split(
        train_indices,
        test_size=float(tr_cfg["validation_split"]),
        random_state=int(cfg["runtime"]["seed"]),
        stratify=y[train_indices],
    )

    batch_size = int(tr_cfg["batch_size"])
    train_seq = NumpyBatchSequence(
        x=x,
        y=y,
        indices=train_fit_indices,
        batch_size=batch_size,
        shuffle=True,
        seed=int(cfg["runtime"]["seed"]),
    )
    val_seq = NumpyBatchSequence(
        x=x,
        y=y,
        indices=val_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=int(cfg["runtime"]["seed"]),
    )
    test_seq = NumpyBatchSequence(
        x=x,
        y=y,
        indices=test_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=int(cfg["runtime"]["seed"]),
    )

    print(f"\n[Split] Train: {len(train_fit_indices):,} samples")
    print(f"[Split] Val:   {len(val_indices):,} samples")
    print(f"[Split] Test:  {len(test_indices):,} samples")

    model = build_model(cfg)
    print("\n[Model] Summary")
    model.summary()

    save_dir = cfg["paths"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    cb_list, timing_cb = build_callbacks(cfg, save_dir)

    print(f"\n[Train] Starting training for up to {tr_cfg['epochs']} epochs")
    history = model.fit(
        train_seq,
        epochs=int(tr_cfg["epochs"]),
        validation_data=val_seq,
        callbacks=cb_list,
        verbose=1,
    )

    epoch_durations_sec = timing_cb.epoch_durations_sec
    if epoch_durations_sec:
        print(f"[Train] Avg epoch time: {np.mean(epoch_durations_sec):.2f}s")
        print(f"[Train] Total train time: {np.sum(epoch_durations_sec):.2f}s")

    print("\n[Eval] Running test evaluation...")
    test_loss, test_acc = model.evaluate(test_seq, verbose=0)
    print(f"[Eval] Test loss: {test_loss:.4f}")
    print(f"[Eval] Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    save_artifacts(cfg, runtime_info, history, model, (test_loss, test_acc), epoch_durations_sec)

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
