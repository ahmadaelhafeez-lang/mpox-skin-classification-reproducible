#!/usr/bin/env python3
"""Train/resume CNN models for four-class mpox skin image classification.

This script mirrors the V2 training workflow: class-weighted cross-entropy,
focal loss, transfer learning, fine-tuning, test-time augmentation, and
calibration-aware metrics. It is designed for Kaggle/Colab execution.
"""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix, classification_report

CLASSES = ["Monkeypox", "Chickenpox", "Measles", "Normal"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def expected_calibration_error(y_true, probs, n_bins=10):
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if np.sum(mask) > 0:
            ece += (np.sum(mask) / len(y_true)) * abs(np.mean(correct[mask]) - np.mean(conf[mask]))
    return float(ece)


def brier_score(y_true, probs):
    y_onehot = np.eye(len(CLASSES))[y_true]
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def metric_dict(y_true, probs):
    pred = np.argmax(probs, axis=1)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "macro_precision": precision_score(y_true, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0),
        "nll": log_loss(y_true, probs, labels=np.arange(len(CLASSES))),
        "ece": expected_calibration_error(y_true, probs),
        "brier": brier_score(y_true, probs),
    }


def focal_loss(alpha, gamma=1.5, label_smoothing=0.03):
    alpha = tf.constant(alpha, dtype=tf.float32)
    def loss(y_true, y_pred):
        eps = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        if label_smoothing > 0:
            n = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - label_smoothing) + label_smoothing / n
        ce = -y_true * tf.math.log(y_pred)
        mod = tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(alpha * mod * ce, axis=-1)
    return loss


def build_model(model_name, img_size=384, weights="imagenet", dropout=0.35):
    if model_name == "DenseNet201":
        base_cls = tf.keras.applications.DenseNet201
        preprocess = tf.keras.applications.densenet.preprocess_input
    elif model_name == "EfficientNetB3":
        base_cls = tf.keras.applications.EfficientNetB3
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    elif model_name == "MobileNetV2":
        base_cls = tf.keras.applications.MobileNetV2
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    else:
        raise ValueError(model_name)

    base = base_cls(include_top=False, weights=weights, input_shape=(img_size, img_size, 3))
    base.trainable = False
    inputs = keras.Input((img_size, img_size, 3))
    x = layers.Lambda(lambda z: preprocess(z * 255.0))(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs), base


def read_image(path, label, img_size, augment_layer=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (img_size, img_size))
    if augment_layer is not None:
        img = augment_layer(img, training=True)
    return img, tf.one_hot(label, depth=len(CLASSES))


def make_ds(df, img_size, batch_size, augment=False, seed=42):
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"), layers.RandomRotation(0.05), layers.RandomZoom(0.08), layers.RandomContrast(0.10)
    ]) if augment else None
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values.astype(np.int32)))
    if augment:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: read_image(p, y, img_size, aug), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def predict_dataframe(model, df, img_size, batch_size):
    imgs = []
    for p in df["path"].values:
        img = tf.io.read_file(p)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (img_size, img_size))
        imgs.append(img.numpy())
    return model.predict(np.asarray(imgs, dtype=np.float32), batch_size=batch_size, verbose=0)


def save_predictions(df, probs, out_csv):
    pred = np.argmax(probs, axis=1)
    out = df[["path", "class", "label"]].copy()
    out["pred_label"] = pred
    out["pred_class"] = [CLASSES[i] for i in pred]
    for i, c in enumerate(CLASSES):
        out[f"prob_{c}"] = probs[:, i]
    out.to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=["DenseNet201", "EfficientNetB3", "MobileNetV2"])
    parser.add_argument("--losses", nargs="+", default=["ce", "focal"])
    parser.add_argument("--epochs_head", type=int, default=8)
    parser.add_argument("--epochs_fine", type=int, default=18)
    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    csv_dir = out / "csv"; csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    weight_dir = out / "weights"; weight_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(csv_dir / "train_split_v2.csv")
    val_df = pd.read_csv(csv_dir / "validation_split_v2.csv")
    test_df = pd.read_csv(csv_dir / "test_split_v2.csv")

    class_weights = {0: 0.6923076923076923, 1: 1.8, 2: 2.125, 3: 0.6538461538461539}
    alpha = np.array([class_weights[i] for i in range(4)], dtype=np.float32)
    alpha = alpha / alpha.sum()

    train_ds = make_ds(train_df, args.img_size, args.batch_size, augment=True, seed=args.seed)
    val_ds = make_ds(val_df, args.img_size, args.batch_size, augment=False, seed=args.seed)
    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    rows = []
    for model_name in args.models:
        for loss_name in args.losses:
            key = f"{model_name}_{loss_name}"
            print("Processing", key)
            ckpt = weight_dir / f"{key}_best.weights.h5"
            pred_csv = csv_dir / f"{key}_predictions.csv"
            if pred_csv.exists():
                print("Prediction file exists; skipping", key)
                continue
            model, base = build_model(model_name, img_size=args.img_size, weights="imagenet")
            loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.03) if loss_name == "ce" else focal_loss(alpha)
            model.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-4), loss=loss, metrics=["accuracy"])
            callbacks = [
                keras.callbacks.ModelCheckpoint(str(ckpt), save_best_only=True, save_weights_only=True, monitor="val_loss"),
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
            ]
            hist1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head, class_weight=class_weights if loss_name == "ce" else None, callbacks=callbacks)
            base.trainable = True
            for layer in base.layers[:int(len(base.layers)*0.60)]: layer.trainable = False
            for layer in base.layers:
                if isinstance(layer, layers.BatchNormalization): layer.trainable = False
            model.compile(optimizer=keras.optimizers.AdamW(1e-5, weight_decay=1e-4), loss=loss, metrics=["accuracy"])
            hist2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_fine, class_weight=class_weights if loss_name == "ce" else None, callbacks=callbacks)
            if ckpt.exists(): model.load_weights(str(ckpt))
            with open(csv_dir / f"{key}_history.json", "w") as f: json.dump({"head": hist1.history, "fine": hist2.history}, f, indent=2)
            val_probs = predict_dataframe(model, val_df, args.img_size, args.batch_size)
            test_probs = predict_dataframe(model, test_df, args.img_size, args.batch_size)
            save_predictions(test_df, test_probs, pred_csv)
            pd.DataFrame(classification_report(y_test, np.argmax(test_probs, axis=1), target_names=CLASSES, output_dict=True, zero_division=0)).T.to_csv(csv_dir / f"{key}_classification_report.csv")
            cm = confusion_matrix(y_test, np.argmax(test_probs, axis=1), labels=np.arange(4))
            pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(csv_dir / f"{key}_confusion_matrix.csv")
            row = {"model_key": key, "architecture": model_name, "loss": loss_name}
            row.update({f"val_{k}": v for k, v in metric_dict(y_val, val_probs).items()})
            row.update({f"test_{k}": v for k, v in metric_dict(y_test, test_probs).items()})
            rows.append(row)
            pd.DataFrame(rows).to_csv(csv_dir / "all_model_results_v2.csv", index=False)

if __name__ == "__main__":
    main()
