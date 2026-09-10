"""
train.py - Train the BiLSTM mask estimator on the full VoiceBank-DEMAND corpus
using the streaming `STFTSequence` (flat RAM, all 11 572 files).

    python src/train.py                          # uses paths from config.py
    python src/train.py --epochs 25 --batch 128
    python src/train.py --max-files 500          # quick sanity run
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

from config import (
    CLEAN_TRAIN_DIR, NOISY_TRAIN_DIR, MODEL_PATH,
    BATCH_SIZE, EPOCHS, SEED,
)
from dataset import split_pairs, STFTSequence
from model import build_bilstm_model, summarise

np.random.seed(SEED)
tf.random.set_seed(SEED)


def device_info():
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(f"[System] {'GPU: ' + str([g.name for g in gpus]) if gpus else 'CPU only'}")


def main():
    ap = argparse.ArgumentParser(description="Train BiLSTM speech enhancer.")
    ap.add_argument("--clean", default=CLEAN_TRAIN_DIR)
    ap.add_argument("--noisy", default=NOISY_TRAIN_DIR)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-files", type=int, default=None,
                    help="cap training files (debug / smoke runs)")
    ap.add_argument("--out", default=MODEL_PATH)
    args = ap.parse_args()

    banner = "  BiLSTM Speech Enhancement - streaming training  "
    print("\n" + "=" * len(banner) + f"\n{banner}\n" + "=" * len(banner))
    device_info()

    train_pairs, val_pairs = split_pairs(args.clean, args.noisy)
    if args.max_files:
        train_pairs = train_pairs[:args.max_files]
        val_pairs = val_pairs[:max(1, args.max_files // 10)]

    train_seq = STFTSequence(train_pairs, batch_size=args.batch, shuffle=True)
    val_seq   = STFTSequence(val_pairs,   batch_size=args.batch, shuffle=False)
    print(f"[Data] train: {len(train_seq.pairs)} files / {train_seq.total_windows:,} windows / {len(train_seq)} batches")
    print(f"[Data] val  : {len(val_seq.pairs)} files / {val_seq.total_windows:,} windows / {len(val_seq)} batches")

    model = build_bilstm_model()
    summarise(model)

    os.makedirs("checkpoints", exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "checkpoints/best.keras", monitor="val_loss",
            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger("checkpoints/history.csv"),
    ]

    history = model.fit(
        train_seq, validation_data=val_seq,
        epochs=args.epochs, callbacks=callbacks, verbose=1,
    )

    model.save(args.out)
    with open("checkpoints/history.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

    best = min(history.history["val_loss"])
    print("\n" + "-" * 52)
    print(f"  saved model    -> {args.out}")
    print(f"  best val_loss  -> {best:.6f}")
    print(f"  history        -> checkpoints/history.csv")
    print("-" * 52)
    print("  next: python src/evaluate.py --model", args.out)


if __name__ == "__main__":
    main()
