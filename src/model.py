
# ── Suppress TF startup noise ─────────────────────────────────────────────────
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide INFO / WARNING / profiler msgs

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

from config import (
    TIME_STEPS, N_FREQ_BINS,
    LSTM_UNITS_1, LSTM_UNITS_2,
    DROPOUT_RATE, LEARNING_RATE,
)


def build_bilstm_model(time_steps:  int = TIME_STEPS,
                       n_freq_bins: int = N_FREQ_BINS) -> Model:
    """
    Build and compile the BiLSTM speech enhancement model.

    Parameters
    ----------
    time_steps  : context window length (sequence length for LSTM)
    n_freq_bins : STFT frequency bins = features per time step

    Returns
    -------
    model : compiled tf.keras.Model
    """

    # ── Input ──────────────────────────────────────────────────────────────────
    inputs = layers.Input(
        shape=(time_steps, n_freq_bins),
        name="noisy_mag"
    )

    # ── BiLSTM block 1 ─────────────────────────────────────────────────────────
    # merge_mode="concat" → output size = LSTM_UNITS_1 * 2
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS_1, return_sequences=True),
        merge_mode="concat",
        name="bilstm_1",
    )(inputs)

    x = layers.Dropout(DROPOUT_RATE, name="drop_1")(x)

    # ── BiLSTM block 2 ─────────────────────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS_2, return_sequences=True),
        merge_mode="concat",
        name="bilstm_2",
    )(x)

    x = layers.Dropout(DROPOUT_RATE, name="drop_2")(x)

    # ── Output: soft mask ──────────────────────────────────────────────────────
    # TimeDistributed applies Dense to every time step independently.
    # Sigmoid → values in (0, 1), perfectly matching IRM range.
    outputs = layers.TimeDistributed(
        layers.Dense(n_freq_bins, activation="sigmoid"),
        name="soft_mask",
    )(x)

    # ── Compile ────────────────────────────────────────────────────────────────
    model = Model(inputs, outputs, name="BiLSTM_Enhancer")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def summarise(model: Model) -> None:
    model.summary(line_length=72)
    n = model.count_params()
    print(f"\n  Total params  : {n:,}")
    print(f"  Approx size   : {n * 4 / 1e6:.2f} MB  (float32)\n")