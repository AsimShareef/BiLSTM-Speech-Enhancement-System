"""
inference.py - enhance one noisy .wav with the trained BiLSTM and (optionally)
print SI-SNR / STOI / PESQ against a clean reference.

    python src/inference.py --noisy noisy.wav --clean clean.wav
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import numpy as np
import tensorflow as tf
import librosa

from data_processing import load_and_stft, frame_spectrogram
from reconstruction import reconstruct_waveform
from metrics import si_snr, stoi_score, pesq_score, align
from config import BATCH_SIZE, MODEL_PATH, ENHANCED_WAV, SAMPLE_RATE

_MODEL_CACHE = {}


def get_model(model_path: str = MODEL_PATH):
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = tf.keras.models.load_model(model_path, compile=False)
    return _MODEL_CACHE[model_path]


def bilstm_enhance(noisy_path: str, model=None, model_path: str = MODEL_PATH) -> np.ndarray:
    """noisy .wav path -> enhanced waveform (float32, 16 kHz)."""
    model = model or get_model(model_path)
    noisy_mag, noisy_phase = load_and_stft(noisy_path)
    X = frame_spectrogram(noisy_mag)
    mask_frames = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    return reconstruct_waveform(mask_frames, noisy_mag, noisy_phase)


def main():
    ap = argparse.ArgumentParser(description="BiLSTM speech enhancement - inference")
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--clean", default=None, help="clean reference for metrics")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--output", default=ENHANCED_WAV)
    args = ap.parse_args()

    print(f"[Inference] enhancing {args.noisy}")
    enhanced = bilstm_enhance(args.noisy, model_path=args.model)

    import soundfile as sf
    sf.write(args.output, enhanced, SAMPLE_RATE, subtype="PCM_16")
    print(f"[Inference] wrote {args.output}")

    if args.clean:
        clean, _ = librosa.load(args.clean, sr=SAMPLE_RATE, mono=True)
        noisy, _ = librosa.load(args.noisy, sr=SAMPLE_RATE, mono=True)
        c_n, n_a = align(clean, noisy)
        c_e, e_a = align(clean, enhanced)
        print("\n" + "=" * 52)
        print("  METRIC     |   NOISY   | ENHANCED  |   GAIN")
        print("-" * 52)
        for name, fn, fmt in (
            ("SI-SNR dB", si_snr, "{:>8.2f}"),
            ("STOI     ", stoi_score, "{:>8.4f}"),
            ("PESQ     ", pesq_score, "{:>8.3f}"),
        ):
            i, f = fn(c_n, n_a), fn(c_e, e_a)
            print(f"  {name} | {fmt.format(i)}  | {fmt.format(f)}  | {f - i:>+8.3f}")
        print("=" * 52)


if __name__ == "__main__":
    main()
