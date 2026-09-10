"""
Signal-processing smoke tests (no TensorFlow needed).

    python tests/test_dsp.py
"""

import os
import sys
import glob

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from manual_stft import manual_stft, manual_istft, magnitude_and_phase  # noqa: E402
from data_processing import (  # noqa: E402
    pre_emphasis, wav_to_stft, frame_spectrogram,
    stft_features_for_pair, n_stft_frames, n_windows,
)
from config import CLEAN_TRAIN_DIR, NOISY_TRAIN_DIR, TIME_STEPS  # noqa: E402


def _sample_files(k=6):
    clean = sorted(glob.glob(os.path.join(CLEAN_TRAIN_DIR, "*.wav")))
    noisy = sorted(glob.glob(os.path.join(NOISY_TRAIN_DIR, "*.wav")))
    common = sorted(set(map(os.path.basename, clean)) & set(map(os.path.basename, noisy)))
    step = max(1, len(common) // k)
    return [(os.path.join(CLEAN_TRAIN_DIR, n), os.path.join(NOISY_TRAIN_DIR, n))
            for n in common[::step][:k]]


def test_istft_roundtrip():
    """manual_istft(manual_stft(x)) ~= x in the valid (non-edge) region."""
    x, _ = sf.read(_sample_files(1)[0][0], dtype="float32")
    D = manual_stft(x)
    xr = manual_istft(D)
    n = min(len(x), len(xr))
    a, b = x[:n], xr[:n]
    edge = 512
    err = np.sqrt(np.mean((a[edge:-edge] - b[edge:-edge]) ** 2))
    rel = err / (np.sqrt(np.mean(a[edge:-edge] ** 2)) + 1e-9)
    print(f"  istft roundtrip  rel-RMSE = {rel:.4e}")
    assert rel < 1e-2, rel


def test_frame_count_matches_prediction():
    """dataset.py's analytic window count must match the real framed array."""
    for cpath, _ in _sample_files():
        x, _ = sf.read(cpath, dtype="float32")
        mag, _ = wav_to_stft(pre_emphasis(x))
        pred_frames = n_stft_frames(len(x))
        pred_wins = n_windows(len(x))
        real_wins = frame_spectrogram(mag).shape[0]
        assert mag.shape[0] == pred_frames, (mag.shape[0], pred_frames)
        assert real_wins == pred_wins, (real_wins, pred_wins)
    print(f"  frame/window arithmetic exact on {len(_sample_files())} files")


def test_psm_range_and_shape():
    for cpath, npath in _sample_files():
        c, _ = sf.read(cpath, dtype="float32")
        n, _ = sf.read(npath, dtype="float32")
        noisy_mag, mask = stft_features_for_pair(c, n)
        assert noisy_mag.shape == mask.shape
        assert noisy_mag.shape[1] == 257
        assert mask.min() >= 0.0 and mask.max() <= 1.0, (mask.min(), mask.max())
        assert np.isfinite(noisy_mag).all() and np.isfinite(mask).all()
    print(f"  PSM in [0,1], shapes OK on {len(_sample_files())} files")


def test_windows_have_context():
    c, n = _sample_files(1)[0]
    cw, _ = sf.read(c, dtype="float32")
    nw, _ = sf.read(n, dtype="float32")
    noisy_mag, mask = stft_features_for_pair(cw, nw)
    Xw = frame_spectrogram(noisy_mag)
    assert Xw.shape[1] == TIME_STEPS and Xw.shape[2] == 257
    print(f"  window tensor {Xw.shape}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            print(f"- {fn.__name__}")
            fn()
        except AssertionError as e:
            fails += 1
            print(f"  FAIL: {e}")
    print("\n" + ("ALL PASSED" if not fails else f"{fails} FAILED"))
    sys.exit(1 if fails else 0)
