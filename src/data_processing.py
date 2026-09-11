"""
data_processing.py - STFT features, framing, and mask (PSM / IRM) computation.

Pure NumPy/SciPy signal-processing front end. No TensorFlow here so it can be
unit-tested and reused by the streaming dataset (`dataset.py`) and by
`inference.py` / `evaluate.py`.
"""

import os
import glob
import numpy as np
import soundfile as sf
import librosa

from config import (
    SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH,
    TIME_STEPS, FRAME_STEP, IRM_EPSILON,
    PRE_EMPHASIS_COEF, MASK_TYPE,
)
from manual_stft import manual_stft, magnitude_and_phase


# == Pre-emphasis ============================================================
def pre_emphasis(signal: np.ndarray, coeff: float = PRE_EMPHASIS_COEF) -> np.ndarray:
    """First-order high-pass: y[n] = x[n] - coeff * x[n-1].

    Flattens the spectral tilt of speech so the network does not ignore the
    low-energy high-frequency consonants. Inverted after reconstruction by
    `reconstruction.de_emphasis`.
    """
    signal = np.asarray(signal, dtype=np.float32)
    return np.append(signal[0], signal[1:] - coeff * signal[:-1]).astype(np.float32)


# == STFT (custom, from manual_stft.py) ======================================
def wav_to_stft(waveform: np.ndarray):
    """waveform -> (magnitude [T, F], phase [T, F]) using the custom STFT.

    Phase is returned as unit complex phasors e^{j*theta} (time-major).
    """
    D = manual_stft(
        waveform.astype(np.float32),
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, center=True,
    )
    magnitude, phase = magnitude_and_phase(D)
    return magnitude.T.astype(np.float32), phase.T


def load_and_stft(filepath: str):
    """Load any audio file -> resample to 16 kHz mono -> pre-emphasis -> STFT."""
    try:
        waveform, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        waveform = pre_emphasis(waveform)
    except Exception as e:  # noqa: BLE001 - surface a helpful message
        raise RuntimeError(f"Failed to load '{filepath}': {e}") from e
    return wav_to_stft(waveform)


# == Sliding-window framing =================================================
def n_stft_frames(n_samples: int) -> int:
    """Frame count that `manual_stft(center=True)` yields for `n_samples`."""
    padded = n_samples + 2 * (N_FFT // 2)          # reflect pad both sides
    return (padded - N_FFT) // HOP_LENGTH + 1


def n_windows(n_samples: int,
              time_steps: int = TIME_STEPS,
              frame_step: int = FRAME_STEP) -> int:
    """Number of sliding windows `frame_spectrogram` yields for `n_samples`."""
    frames = n_stft_frames(n_samples)
    if frames < time_steps:
        return 0
    return (frames - time_steps) // frame_step + 1


def frame_spectrogram(spec: np.ndarray,
                      time_steps: int = TIME_STEPS,
                      frame_step: int = FRAME_STEP) -> np.ndarray:
    """[T, F] spectrogram -> [N, time_steps, F] overlapping windows."""
    T, F = spec.shape
    n_windows = (T - time_steps) // frame_step + 1
    if n_windows <= 0:
        raise ValueError(
            f"Signal too short: {T} STFT frames but TIME_STEPS={time_steps}."
        )
    frames = np.lib.stride_tricks.sliding_window_view(
        spec, window_shape=(time_steps, F)
    )[::frame_step, 0]
    return np.ascontiguousarray(frames, dtype=np.float32)


# == Masks =================================================================
def compute_irm(clean_mag: np.ndarray, noisy_mag: np.ndarray,
                epsilon: float = IRM_EPSILON) -> np.ndarray:
    """Ideal Ratio Mask: |S| / (|Y| + eps), clipped to [0, 1]."""
    irm = clean_mag / (noisy_mag + epsilon)
    return np.clip(irm, 0.0, 1.0).astype(np.float32)


def compute_psm(clean_mag: np.ndarray, clean_phase: np.ndarray,
                noisy_mag: np.ndarray, noisy_phase: np.ndarray,
                epsilon: float = IRM_EPSILON) -> np.ndarray:
    """Phase-Sensitive Mask: (|S| / |Y|) * cos(theta_S - theta_Y), clip [0, 1].

    `*_phase` are unit complex phasors, so cos(theta_S - theta_Y) is
    Re(clean_phase * conj(noisy_phase)).
    """
    phase_cos = np.real(clean_phase * np.conj(noisy_phase))
    psm = (clean_mag / (noisy_mag + epsilon)) * phase_cos
    return np.clip(psm, 0.0, 1.0).astype(np.float32)


def stft_features_for_pair(clean_wav: np.ndarray, noisy_wav: np.ndarray,
                           mask_type: str = MASK_TYPE):
    """One (clean, noisy) waveform pair -> (noisy_mag [T, F], mask [T, F]).

    This is the per-file work the streaming dataset does. Pre-emphasis is
    applied to both signals before the STFT.
    """
    clean_wav = pre_emphasis(clean_wav)
    noisy_wav = pre_emphasis(noisy_wav)

    clean_mag, clean_phase = wav_to_stft(clean_wav)
    noisy_mag, noisy_phase = wav_to_stft(noisy_wav)

    # Align frame counts (reflect padding can differ by 1 for odd lengths).
    T = min(clean_mag.shape[0], noisy_mag.shape[0])
    clean_mag, clean_phase = clean_mag[:T], clean_phase[:T]
    noisy_mag, noisy_phase = noisy_mag[:T], noisy_phase[:T]

    if mask_type == "irm":
        mask = compute_irm(clean_mag, noisy_mag)
    else:
        mask = compute_psm(clean_mag, clean_phase, noisy_mag, noisy_phase)
    return noisy_mag, mask


# == Small in-memory dataset (kept for quick experiments only) ================
def build_dataset(clean_wav: np.ndarray, noisy_wav: np.ndarray):
    """Single pair -> (X, y, noisy_mag, noisy_phase). Used by inference tests."""
    clean_wav = pre_emphasis(clean_wav)
    noisy_wav = pre_emphasis(noisy_wav)
    clean_mag, clean_phase = wav_to_stft(clean_wav)
    noisy_mag, noisy_phase = wav_to_stft(noisy_wav)
    T = min(clean_mag.shape[0], noisy_mag.shape[0])
    if MASK_TYPE == "irm":
        mask = compute_irm(clean_mag[:T], noisy_mag[:T])
    else:
        mask = compute_psm(clean_mag[:T], clean_phase[:T],
                           noisy_mag[:T], noisy_phase[:T])
    X = frame_spectrogram(noisy_mag[:T])
    y = frame_spectrogram(mask)
    return X, y, noisy_mag, noisy_phase


def build_dataset_from_directory(clean_dir: str, noisy_dir: str,
                                 max_files: int | None = None):
    """Eager in-memory loader. WARNING: RAM ~ 22 MB/file; use dataset.py for
    anything above a few hundred files."""
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    if not clean_files or not noisy_files:
        raise ValueError(f"No .wav files found in {clean_dir} or {noisy_dir}")

    clean_dict = {os.path.basename(f): f for f in clean_files}
    noisy_dict = {os.path.basename(f): f for f in noisy_files}
    common = sorted(set(clean_dict) & set(noisy_dict))
    if max_files:
        common = common[:max_files]

    X_all, y_all = [], []
    for name in common:
        c, _ = sf.read(clean_dict[name])
        n, _ = sf.read(noisy_dict[name])
        if c.ndim > 1:
            c = c.mean(axis=1)
        if n.ndim > 1:
            n = n.mean(axis=1)
        Xb, yb, _, _ = build_dataset(c, n)
        X_all.append(Xb)
        y_all.append(yb)

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    print(f"[Dataset] Final training shape: X={X.shape}, y={y.shape}")
    return X, y
