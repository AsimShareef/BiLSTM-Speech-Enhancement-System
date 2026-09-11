"""
reconstruction.py - predicted mask -> overlap-add unframe -> spectral floor
-> apply to noisy magnitude -> custom ISTFT (noisy phase) -> de-emphasis.
"""

import numpy as np
import soundfile as sf
import scipy.signal

from config import (
    SAMPLE_RATE, HOP_LENGTH, WIN_LENGTH, TIME_STEPS, FRAME_STEP,
    SPECTRAL_FLOOR, PRE_EMPHASIS_COEF,
)
from manual_stft import manual_istft, reconstruct_from_magnitude_and_phase

DE_EMPHASIS_COEF = PRE_EMPHASIS_COEF


def de_emphasis(signal: np.ndarray, coeff: float = DE_EMPHASIS_COEF) -> np.ndarray:
    """Inverse of the pre-emphasis FIR: IIR  x[n] = y[n] + coeff * x[n-1]."""
    return scipy.signal.lfilter([1.0], [1.0, -coeff], signal)


def unframe(frames: np.ndarray, T: int,
            time_steps: int = TIME_STEPS, frame_step: int = FRAME_STEP) -> np.ndarray:
    """[N, time_steps, F] overlapping windows -> [T, F] via averaging overlap-add."""
    N, W, F = frames.shape
    out   = np.zeros((T, F), dtype=np.float64)
    count = np.zeros((T, 1), dtype=np.float64)
    for i in range(N):
        s = i * frame_step
        e = s + time_steps
        if e > T:
            break
        out[s:e]   += frames[i]
        count[s:e] += 1.0
    return (out / np.maximum(count, 1.0)).astype(np.float32)


def reconstruct_waveform(mask_frames: np.ndarray, noisy_mag: np.ndarray,
                         noisy_phase: np.ndarray) -> np.ndarray:
    """Core reconstruction, returns the enhanced waveform (no file I/O)."""
    T = noisy_mag.shape[0]
    full_mask = unframe(mask_frames, T)

    # Spectral floor: never suppress a bin to absolute zero. This mirrors the
    # classical spectral-subtraction floor and prevents isolated-peak
    # ("musical noise") artefacts.
    floored_mask = np.maximum(full_mask, SPECTRAL_FLOOR)
    enhanced_mag = floored_mask * noisy_mag

    enhanced_stft = reconstruct_from_magnitude_and_phase(
        enhanced_mag.T.astype(np.float32), noisy_phase.T
    )
    waveform = manual_istft(
        enhanced_stft, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        center=True, normed=True,
    )
    waveform = de_emphasis(waveform)
    peak = np.max(np.abs(waveform))
    if peak > 1e-8:
        waveform = waveform / peak * 0.95
    return waveform.astype(np.float32)


def reconstruct_audio(mask_frames: np.ndarray, noisy_mag: np.ndarray,
                      noisy_phase: np.ndarray, output_path: str) -> np.ndarray:
    """`reconstruct_waveform` + write a 16-bit PCM .wav."""
    waveform = reconstruct_waveform(mask_frames, noisy_mag, noisy_phase)
    sf.write(output_path, waveform, SAMPLE_RATE, subtype="PCM_16")
    return waveform
