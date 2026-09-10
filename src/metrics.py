"""
metrics.py - objective speech-enhancement metrics.

  si_snr      scale-invariant SNR in dB   (physical energy; -inf..+inf)
  stoi_score  Short-Time Objective Intelligibility  (0..1, higher better)
  pesq_score  Perceptual Evaluation of Speech Quality, wideband  (-0.5..4.5)

All are intrusive: they need the clean reference. `align` fixes the small
constant delay the STFT/framing introduces before scoring.
"""

import numpy as np
from scipy import signal as _sp

try:
    from pesq import pesq as _pesq
except Exception:  # noqa: BLE001 - optional; evaluate.py reports NaN if missing
    _pesq = None

try:
    from pystoi import stoi as _stoi
except Exception:  # noqa: BLE001
    _stoi = None


def align(ref: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Delay-align `test` to `ref` by cross-correlation, trim to equal length."""
    ref = np.asarray(ref, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    corr = _sp.correlate(ref, test, mode="full", method="fft")
    lags = _sp.correlation_lags(len(ref), len(test), mode="full")
    lag = int(lags[np.argmax(corr)])
    if lag > 0:
        test = np.pad(test, (lag, 0))
    elif lag < 0:
        test = test[-lag:]
    n = min(len(ref), len(test))
    return ref[:n], test[:n]


def si_snr(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target ** 2) + 1e-8) /
                               (np.sum(noise ** 2) + 1e-8)))


def stoi_score(ref: np.ndarray, est: np.ndarray, sr: int = 16000) -> float:
    if _stoi is None:
        return float("nan")
    try:
        return float(_stoi(ref, est, sr, extended=False))
    except Exception:  # noqa: BLE001
        return float("nan")


def pesq_score(ref: np.ndarray, est: np.ndarray, sr: int = 16000) -> float:
    if _pesq is None:
        return float("nan")
    try:
        return float(_pesq(sr, np.asarray(ref, np.float32),
                           np.asarray(est, np.float32), "wb"))
    except Exception:  # noqa: BLE001 - PESQ throws on near-silent / too-short
        return float("nan")
