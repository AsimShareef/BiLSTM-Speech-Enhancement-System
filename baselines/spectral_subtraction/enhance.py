"""
enhance.py - in-memory wrapper around the classical spectral-subtraction
baseline (this folder's original CS60116 code), so `src/evaluate.py` can run it
on VoiceBank-DEMAND without shelling out to temp files.

Algorithm (unchanged from the baseline):
  energy-based VAD -> noise profile from noise-only frames
  -> over-subtraction |Y| - alpha*|N|  with spectral floor beta*|N|
  -> ISTFT with the original noisy phase.

The three baseline leaf modules are loaded by path under private names, so
nothing is added to `sys.path` (avoids `metrics`/`utils`/`config` name clashes
with the main `src/` package).
"""

import importlib.util
import pathlib

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parent / "src"


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(f"_specsub_{mod_name}",
                                                  _SRC / f"{mod_name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_utils = _load("utils")
_vad = _load("vad")
_ss = _load("spectral_sub")

# Tuned on the baseline's own grid search (src/optimize_params.py).
DEFAULT_ALPHA = 2.5
DEFAULT_BETA = 0.02


def spectral_subtract(noisy: np.ndarray, sr: int = 16000,
                      alpha: float = DEFAULT_ALPHA,
                      beta: float = DEFAULT_BETA) -> np.ndarray:
    """noisy waveform -> enhanced waveform via classical spectral subtraction."""
    noisy = np.asarray(noisy, dtype=np.float32)
    magnitude, phase = _utils.get_stft(noisy)
    speech_mask = _vad.energy_vad(magnitude)
    noise_profile = _ss.estimate_noise(magnitude, speech_mask)
    enhanced_mag = _ss.apply_subtraction(magnitude, noise_profile,
                                         alpha=alpha, beta=beta)
    enhanced = _utils.get_istft(enhanced_mag, phase)
    peak = np.max(np.abs(enhanced))
    if peak > 1e-8:
        enhanced = enhanced / peak * 0.95
    return enhanced.astype(np.float32)
