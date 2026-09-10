"""
manual_stft.py — Custom STFT/ISTFT implementations from scratch.
"""

import numpy as np
from scipy.fftpack import fft, ifft
from config import N_FFT, HOP_LENGTH, WIN_LENGTH, N_FREQ_BINS


def create_hann_window(n_fft: int, win_length: int) -> np.ndarray:
    """
    Create a Hann window (Hanning window).

    The Hann window is defined as:
        w[n] = 0.5 * (1 - cos(2π·n / (N-1)))    for n = 0, 1, ..., N-1

    For STFT/ISTFT with 50% overlap and proper normalization,
    adjacent Hann windows sum to 1.0 (perfect reconstruction property).
    """
    # Create full-length Hann window
    window = np.hanning(win_length).astype(np.float32)

    # If win_length < n_fft, zero-pad to match FFT size
    if win_length < n_fft:
        padded_window = np.zeros(n_fft, dtype=np.float32)
        start = (n_fft - win_length) // 2
        padded_window[start:start + win_length] = window
        window = padded_window

    return window


def manual_stft(x: np.ndarray,
                n_fft: int = N_FFT,
                hop_length: int = HOP_LENGTH,
                win_length: int = WIN_LENGTH,
                center: bool = True) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform manually.

    Parameters
    ----------
    x : np.ndarray [n_samples]
        Input waveform
    n_fft : int
        FFT size (typically power of 2)
    hop_length : int
        Number of samples between consecutive frames
    win_length : int
        Window length (typically <= n_fft)
    center : bool
        If True, pad signal at start/end for symmetric framing

    Returns
    -------
    D : np.ndarray [n_fft//2 + 1, n_frames] (frequency-major, librosa convention)
        Complex STFT coefficients
        Each column is one time frame's FFT
    """
    x = x.astype(np.float32)

    # ── Center padding ─────────────────────────────────────────────────────────
    if center:
        # Pad n_fft//2 samples on each side for symmetric analysis
        pad_amount = n_fft // 2
        x = np.pad(x, (pad_amount, pad_amount), mode='reflect')

    # ── Prepare window ─────────────────────────────────────────────────────────
    window = create_hann_window(n_fft, win_length)

    # ── Framing ────────────────────────────────────────────────────────────────
    # Extract overlapping frames from the padded signal
    n_samples = len(x)
    n_frames = (n_samples - n_fft) // hop_length + 1

    if n_frames <= 0:
        raise ValueError(
            f"Signal too short: {len(x)} samples with n_fft={n_fft}, "
            f"hop_length={hop_length}. Cannot create any frames."
        )

    # ── Compute STFT via FFT of each frame ─────────────────────────────────────
    D = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)

    for m in range(n_frames):
        # Extract frame [n_fft samples starting at m*hop_length]
        start = m * hop_length
        frame = x[start:start + n_fft]

        # Apply window
        windowed_frame = frame * window

        # Compute FFT
        fft_result = fft(windowed_frame, n=n_fft)

        # Keep only positive frequency bins (0 to n_fft//2)
        D[:, m] = fft_result[:n_fft // 2 + 1]

    return D


def manual_istft(D: np.ndarray,
                 hop_length: int = HOP_LENGTH,
                 win_length: int = WIN_LENGTH,
                 center: bool = True,
                 normed: bool = True) -> np.ndarray:
    """
    Compute Inverse Short-Time Fourier Transform manually.

    Parameters
    ----------
    D : np.ndarray [n_fft//2 + 1, n_frames]
        Complex STFT coefficients (frequency-major)
    hop_length : int
        Hop length used in forward STFT
    win_length : int
        Window length used in forward STFT
    center : bool
        If True, was center-padded in forward STFT; remove padding from output
    normed : bool
        If True, normalize output by sum-of-windows (overlap-add normalization)

    Returns
    -------
    x : np.ndarray [n_samples]
        Reconstructed time-domain waveform
    """
    # ── Reconstruct FFT size from STFT shape ───────────────────────────────────
    n_freq_bins = D.shape[0]
    n_fft = 2 * (n_freq_bins - 1)
    n_frames = D.shape[1]

    # ── Prepare window ─────────────────────────────────────────────────────────
    window = create_hann_window(n_fft, win_length)

    # ── Length of output signal (before center-padding removal) ────────────────
    n_samples_padded = (n_frames - 1) * hop_length + n_fft

    # ── Inverse FFT + overlap-add ──────────────────────────────────────────────
    x_reconstructed = np.zeros(n_samples_padded, dtype=np.float32)
    window_sum = np.zeros(n_samples_padded, dtype=np.float32)

    for m in range(n_frames):
        # Reconstruct full FFT from positive frequencies
        # For real signal, negative frequencies are complex conjugate of positive
        fft_full = np.zeros(n_fft, dtype=np.complex64)
        fft_full[:n_freq_bins] = D[:, m]

        # Mirror negative frequencies (conjugate symmetry for real signals)
        # DC and Nyquist bins don't have mirrors
        if n_fft > 1:
            # For n_fft = 512: freq_bins = 257
            # Positive freqs: 0 to 256
            # Negative freqs (mirrored): 511 down to 1
            for k in range(1, n_freq_bins):
                fft_full[n_fft - k] = np.conj(fft_full[k])

        # Inverse FFT
        frame = np.real(ifft(fft_full, n=n_fft)).astype(np.float32)

        # Apply synthesis window (same window as analysis -> w^2 weighting)
        windowed_frame = frame * window

        # Overlap-add
        start = m * hop_length
        x_reconstructed[start:start + n_fft] += windowed_frame
        # Accumulate SQUARED window: the signal is weighted by w in both
        # analysis and synthesis, so perfect reconstruction needs division by
        # sum(w^2), not sum(w). (The original build divided by sum(w), which
        # scaled the output by mean(w^2)/mean(w) ~= 0.75 for a Hann window.)
        window_sum[start:start + n_fft] += window ** 2

    # ── Normalize by sum of squared windows (perfect reconstruction) ──────────
    if normed:
        # Guard against division by zero
        window_sum = np.maximum(window_sum, 1e-8)
        x_reconstructed /= window_sum

    # ── Remove center padding ──────────────────────────────────────────────────
    if center:
        pad_amount = n_fft // 2
        x_reconstructed = x_reconstructed[pad_amount:-pad_amount]

    return x_reconstructed


def magnitude_and_phase(D: np.ndarray) -> tuple:
    """
    Extract magnitude and phase from complex STFT.

    Parameters
    ----------
    D : np.ndarray [n_freq_bins, n_frames] complex
        Complex STFT coefficients

    Returns
    -------
    magnitude : np.ndarray [n_freq_bins, n_frames]
        Magnitude (absolute value)
    phase : np.ndarray [n_freq_bins, n_frames] complex
        Phase as unit complex phasors: e^{jθ}
    """
    magnitude = np.abs(D).astype(np.float32)

    # Extract phase as unit magnitude complex numbers
    # This preserves phase information while normalizing magnitude
    phase = np.where(
        magnitude > 1e-8,
        D / (magnitude + 1e-10),  # Normalize to unit magnitude
        1.0 + 0.0j  # Default for silence bins
    ).astype(np.complex64)

    return magnitude, phase


def reconstruct_from_magnitude_and_phase(magnitude: np.ndarray,
                                         phase: np.ndarray) -> np.ndarray:
    """
    Reconstruct complex STFT from magnitude and phase.

    Parameters
    ----------
    magnitude : np.ndarray [n_freq_bins, n_frames]
    phase : np.ndarray [n_freq_bins, n_frames] complex unit phasors

    Returns
    -------
    D : np.ndarray [n_freq_bins, n_frames] complex
        Reconstructed complex STFT
    """
    return (magnitude * phase).astype(np.complex64)
