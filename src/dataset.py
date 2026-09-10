"""
dataset.py - Streaming, flat-RAM training data for the BiLSTM.

The original course build loaded every framed spectrogram into one NumPy array
via `np.concatenate`, which cost ~22 MB of RAM per 3 s file and capped training
at ~100 files (of 11 572). This `tf.keras.utils.Sequence` streams instead:

  * on construction it reads only file *headers* (`soundfile.info`) to count
    windows and build a flat index - no audio is decoded;
  * each epoch it shuffles the file order, and per file loads the pair once,
    runs pre-emphasis + custom STFT + mask, slices that file's windows, and
    frees the audio before moving on.

RAM stays flat at a few hundred MB regardless of corpus size.
"""

from __future__ import annotations

import os
import glob
import math

import numpy as np
import soundfile as sf
import tensorflow as tf

from config import (
    TIME_STEPS, FRAME_STEP, BATCH_SIZE,
    MAX_WINDOWS_PER_FILE, MASK_TYPE, VAL_SPLIT, SEED,
)
from data_processing import stft_features_for_pair, n_windows as _n_windows


def _pair_files(clean_dir: str, noisy_dir: str) -> list[tuple[str, str]]:
    clean = {os.path.basename(f): f for f in glob.glob(os.path.join(clean_dir, "*.wav"))}
    noisy = {os.path.basename(f): f for f in glob.glob(os.path.join(noisy_dir, "*.wav"))}
    common = sorted(set(clean) & set(noisy))
    if not common:
        raise ValueError(f"No matching .wav pairs in {clean_dir} / {noisy_dir}")
    return [(clean[n], noisy[n]) for n in common]


def split_pairs(clean_dir: str, noisy_dir: str,
                val_split: float = VAL_SPLIT, seed: int = SEED):
    """Deterministic file-level train/val split (no window leakage across sets)."""
    pairs = _pair_files(clean_dir, noisy_dir)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_val = int(round(len(pairs) * val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx]


class STFTSequence(tf.keras.utils.Sequence):
    """Yields (X, y) batches of shape (batch, TIME_STEPS, N_FREQ_BINS).

    X = noisy magnitude windows, y = PSM (or IRM) windows.
    """

    def __init__(self, pairs: list[tuple[str, str]], *,
                 batch_size: int = BATCH_SIZE,
                 shuffle: bool = True,
                 max_windows_per_file: int | None = MAX_WINDOWS_PER_FILE,
                 mask_type: str = MASK_TYPE,
                 seed: int = SEED):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_wpf = max_windows_per_file
        self.mask_type = mask_type
        self.rng = np.random.default_rng(seed)

        # Header-only pass: keep files that yield at least one window.
        # full_win_counts = every window in the file; win_counts = per-epoch
        # quota after the MAX_WINDOWS_PER_FILE cap.
        self.pairs, self.full_win_counts, self.win_counts = [], [], []
        for c, n in pairs:
            try:
                w = _n_windows(sf.info(c).frames)
            except Exception:  # noqa: BLE001 - skip unreadable file
                continue
            if w <= 0:
                continue
            self.pairs.append((c, n))
            self.full_win_counts.append(w)
            self.win_counts.append(min(w, self.max_wpf) if self.max_wpf else w)

        if not self.pairs:
            raise ValueError("No usable file pairs (all shorter than one window).")

        self.total_windows = int(sum(self.win_counts))
        self._build_epoch_index()

    # -- epoch bookkeeping --------------------------------------------------
    def _build_epoch_index(self):
        order = np.arange(len(self.pairs))
        if self.shuffle:
            self.rng.shuffle(order)
        # Flat list of (file_id, window_id), file-major so each file loads once.
        flat = []
        for fid in order:
            full = self.full_win_counts[fid]
            quota = self.win_counts[fid]
            if self.shuffle:
                wids = self.rng.choice(full, size=quota, replace=False)
            else:
                wids = np.arange(quota)
            flat.extend((int(fid), int(w)) for w in wids)
        self.index = flat

    def __len__(self) -> int:
        return math.ceil(self.total_windows / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            self._build_epoch_index()

    # -- per-file cache (last file only; index is file-major) --------------
    def _windows_for_file(self, fid: int):
        cache = getattr(self, "_cache", None)
        if cache is not None and cache[0] == fid:
            return cache[1], cache[2]
        c_path, n_path = self.pairs[fid]
        c, _ = sf.read(c_path, dtype="float32")
        n, _ = sf.read(n_path, dtype="float32")
        if c.ndim > 1:
            c = c.mean(axis=1)
        if n.ndim > 1:
            n = n.mean(axis=1)
        noisy_mag, mask = stft_features_for_pair(c, n, self.mask_type)

        T = noisy_mag.shape[0]
        starts = np.arange(0, T - TIME_STEPS + 1, FRAME_STEP)
        Xw = np.stack([noisy_mag[s:s + TIME_STEPS] for s in starts]).astype(np.float32)
        yw = np.stack([mask[s:s + TIME_STEPS] for s in starts]).astype(np.float32)
        self._cache = (fid, Xw, yw)
        return Xw, yw

    def __getitem__(self, batch_idx: int):
        lo = batch_idx * self.batch_size
        hi = min(lo + self.batch_size, len(self.index))
        entries = self.index[lo:hi]

        Xb, yb = [], []
        for fid, wid in entries:
            Xw, yw = self._windows_for_file(fid)
            if wid >= len(Xw):                     # capped file -> wrap safely
                wid = wid % len(Xw)
            Xb.append(Xw[wid])
            yb.append(yw[wid])
        return np.stack(Xb).astype(np.float32), np.stack(yb).astype(np.float32)


if __name__ == "__main__":
    # Smoke test: report window counts and time one batch.
    import time
    from config import CLEAN_TRAIN_DIR, NOISY_TRAIN_DIR

    tr, va = split_pairs(CLEAN_TRAIN_DIR, NOISY_TRAIN_DIR)
    seq = STFTSequence(tr, batch_size=128)
    print(f"train files      : {len(seq.pairs)}")
    print(f"total windows    : {seq.total_windows:,}")
    print(f"batches / epoch  : {len(seq)}")
    t0 = time.time()
    X, y = seq[0]
    print(f"first batch      : X={X.shape} y={y.shape} in {time.time() - t0:.2f}s")
