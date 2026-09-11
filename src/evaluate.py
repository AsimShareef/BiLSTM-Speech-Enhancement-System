"""
evaluate.py - score the BiLSTM against the noisy input and the classical
spectral-subtraction baseline on the full VoiceBank-DEMAND test set (824 files).

Writes:
  results/metrics.md            mean table (paste into the README / report)
  results/metrics_per_file.csv  per-utterance rows
  results/samples/              a few noisy / clean / bilstm / specsub clips
  results/samples/*.png         matching spectrograms

    python src/evaluate.py --model bilstm_enhancer.keras
    python src/evaluate.py --limit 100      # quick pass
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import csv
import glob
import sys

import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "baselines", "spectral_subtraction"))
from enhance import spectral_subtract              # noqa: E402

from config import CLEAN_TEST_DIR, NOISY_TEST_DIR, MODEL_PATH, SAMPLE_RATE  # noqa: E402
from inference import bilstm_enhance, get_model    # noqa: E402
from metrics import si_snr, stoi_score, pesq_score, align  # noqa: E402

METHODS = ("noisy", "specsub", "bilstm")
SAMPLE_NAMES = ("p232_005", "p232_052", "p257_074", "p257_166")


def _pairs(clean_dir, noisy_dir):
    clean = {os.path.basename(f): f for f in glob.glob(os.path.join(clean_dir, "*.wav"))}
    noisy = {os.path.basename(f): f for f in glob.glob(os.path.join(noisy_dir, "*.wav"))}
    return [(n, clean[n], noisy[n]) for n in sorted(set(clean) & set(noisy))]


def _spectrogram_png(path_out, wavs: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    fig, axes = plt.subplots(1, len(wavs), figsize=(4 * len(wavs), 3.2), sharey=True)
    for ax, (name, w) in zip(np.atleast_1d(axes), wavs.items()):
        S = librosa.amplitude_to_db(np.abs(librosa.stft(w, n_fft=512, hop_length=128)),
                                    ref=np.max)
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, len(w) / SAMPLE_RATE, 0, SAMPLE_RATE / 2000])
        ax.set_title(name)
        ax.set_xlabel("s")
    np.atleast_1d(axes)[0].set_ylabel("kHz")
    fig.tight_layout()
    fig.savefig(path_out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--clean", default=CLEAN_TEST_DIR)
    ap.add_argument("--noisy", default=NOISY_TEST_DIR)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=2.5)
    ap.add_argument("--beta", type=float, default=0.02)
    args = ap.parse_args()

    os.makedirs("results/samples", exist_ok=True)
    model = get_model(args.model)

    pairs = _pairs(args.clean, args.noisy)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"[Eval] {len(pairs)} test files  |  model: {args.model}")

    acc = {m: {"si_snr": [], "stoi": [], "pesq": []} for m in METHODS}
    rows = []

    for i, (name, cpath, npath) in enumerate(pairs, 1):
        clean, _ = librosa.load(cpath, sr=SAMPLE_RATE, mono=True)
        noisy, _ = librosa.load(npath, sr=SAMPLE_RATE, mono=True)

        est = {
            "noisy": noisy,
            "specsub": spectral_subtract(noisy, SAMPLE_RATE, args.alpha, args.beta),
            "bilstm": bilstm_enhance(npath, model=model),
        }

        row = {"file": name}
        for m in METHODS:
            r, e = align(clean, est[m])
            vals = {"si_snr": si_snr(r, e), "stoi": stoi_score(r, e),
                    "pesq": pesq_score(r, e)}
            for k, v in vals.items():
                if not np.isnan(v):
                    acc[m][k].append(v)
                row[f"{m}_{k}"] = round(v, 4)
        rows.append(row)

        stem = name.replace(".wav", "")
        if stem in SAMPLE_NAMES:
            for m, w in {"clean": clean, **est}.items():
                sf.write(f"results/samples/{stem}_{m}.wav", w, SAMPLE_RATE, subtype="PCM_16")
            _spectrogram_png(f"results/samples/{stem}.png",
                             {"noisy": noisy, "spec-sub": est["specsub"],
                              "bilstm": est["bilstm"], "clean": clean})

        if i % 50 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)}")

    # -- write per-file csv ----------------------------------------------------
    with open("results/metrics_per_file.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # -- mean table ---------------------------------------------------------
    def mean(m, k):
        xs = acc[m][k]
        return sum(xs) / len(xs) if xs else float("nan")

    lines = [
        "# VoiceBank-DEMAND test set - objective metrics",
        "",
        f"Files scored: {len(pairs)}   |   model: `{os.path.basename(args.model)}`",
        f"Spectral-subtraction baseline: alpha={args.alpha}, beta={args.beta}",
        "",
        "| Method | PESQ (wb) | STOI | SI-SNR (dB) |",
        "|---|---|---|---|",
    ]
    labels = {"noisy": "Noisy (unprocessed)",
              "specsub": "Spectral subtraction (classical baseline)",
              "bilstm": "BiLSTM + PSM (this work)"}
    for m in METHODS:
        lines.append(f"| {labels[m]} | {mean(m,'pesq'):.3f} | "
                     f"{mean(m,'stoi'):.4f} | {mean(m,'si_snr'):.2f} |")
    lines += ["", "Deltas vs. noisy input:", "",
              "| Method | dPESQ | dSTOI | dSI-SNR (dB) |", "|---|---|---|---|"]
    for m in ("specsub", "bilstm"):
        lines.append(f"| {labels[m]} | {mean(m,'pesq')-mean('noisy','pesq'):+.3f} | "
                     f"{mean(m,'stoi')-mean('noisy','stoi'):+.4f} | "
                     f"{mean(m,'si_snr')-mean('noisy','si_snr'):+.2f} |")
    report = "\n".join(lines) + "\n"

    with open("results/metrics.md", "w") as f:
        f.write(report)
    print("\n" + report)
    print("wrote results/metrics.md, results/metrics_per_file.csv, results/samples/")


if __name__ == "__main__":
    main()
