# BiLSTM Speech Enhancement with Phase-Sensitive Masking

Technical report. Course pipeline + streaming/evaluation rebuild.

- **Course project (ATSP, IIT Kharagpur):** Shaik Asim Shareef (23CS10065),
  Dasari Pardha Saradhi (23CS10016). Original report:
  [`ATSP_CourseProject_original.pdf`](ATSP_CourseProject_original.pdf).
- **Rebuild (this repo):** streaming data pipeline, full-corpus GPU training,
  reproducible benchmark vs. a classical baseline.

---

## 1. Problem

Single-channel additive-noise speech enhancement: recover clean speech `s[n]`
from `y[n] = s[n] + d[n]` with one microphone. We estimate a real-valued gain
**mask** in the STFT domain rather than regressing the waveform or the complex
spectrum directly - masks are bounded, stable to train, and keep the pipeline
interpretable in classical DSP terms.

## 2. Time-frequency front end (`src/manual_stft.py`)

A full-file FFT loses time localisation, so we use the STFT: 512-point FFT
(32 ms at 16 kHz), 128-sample hop (8 ms, 75 % overlap), Hann window. For a real
signal the spectrum is conjugate-symmetric, so we keep `512/2 + 1 = 257` bins
(DC ... Nyquist). ISTFT mirrors the negative frequencies back, does
inverse-FFT + windowed overlap-add, and divides by the window-sum for
amplitude-correct reconstruction. Analysis-only round-trip error is checked in
`tests/` against the input.

**Pre-emphasis** `y[n] = x[n] - 0.97 x[n-1]` is applied before the STFT to
flatten the ~ -6 dB/oct spectral tilt of voiced speech, so MSE training is not
dominated by low-frequency vowel energy and high-frequency consonant cues are
preserved. It is inverted (`de_emphasis`, an IIR) after the ISTFT.

## 3. Target: Phase-Sensitive Mask (`src/data_processing.py`)

The Ideal Ratio Mask `IRM = |S| / (|Y| + eps)` uses magnitude only. Since
reconstruction reuses the **noisy phase**, a bin whose phase noise has rotated
is not worth enhancing at full magnitude. The Phase-Sensitive Mask adds that
information:

```
PSM(k,m) = ( |S(k,m)| / |Y(k,m)| ) * cos( theta_S(k,m) - theta_Y(k,m) )
```

clipped to `[0, 1]`. When clean and noisy phase agree the cosine is 1 (plain
IRM); when the noise has rotated the phase ~90 deg the target collapses toward
0, teaching the network to suppress phase-corrupted bins instead of restoring a
magnitude that will be played back out of sync. Phase arrays are stored as unit
phasors `e^{j theta}`, so `cos(theta_S - theta_Y) = Re(phase_S * conj(phase_Y))`.

## 4. Model (`src/model.py`)

| Layer | Output | Notes |
|---|---|---|
| Input | (32, 257) | 32-frame context window of noisy magnitude |
| Bidirectional LSTM(128), return_sequences | (32, 256) | forward+backward context |
| Dropout 0.3 | | |
| Bidirectional LSTM(64), return_sequences | (32, 128) | |
| Dropout 0.3 | | |
| TimeDistributed(Dense(257, sigmoid)) | (32, 257) | per-frame mask in (0,1) |

Loss MSE, optimiser Adam (1e-3), ~0.6 M parameters. Bidirectional because this
is offline enhancement - the decay of a phoneme informs the mask at its onset.
`TimeDistributed` shares the 128->257 projection across all 32 steps so timing
is preserved.

## 5. Streaming dataset (`src/dataset.py`) - the rebuild

The course build called `np.concatenate` on every framed spectrogram: ~22 MB
RAM per 3 s file, so ~100 files of 11 572 before a 16 GB machine thrashed. The
`STFTSequence` (`tf.keras.utils.Sequence`) instead:

1. **Header pass** - `soundfile.info` gives sample counts with no decode;
   window counts are computed analytically from the STFT/framing arithmetic and
   flattened into a per-epoch index.
2. **File-major, shuffled** - file order is permuted each epoch, windows are
   drawn without replacement per file (capped at `MAX_WINDOWS_PER_FILE` so long
   utterances do not dominate), and each file is loaded, transformed, and freed
   exactly once per epoch.
3. **Config change** `FRAME_STEP` 1 -> 8: ~8x fewer windows and less temporal
   over-smoothing in overlap-add (the course "reverb" artefact), keeping 75 %
   window overlap.

RAM is flat at a few hundred MB for the full corpus. Train/val split is
**file-level** (`VAL_SPLIT = 0.10`) so windows from one utterance never straddle
the split.

## 6. Training (`src/train.py`)

Full 28-speaker trainset, batch 128, Adam 1e-3, `ReduceLROnPlateau`
(x0.5, patience 3), `EarlyStopping` (patience 6, restore best), `ModelCheckpoint`
on `val_loss`, CSV history. GPU via `notebooks/train_colab.ipynb`.

## 7. Evaluation (`src/evaluate.py`)

Official 824-file test set, three systems scored identically against the clean
reference:

- **Noisy** input (lower bound),
- **Spectral subtraction** - `baselines/spectral_subtraction/`, energy VAD +
  over-subtraction (`alpha = 2.5`) + spectral floor (`beta = 0.02`),
- **BiLSTM + PSM** - this model.

Metrics: **PESQ** (wideband, perceptual quality, -0.5..4.5), **STOI**
(intelligibility, 0..1), **SI-SNR** (scale-invariant, dB). Estimates are
delay-aligned to the reference by cross-correlation before scoring
(`src/metrics.py`). Outputs: `results/metrics.md`, `results/metrics_per_file.csv`,
and before/after clips + spectrograms in `results/samples/`.

## 8. Results

<!-- Paste results/metrics.md here after the Colab run. -->
_Pending full-corpus training run._

## 9. Objective vs. subjective behaviour

Observed on the course build and expected to persist: on **high-SNR / already
clean** input the mask still attenuates real speech (it "expects" noise), so
SI-SNR *gain* can go negative even when a listener hears no degradation - the
"do no harm" limitation. Reusing noisy phase also adds sample-level
misalignment that intrusive metrics penalise more than the ear does. This is
why the benchmark reports absolute PESQ/STOI/SI-SNR for every system rather than
only gains, and why per-file CSV is kept for stratified analysis by input SNR.

## 10. Limitations & future work

- Magnitude-only mask; no learned/complex phase.
- Single corpus (VoiceBank-DEMAND); no cross-noise / cross-language test.
- No SNR-adaptive bypass for near-clean input.
- Next: `tf.data` + tfrecord sharding for multi-GPU; complex ratio mask (cRM);
  PESQ/STOI-aware or SI-SNR loss instead of mask MSE.

## 11. Contributions

- **Course pipeline (both members):** custom STFT/ISTFT, BiLSTM mask model, PSM
  target, pre/de-emphasis, spectral floor, reconstruction, initial report.
- **This rebuild (Shaik Asim Shareef):** `STFTSequence` streaming loader,
  file-level split, full-corpus training script, `metrics.py`, `evaluate.py`
  benchmark harness + spectral-subtraction comparison, Colab notebook, repo
  restructure and documentation.
