"""
config.py - Central configuration. Single source of truth for all parameters.
"""

# == Audio ====================================================================
SAMPLE_RATE = 16_000        # Hz (standard for wideband speech / VoiceBank-DEMAND)

# == STFT ====================================================================
# n_fft = 512  -> 257 unique frequency bins (n_fft/2 + 1)
# hop   = 128  -> 8 ms frame shift  -> ~375 frames for a 3 s clip at 16 kHz
N_FFT       = 512
HOP_LENGTH  = 128
WIN_LENGTH  = 512
N_FREQ_BINS = N_FFT // 2 + 1        # = 257

# == Framing / LSTM input shape =============================================
# BiLSTM expects [batch, TIME_STEPS, N_FREQ_BINS]
TIME_STEPS = 32     # consecutive STFT frames per window (~256 ms context)

# FRAME_STEP was 1 in the original course build. A stride of 1 produces ~344
# highly-redundant windows per 3 s file: it made full-dataset training
# intractable and over-smoothed the mask in the overlap-add step (the "reverb"
# artefact noted in the report). A stride of 8 keeps 75 % window overlap,
# cuts the window count ~8x, and sharpens transients.
FRAME_STEP = 8

# Cap windows contributed per utterance so a few long files cannot dominate
# an epoch. None = use every window.
MAX_WINDOWS_PER_FILE = 96

# == Model ==================================================================
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DROPOUT_RATE = 0.3

# == Training ===============================================================
BATCH_SIZE    = 128
EPOCHS        = 40
LEARNING_RATE = 1e-3
VAL_SPLIT     = 0.10        # held out at the *file* level (no window leakage)
SEED          = 42

# == Mask / target =========================================================
IRM_EPSILON   = 1e-8        # prevents division by zero in mask computation
MASK_TYPE     = "psm"       # "psm" (phase-sensitive) or "irm"
SPECTRAL_FLOOR = 0.05       # min mask value at reconstruction (anti musical-noise)
PRE_EMPHASIS_COEF = 0.97

# == Paths =================================================================
CLEAN_TRAIN_DIR = "speech/clean_trainset_wav"
NOISY_TRAIN_DIR = "speech/noisy_trainset_wav"
CLEAN_TEST_DIR  = "speech/clean_testset_wav"
NOISY_TEST_DIR  = "speech/noisy_testset_wav"

ENHANCED_WAV = "enhanced_output.wav"
MODEL_PATH   = "bilstm_enhancer.keras"
