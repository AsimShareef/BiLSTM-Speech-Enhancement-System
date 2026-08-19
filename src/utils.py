import librosa
import soundfile as sf
import numpy as np

def load_audio(file_path, target_sr=16000):
    """
    Loads an audio file and standardizes the sample rate.
    16kHz is standard for speech processing tasks.
    """
    signal, sr = librosa.load(file_path, sr=target_sr)
    return signal, sr

def save_audio(file_path, signal, sr=16000):
    """
    Saves the processed numpy array back to a .wav file.
    """
    sf.write(file_path, signal, sr)

def get_stft(signal, n_fft=512, hop_length=256):
    """
    Calculates the Short-Time Fourier Transform.
    Returns the magnitude (for subtraction) and phase (for reconstruction).
    """
    stft_matrix = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    return magnitude, phase

def get_istft(magnitude, phase, hop_length=256):
    """
    Reconstructs the time-domain signal from magnitude and phase.
    """
    complex_stft = magnitude * np.exp(1j * phase)
    signal = librosa.istft(complex_stft, hop_length=hop_length)
    return signal