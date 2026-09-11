import numpy as np
from utils import load_audio, get_stft, get_istft
from vad import energy_vad
from spectral_sub import estimate_noise, apply_subtraction
from metrics import calculate_pesq, calculate_snr

def find_optimal_parameters(clean_path, noisy_path):
    """
    Runs a grid search over alpha and beta to find the combination
    that produces the highest PESQ score.
    """
    print("Loading audio and pre-calculating constants...")
    clean_signal, sr = load_audio(clean_path)
    noisy_signal, _ = load_audio(noisy_path)
    
    # Pre-calculate the STFT and noise profile once to save massive amounts of compute time
    magnitude, phase = get_stft(noisy_signal)
    speech_mask = energy_vad(magnitude)
    noise_profile = estimate_noise(magnitude, speech_mask)
    
    # Define the search space
    # Alpha: 1.0 (no over-subtraction) up to 5.0 (aggressive subtraction)
    alphas = np.arange(1.0, 5.5, 0.5) 
    # Beta: 0.001 (deep floor) up to 0.1 (shallow floor)
    betas = np.array([0.001, 0.01, 0.05, 0.1])
    
    best_pesq = -1
    best_params = (None, None)
    
    print(f"\n{'Alpha':<8} | {'Beta':<8} | {'PESQ':<8} | {'SNR (dB)':<8}")
    print("-" * 40)
    
    for alpha in alphas:
        for beta in betas:
            # Apply subtraction with current parameters
            enhanced_mag = apply_subtraction(magnitude, noise_profile, alpha, beta)
            
            # Reconstruct audio
            enhanced_signal = get_istft(enhanced_mag, phase)
            
            # Calculate metrics
            pesq_score = calculate_pesq(clean_signal, enhanced_signal, sr)
            snr_score = calculate_snr(clean_signal, enhanced_signal)
            
            # Handle potential PESQ calculation failures
            if pesq_score is None:
                continue
                
            print(f"{alpha:<8.1f} | {beta:<8.3f} | {pesq_score:<8.3f} | {snr_score:<8.3f}")
            
            # Track the best score
            if pesq_score > best_pesq:
                best_pesq = pesq_score
                best_params = (alpha, beta)
                
    print("-" * 40)
    print(f"Optimal Parameters Found -> Alpha: {best_params[0]}, Beta: {best_params[1]}")
    print(f"Max PESQ Score: {best_pesq:.3f}")

if __name__ == "__main__":
    # Point these to a matching pair of clean and noisy audio files
    CLEAN_FILE = "../data/clean_speech/sample_clean.wav"
    NOISY_FILE = "../data/noisy_speech/sample_noisy.wav"
    
    find_optimal_parameters(CLEAN_FILE, NOISY_FILE)