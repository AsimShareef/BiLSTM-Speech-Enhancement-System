import numpy as np
# Note: You will need to pip install the 'pesq' package for this to work
from pesq import pesq

def calculate_snr(clean_signal, enhanced_signal):
    """
    Calculates the Signal-to-Noise Ratio (SNR) in decibels (dB).
    Higher is better.
    
    Args:
        clean_signal: 1D array of the original, uncorrupted audio
        enhanced_signal: 1D array of the processed audio
        
    Returns:
        snr_db: Float representing the SNR in dB
    """
    # Ensure both signals are the exact same length before mathematical operations
    min_length = min(len(clean_signal), len(enhanced_signal))
    clean = clean_signal[:min_length]
    enhanced = enhanced_signal[:min_length]
    
    # The noise remaining in our enhanced signal is the difference 
    # between the pristine clean signal and our enhanced output.
    residual_noise = clean - enhanced
    
    # Calculate signal power and noise power
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum(residual_noise ** 2)
    
    # Handle the edge case of perfect reconstruction (division by zero)
    if noise_power == 0:
        return float('inf')
        
    # Calculate SNR in dB: 10 * log10(Signal Power / Noise Power)
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

def calculate_pesq(clean_signal, enhanced_signal, sr=16000):
    """
    Calculates the PESQ score. This evaluates how good the speech sounds 
    to a human ear, not just raw mathematical energy.
    Scores range from -0.5 to 4.5. Higher is better.
    
    Args:
        clean_signal: 1D array of the original, uncorrupted audio
        enhanced_signal: 1D array of the processed audio
        sr: Sample rate (must be 8000 or 16000 for standard PESQ)
        
    Returns:
        pesq_score: Float representing the perceptual quality
    """
    # PESQ strictly requires 16kHz or 8kHz sample rates
    if sr not in [8000, 16000]:
        raise ValueError("PESQ requires a sample rate of 8000Hz or 16000Hz.")
        
    min_length = min(len(clean_signal), len(enhanced_signal))
    clean = clean_signal[:min_length]
    enhanced = enhanced_signal[:min_length]
    
    # The 'wb' flag stands for wideband (used for 16kHz audio)
    # Use 'nb' (narrowband) if your audio is 8kHz
    mode = 'wb' if sr == 16000 else 'nb'
    
    try:
        score = pesq(sr, clean, enhanced, mode)
        return score
    except Exception as e:
        print(f"PESQ Calculation Error: {e}")
        return None