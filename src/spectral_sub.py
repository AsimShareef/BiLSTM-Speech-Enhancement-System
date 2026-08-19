import numpy as np

def estimate_noise(magnitude_spectrogram, speech_mask):
    """
    Estimates the noise profile by averaging the frequency bins 
    across all frames where speech is absent (noise-only frames).
    
    Args:
        magnitude_spectrogram: 2D array (frequency_bins, frames)
        speech_mask: 1D boolean array from VAD (True for speech, False for noise)
        
    Returns:
        noise_profile: 1D array representing the average noise magnitude per frequency bin
    """
    # Invert the mask to get noise-only frames
    noise_mask = ~speech_mask
    noise_frames = magnitude_spectrogram[:, noise_mask]
    
    # Fallback: if VAD fails and marks everything as speech, use the first 5 frames
    if noise_frames.shape[1] == 0:
        noise_profile = np.mean(magnitude_spectrogram[:, :5], axis=1)
    else:
        # Average the noise across time (frames) to get a stable noise spectrum
        noise_profile = np.mean(noise_frames, axis=1)
        
    # Reshape to (frequency_bins, 1) for broadcasting during subtraction
    return noise_profile.reshape(-1, 1)

def apply_subtraction(magnitude_spectrogram, noise_profile, alpha=2.0, beta=0.05):
    """
    Performs the spectral subtraction with over-subtraction and spectral flooring.
    
    Args:
        magnitude_spectrogram: 2D array of the noisy audio
        noise_profile: 1D array of the estimated noise
        alpha: Over-subtraction factor to aggressively remove noise (e.g., 1.5 to 4.0)
        beta: Spectral floor to prevent negative values and reduce musical noise
        
    Returns:
        enhanced_magnitude: 2D array of the cleaned audio magnitude
    """
    # Subtract the scaled noise profile from the noisy magnitude
    enhanced_magnitude = magnitude_spectrogram - (alpha * noise_profile)
    
    # Calculate the minimum allowed energy (spectral floor)
    spectral_floor = beta * noise_profile
    
    # Ensure no values drop below the spectral floor (prevents negative magnitudes)
    enhanced_magnitude = np.maximum(enhanced_magnitude, spectral_floor)
    
    return enhanced_magnitude