import numpy as np

def energy_vad(magnitude_spectrogram, energy_threshold_ratio=1.5, initial_noise_frames=5):
    """
    A basic energy-based Voice Activity Detector operating on the spectrogram.
    
    Args:
        magnitude_spectrogram: 2D array (frequency_bins, frames)
        energy_threshold_ratio: Multiplier to set the speech threshold
        initial_noise_frames: Assumes the first N frames are pure noise
        
    Returns:
        speech_mask: 1D boolean array (True for speech, False for noise)
    """
    # Calculate the sum of energy across all frequencies for each frame
    frame_energies = np.sum(magnitude_spectrogram ** 2, axis=0)
    
    # Estimate the baseline noise energy from the first few frames
    if len(frame_energies) > initial_noise_frames:
        baseline_noise_energy = np.mean(frame_energies[:initial_noise_frames])
    else:
        baseline_noise_energy = np.mean(frame_energies)
        
    # Define the threshold for speech
    threshold = baseline_noise_energy * energy_threshold_ratio
    
    # Create a boolean mask: True if energy > threshold (Speech), else False (Noise)
    speech_mask = frame_energies > threshold
    
    return speech_mask