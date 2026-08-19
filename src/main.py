import os
from utils import load_audio, save_audio, get_stft, get_istft
from vad import energy_vad
from spectral_sub import estimate_noise, apply_subtraction

def enhance_speech(input_path, output_path, alpha=2.0, beta=0.05):
    """
    Executes the full spectral subtraction pipeline on a single audio file.
    """
    print(f"Processing: {input_path}")
    
    # 1. Load the noisy audio signal
    signal, sr = load_audio(input_path)
    
    # 2. Convert time-domain signal to frequency domain (STFT)
    # This gives us the magnitude (amplitude of frequencies) and phase (timing)
    magnitude, phase = get_stft(signal)
    
    # 3. Perform Voice Activity Detection (VAD)
    # Find which frames contain speech and which are just background noise
    speech_mask = energy_vad(magnitude)
    
    # 4. Estimate the noise profile
    # Average the frequency bins across the noise-only frames
    noise_profile = estimate_noise(magnitude, speech_mask)
    
    # 5. Apply Spectral Subtraction
    # Subtract the noise profile from the noisy magnitude, applying over-subtraction
    # and spectral flooring to reduce musical noise artifacts.
    enhanced_magnitude = apply_subtraction(magnitude, noise_profile, alpha=alpha, beta=beta)
    
    # 6. Reconstruct the time-domain signal (ISTFT)
    # Combine the new, clean magnitude with the original noisy phase
    enhanced_signal = get_istft(enhanced_magnitude, phase)
    
    # 7. Save the final enhanced audio
    save_audio(output_path, enhanced_signal, sr)
    print(f"Successfully saved enhanced audio to: {output_path}")

if __name__ == "__main__":
    # Define input and output paths relative to the src/ directory
    # (Assuming the project structure we outlined earlier)
    INPUT_FILE = "../data/noisy_speech/sample_noisy.wav"
    OUTPUT_FILE = "../results/enhanced_audio/sample_enhanced.wav"
    
    # Ensure the output directory exists before trying to save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Run the enhancement pipeline
    # You will spend most of your time tweaking alpha and beta to get the best results!
    enhance_speech(INPUT_FILE, OUTPUT_FILE, alpha=2.5, beta=0.02)