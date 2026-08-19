# Speech Enhancement: Spectral Subtraction

## Overview
This repository contains a Python implementation of a Speech Enhancement system using the Spectral Subtraction method. It is designed to estimate and suppress background noise from corrupted audio signals while minimizing musical noise artifacts through over-subtraction ($\alpha$) and spectral flooring ($\beta$).

This project was developed for the Advanced Topics in Speech Processing (CS60116/IT60116) course at IIT Kharagpur.

## Project Structure
* `src/`: Contains all source code for the enhancement pipeline.
  * `main.py`: Entry point for processing a single audio file.
  * `spectral_sub.py`: Core logic for noise estimation and spectral subtraction.
  * `vad.py`: Energy-based Voice Activity Detection (VAD).
  * `utils.py`: Audio I/O and STFT/ISTFT operations.
  * `metrics.py`: SNR and PESQ evaluation functions.
  * `optimize_params.py`: Grid search script to find optimal $\alpha$ and $\beta$ values.
* `data/`: Directory for input audio (clean, noisy, and noise profiles).
* `results/`: Directory where enhanced audio outputs and spectrograms are saved.

## Setup and Installation
1. Ensure you have Python 3.8+ installed.
2. Clone or download this repository.
3. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt