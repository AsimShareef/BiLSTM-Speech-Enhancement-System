# Speech Enhancement: BiLSTM Mask Estimation

## Overview
This repository contains a Deep Learning-based Speech Enhancement pipeline developed for the Advanced Topics in Speech Processing (CS60116/IT60116) course at IIT Kharagpur. 

Replacing classical baseline methods like Spectral Subtraction, this modern approach utilizes a Bidirectional Long Short-Term Memory (BiLSTM) neural network to estimate an Ideal Ratio Mask (IRM). Operating entirely in the time-frequency domain, the network acts as a highly contextual, non-linear gain filter to suppress background noise while preserving human speech formants. 

The target Ideal Ratio Mask is mathematically defined as:
$$M(k, m) = \frac{|S(k, m)|}{|Y(k, m)| + \epsilon}$$

## Project Structure
* **`config.py`**: Centralized source of truth for all hyperparameters, FFT sizes, and layer dimensions.
* **`data_processing.py`**: Pipeline for synthetic data generation, directory-based pair matching, and STFT tensor framing.
* **`manual_stft.py`**: Custom, from-scratch DSP implementations of the STFT and ISTFT (bypassing standard libraries).
* **`model.py`**: TensorFlow/Keras architecture for the BiLSTM mask predictor with a Time-Distributed Dense output.
* **`train.py`**: End-to-end training script supporting synthetic data, single-file pairs, or large directory batches.
* **`inference.py`**: Command-line tool to enhance unseen noisy audio using saved `.keras` model weights.
* **`reconstruction.py`**: Logic for overlap-add mask unframing and phase-aware audio waveform reconstruction.

## Setup and Usage
1. Ensure you have Python 3.8+ installed. 
2. Install the required dependencies (`tensorflow`, `numpy`, `soundfile`). 
> **Note for Apple Silicon users:** You must install `tensorflow-macos` and `tensorflow-metal` to prevent profiler crashes.

### Training the Model
To train the model on a directory of matching clean and noisy `.wav` files:
```bash
python train.py --clean clean_trainset_wav/ --noisy noisy_trainset_wav/