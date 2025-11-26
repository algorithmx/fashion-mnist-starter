# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Fashion-MNIST starter repository containing minimal CNN implementations in both PyTorch and TensorFlow/Keras. The project focuses on providing simple, clean examples for training on the Fashion-MNIST dataset with proper GPU/CPU handling and data loading utilities.

## Architecture

### Core Components

- **dataset_loader.py**: Central data loading module that handles Fashion-MNIST dataset loading from local IDX files or downloads via torchvision. Provides normalized data in range [-1, 1] and returns both numpy arrays and PyTorch datasets/DataLoaders.

- **starter_pytorch.py**: Minimal PyTorch CNN implementation using a single conv layer, max pooling, and fully connected layer. Includes GPU capability checking with compute capability validation (falls back to CPU for GPUs < 7.0).

- **starter_keras.py**: Equivalent TensorFlow/Keras implementation with similar architecture. Includes sophisticated GPU/CPU detection that disables CUDA when `ptxas` is missing or for older GPUs.

### Data Flow

Both implementations follow the same pattern:
1. Load normalized Fashion-MNIST data through `dataset_loader.py`
2. Define minimal CNN model with one conv layer
3. Train with Adam optimizer and cross-entropy loss
4. Print training/validation accuracy per epoch

## Common Development Commands

### Running Training
```bash
# PyTorch version
python starter_pytorch.py

# TensorFlow/Keras version
python starter_keras.py
```

### Data Handling
The dataset will be automatically downloaded to `data/FashionMNIST/raw/` on first run if not present. Both scripts use the shared `dataset_loader.py` module which:
- Reads raw IDX format files directly
- Falls back to torchvision download if files missing
- Returns normalized float32 arrays in range [-1, 1]

### Dependencies
- PyTorch version: `torch`, `torchvision`, `matplotlib`
- TensorFlow version: `tensorflow`, `numpy`, `matplotlib`
- Both versions share the same `dataset_loader.py` module

## Device Selection

Both implementations include intelligent GPU/CPU selection:

- **PyTorch**: Checks CUDA availability and GPU compute capability, falls back to CPU for compute capability < 7.0
- **TensorFlow**: More sophisticated detection that disables CUDA when `ptxas` (CUDA assembler) is missing or for older GPUs. Override with `FORCE_TF_GPU=1` environment variable.

## Model Architecture

Both CNNs implement identical minimal architecture:
- Conv2D layer: 16 filters, 3x3 kernel, valid padding
- ReLU activation
- Max pooling: 2x2, stride 2
- Flatten layer
- Dense layer: 10 output units (logits for 10 classes)