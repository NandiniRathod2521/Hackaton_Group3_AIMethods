---
language:
- en
license: mit
tags:
- anomaly-detection
- construction
- imu
- autoencoder
- onnx
- pytorch
- iaac
---

# MotionAtlas — Experimental Conv1D-32 Autoencoder

## Overview

Experimental convolutional autoencoder for anomaly detection
in construction-worker motion data.

Developed as part of the IAAC AI Methods Hackathon,
Group 3.

This model is a research prototype, not a certified
workplace safety system.

## Dataset

VTT-ConIoT construction-worker motion dataset.

Source: https://doi.org/10.5281/zenodo.4683703

- Sampling frequency: 100 Hz
- Window size: 200 samples (2 seconds)
- Input features: 18 IMU measurements
- Normal activities: 13, 14, 15
- Anomaly activities: 7, 8, 11, 12

## Experimental protocol

- Training workers: 1–9
- Validation workers: 10–11
- Test workers: 12–13
- Training: normal activities only
- Validation: normal activities only
- Test: normal and anomalous activities

Training windows: 1566
Validation windows: 175
Test windows: 406

Normalization statistics were fitted using training
data only.

## Model

Architecture: Conv1DAutoencoder

Latent dimension: 32

Input shape: (batch, 200, 18)

Output shape: (batch, 200, 18)

Anomaly score: mean squared reconstruction error.

Threshold: 1.61768460

Threshold calibration: 95th percentile of normal
validation reconstruction errors.

## Test results

| Metric | Value |
|---|---:|
| ROC AUC | 0.838907 |
| Precision | 0.934066 |
| Recall | 0.366379 |
| F1-score | 0.526316 |

Confusion matrix:

| | Predicted normal | Predicted anomaly |
|---|---:|---:|
| Actual normal | 168 | 6 |
| Actual anomaly | 147 | 85 |

## Files

- `model.onnx`: exported Conv1D-32 model
- `scaler.npz`: training normalization statistics
- `config.json`: inference configuration and threshold

The input features must follow the exact order
specified in `config.json`.

## Limitations

- Evaluated on only two held-out workers.
- Misses 147 of 232 anomalous test windows.
- The threshold is dataset-specific.
- Results may not generalize to other workers,
  sensors, activities, or environments.
- Test results must not be used to select or tune
  the model without an independent evaluation.
- Not suitable for safety-critical deployment.

## Intended use

Educational research, experimental anomaly detection,
and reproducible comparison of autoencoder architectures.

Not intended for autonomous workplace safety decisions.
