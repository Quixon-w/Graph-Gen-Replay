# VGAE Reimplementation

This directory contains an engineering-style implementation of **Variational Graph Auto-Encoder (VGAE)**.

## How to Run
1. Install dependencies
2. Run training: `python main.py`

## Results
- Target Dataset: Cora
- Expected AUC: ~0.90+ (Link Prediction)

## Structure
- `configs/`: Hyperparameters in YAML.
- `model/`: Modular GCN Encoder.
- `utils/`: Data loading and splitting.