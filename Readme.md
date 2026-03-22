# Toothbrush Defect Detection (U-Net)

This project implements a PyTorch-based U-Net architecture for the automated detection and segmentation of defects on toothbrushes. It uses patch-based training to effectively identify anomalies using a limited dataset.

## Project Structure

The codebase has been modularized for readability and scalability:

* `main.py` - The entry point for running training and evaluation via CLI.
* `config.py` - Contains dataset paths and core hyperparameters (e.g., patch size).
* `dataset.py` - Custom PyTorch Dataset class for handling images and masks with synchronized augmentations.
* `model.py` - Definition of the U-Net architecture.
* `trainer.py` - Handles data preparation, patch extraction, and the model training loop.
* `evaluator.py` - Loads trained weights, runs inference, and calculates evaluation metrics (IoU, Dice, Precision, Recall).
* `utils.py` - Helper functions (loss calculation, mathematical kernels, image preprocessing).
* `requirements.txt` - List of Python dependencies.

## Installation

1. Clone the repository.
2. Ensure you have Python 3.8+ installed.
3. Install the required dependencies:

```bash
pip install -r requirements.txt