# Facial Expression Classification with CNNs

A PyTorch computer vision project that trains and evaluates convolutional neural networks (CNNs) on the FER2013+ facial expression dataset. The notebooks explore image preprocessing, class imbalance, augmentation, hyperparameter tuning, and per-class evaluation.

## Overview

The workflow uses grayscale facial images and an eight-class CNN classifier. It compares experiments under two dataset preparation approaches, then evaluates saved model weights on held-out images. The repository is a collection of research notebooks and checkpoints, rather than a deployed application.

## Experiments

| Area | What it covers |
| --- | --- |
| `cnn_emotions_before_augmentation/` | Initial CNN, hyperparameter search, evaluation, and saved weights using the original dataset folders with notebook-defined image transforms. |
| `cnn_emotions_after_augmentation/` | Initial CNN, hyperparameter search, evaluation, and saved weights using augmented dataset folders. |
| `cnn_stratify.ipynb` | A separate training experiment with a stratified validation approach and weighted sampling. |
| `cnn_stratify_hyperparam.ipynb` | A separate hyperparameter experiment using Optuna. |
| `logs/` | TensorBoard event files from training runs. |

Despite the folder names, notebooks in the “before” track also define training-time image transforms. The distinction is the use of original versus augmented dataset folders in the two tracks.

## Methods

- **Model:** PyTorch CNNs with convolutional layers, pooling, batch normalization, fully connected layers, and dropout.
- **Preprocessing:** grayscale conversion, resizing/cropping, normalization, and image augmentation in training pipelines.
- **Class imbalance:** weighted sampling in training data loaders.
- **Training:** validation monitoring, learning-rate scheduling, early stopping, and TensorBoard logging in the main experiments.
- **Tuning:** search over parameters such as learning rate, batch size, optimizer, dropout, and weight decay; the notebooks use scikit-optimize or Optuna in different experiments.
- **Evaluation:** overall and class-wise accuracy, plus visual comparisons of predictions and true labels.

The notebook narratives report test results for specific runs. Those numbers depend on the dataset version, split, random state, and training environment, and have not been independently rerun for this README.

## Repository structure

| Path | Contents |
| --- | --- |
| [`cnn_emotions_before_augmentation/`](cnn_emotions_before_augmentation/) | Initial training, tuning, evaluation notebooks, and `.pth` checkpoints. |
| [`cnn_emotions_after_augmentation/`](cnn_emotions_after_augmentation/) | Corresponding experiments with augmented dataset folders and checkpoints. |
| [`cnn_stratify.ipynb`](cnn_stratify.ipynb) | Stratified split and weighted-sampling experiment. |
| [`cnn_stratify_hyperparam.ipynb`](cnn_stratify_hyperparam.ipynb) | Optuna tuning experiment. |
| [`requirements.txt`](requirements.txt) | Python dependencies. |

## Getting started

1. Create a Python environment and install the dependencies in `requirements.txt`.
2. Open a notebook with Jupyter from the repository root or its containing folder.
3. Run the dataset download and preparation cells. The notebooks use `kagglehub` to obtain `subhaditya/fer2013plus`; some cells expect a local `~/datasets/fer2013plus/fer2013/` layout.
4. Follow the initial training notebook before the tuning and evaluation notebook for the same track. The evaluation notebooks load the corresponding saved `.pth` state dictionary.

A GPU can speed up training, but the notebooks check whether CUDA is available. Dataset download and notebook-specific local paths are required to reproduce the runs. The saved model files are weights, not standalone applications.

## Technology

Python · PyTorch · torchvision · Jupyter · kagglehub · scikit-optimize · Optuna · TensorBoard · NumPy · Matplotlib
