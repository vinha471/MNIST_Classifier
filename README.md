# MNIST Classifier

A PyTorch-based implementation of neural network classifiers for binary classification on the Fashion MNIST dataset. This project demonstrates both manual neural network implementation from scratch and using PyTorch's built-in modules.

## Overview

This project implements binary classification models to distinguish between two classes from the Fashion MNIST dataset:

-   **T-shirts** (label 0)
-   **Sneakers** (label 9)

The project serves as an educational tool for understanding neural network fundamentals and PyTorch implementation.

## Project Structure

```
MNIST_Classifier/
├── MNIST_Classifier.ipynb    # Main Jupyter notebook with implementation
└── README.md                 # This file
```

## Features

### 1. Manual Neural Network Implementation (`MLPFromScratch`)

-   **Architecture**: 3-layer Multi-Layer Perceptron (MLP)
    -   Input layer: 784 neurons (28×28 flattened images)
    -   Hidden layer 1: 512 neurons with ReLU activation
    -   Hidden layer 2: 128 neurons with ReLU activation
    -   Output layer: 1 neuron with sigmoid activation
-   **Implementation**: Built from scratch using only PyTorch tensors and basic operations
-   **Training**: Manual gradient descent implementation

### 2. PyTorch Module Implementation (`MLPWithNN`)

-   **Architecture**: Same 3-layer MLP structure
-   **Implementation**: Uses PyTorch's `nn.Module` and `nn.Sequential`
-   **Training**: Leverages PyTorch's built-in optimizers and loss functions
-   **Initialization**: Custom weight initialization function

### 3. Data Handling

-   **Dataset**: Fashion MNIST (automatically downloaded)
-   **Preprocessing**: Binary classification between T-shirts and Sneakers
-   **Transforms**: Normalization and tensor conversion
-   **Data Loaders**: Batch processing with configurable batch sizes

## Requirements

-   Python 3.x
-   PyTorch
-   torchvision
-   matplotlib
-   numpy

## Installation

1. Ensure you have Python and pip installed
2. Install required packages:

```bash
pip install torch torchvision matplotlib numpy
```

## Usage

### Running the Notebook

1. Navigate to the `MNIST_Classifier` directory
2. Open `MNIST_Classifier.ipynb` in Jupyter Notebook or JupyterLab
3. Run cells sequentially to:
    - Load and visualize the Fashion MNIST dataset
    - Train the manual implementation
    - Train the PyTorch module implementation
    - View training progress and results

### Key Functions

-   `load_binary_fashion_mnist(batch_size=64)`: Loads and filters Fashion MNIST data
-   `show_samples(dataloader)`: Visualizes sample images from the dataset
-   `train_one_epoch(model, trainloader, lr=0.001)`: Trains manual implementation for one epoch
-   `train_model(model, trainloader, testloader, epochs=10, lr=0.001)`: Full training loop for PyTorch model

## Results

The PyTorch implementation typically achieves:

-   **Training Loss**: Near 0.0000 after 10 epochs
-   **Test Accuracy**: 99.95% on the binary classification task

## Learning Objectives

This project demonstrates:

1. **Neural Network Fundamentals**: Forward/backward propagation, activation functions
2. **PyTorch Basics**: Tensors, autograd, data loaders, modules
3. **Binary Classification**: Loss functions (BCE), evaluation metrics
4. **Model Comparison**: Manual vs. framework-based implementation
5. **Data Preprocessing**: Image normalization, dataset filtering

## Technical Details

### Model Architecture

-   **Input**: 784-dimensional flattened images (28×28 pixels)
-   **Hidden Layers**: Fully connected with ReLU activation
-   **Output**: Single neuron with sigmoid activation for binary classification
-   **Loss Function**: Binary Cross Entropy (BCE)
-   **Optimizer**: Adam (for PyTorch implementation)

### Data Processing

-   Images are normalized to range [-1, 1]
-   Labels are converted to binary (0 for T-shirt, 1 for Sneaker)
-   Batch processing with configurable batch sizes
-   Train/test split using Fashion MNIST's built-in partitioning

## Contributing

This is an educational project. Feel free to:

-   Experiment with different architectures
-   Try different activation functions
-   Implement additional evaluation metrics
-   Add data augmentation techniques

## License

This project is for educational purposes. Please refer to the original Fashion MNIST dataset license for data usage terms.
