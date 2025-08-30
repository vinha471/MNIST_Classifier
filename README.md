# MNIST Classifier

A PyTorch-based implementation of neural network classifiers for binary classification on the Fashion MNIST dataset. This project demonstrates both manual neural network implementation from scratch and using PyTorch's built-in modules.

## Table of Contents

-   [Overview](#overview)
-   [File Structure](#file-structure)
-   [Installation](#installation)
-   [Quick Start](#quick-start)
-   [Features](#features)
-   [Usage](#usage)
-   [Examples](#examples)
-   [API Reference](#api-reference)
-   [Requirements](#requirements)
-   [Results](#results)
-   [Learning Objectives](#learning-objectives)
-   [Technical Details](#technical-details)
-   [Contributing](#contributing)
-   [License](#license)

## Overview

This project implements binary classification models to distinguish between two classes from the Fashion MNIST dataset:

-   **T-shirts** (label 0)
-   **Sneakers** (label 9)

The project serves as an educational tool for understanding neural network fundamentals and PyTorch implementation.

## File Structure

```
MNIST_Classifier/
├── README.md                 # This file
├── MNIST_Classifier.ipynb    # Main Jupyter notebook with implementation
├── requirements.txt          # Python dependencies
├── data/                    # Dataset storage (auto-created)
│   └── FashionMNIST/       # Fashion MNIST dataset files
│       ├── raw/            # Raw dataset files
│       │   ├── train-images-idx3-ubyte
│       │   ├── train-labels-idx1-ubyte
│       │   ├── t10k-images-idx3-ubyte
│       │   └── t10k-labels-idx1-ubyte
│       └── processed/      # Processed dataset files
```

## Installation

### Prerequisites

-   Python 3.7 or higher
-   pip package manager
-   Jupyter Notebook or JupyterLab

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd MNIST_Classifier
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:

```bash
jupyter notebook MNIST_Classifier.ipynb
```

## Quick Start

1. **Load the dataset**:

```python
from MNIST_Classifier import load_binary_fashion_mnist

trainloader, testloader = load_binary_fashion_mnist(batch_size=64)
```

2. **Train the manual model**:

```python
from MNIST_Classifier import MLPFromScratch, train_one_epoch

model = MLPFromScratch()
train_one_epoch(model, trainloader, lr=0.001)
```

3. **Train the PyTorch model**:

```python
from MNIST_Classifier import MLPWithNN, train_model

model = MLPWithNN()
train_model(model, trainloader, testloader, epochs=10)
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

## Examples

### Basic Training Example

```python
# Load data
trainloader, testloader = load_binary_fashion_mnist(batch_size=32)

# Create model
model = MLPFromScratch()

# Train for one epoch
train_one_epoch(model, trainloader, lr=0.001)
```

### Full Training Pipeline

```python
# Load data
trainloader, testloader = load_binary_fashion_mnist(batch_size=64)

# Create and initialize PyTorch model
model = MLPWithNN()
model.apply(init_weights)

# Train for multiple epochs
train_model(model, trainloader, testloader, epochs=10, lr=0.001)
```

### Data Visualization

```python
# Show sample images
show_samples(trainloader)
show_samples(testloader)
```

## API Reference

### Core Classes

#### `MLPFromScratch`

Manual implementation of a 3-layer MLP using PyTorch tensors.

**Methods:**

-   `__init__(input_size=784)`: Initialize weights and biases
-   `forward(x)`: Forward pass through the network
-   `parameters()`: Return list of trainable parameters

#### `MLPWithNN`

PyTorch module-based implementation of the same 3-layer MLP.

**Methods:**

-   `__init__()`: Initialize the network architecture
-   `forward(x)`: Forward pass through the network

### Utility Functions

#### `load_binary_fashion_mnist(batch_size=64)`

Loads Fashion MNIST dataset and filters for binary classification.

**Parameters:**

-   `batch_size`: Size of training batches

**Returns:**

-   `trainloader`: DataLoader for training data
-   `testloader`: DataLoader for test data

#### `show_samples(dataloader)`

Visualizes sample images from the dataset.

**Parameters:**

-   `dataloader`: DataLoader containing images and labels

#### `train_one_epoch(model, trainloader, lr=0.001)`

Trains a model for one epoch using manual gradient descent.

**Parameters:**

-   `model`: MLPFromScratch model instance
-   `trainloader`: Training data loader
-   `lr`: Learning rate for gradient descent

#### `train_model(model, trainloader, testloader, epochs=10, lr=0.001)`

Full training loop for PyTorch models with evaluation.

**Parameters:**

-   `model`: PyTorch model instance
-   `trainloader`: Training data loader
-   `testloader`: Test data loader
-   `epochs`: Number of training epochs
-   `lr`: Learning rate for optimizer

## Requirements

-   Python 3.7+
-   PyTorch 1.9+
-   torchvision 0.10+
-   matplotlib 3.3+
-   numpy 1.19+
-   Jupyter Notebook

## Results

The PyTorch implementation typically achieves:

-   **Training Loss**: Near 0.0000 after 10 epochs
-   **Test Accuracy**: 99.95% on the binary classification task

### Performance Comparison

| Model Type            | Training Loss | Test Accuracy | Training Time |
| --------------------- | ------------- | ------------- | ------------- |
| Manual Implementation | ~0.69         | N/A           | Slower        |
| PyTorch Module        | ~0.0000       | 99.95%        | Faster        |

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

### Training Parameters

-   **Learning Rate**: 0.001 (default)
-   **Batch Size**: 64 (configurable)
-   **Epochs**: 10 (configurable)
-   **Loss Function**: Binary Cross Entropy
-   **Activation Functions**: ReLU (hidden), Sigmoid (output)

## Contributing

This is an educational project. Feel free to:

-   Experiment with different architectures
-   Try different activation functions
-   Implement additional evaluation metrics
-   Add data augmentation techniques

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is for educational purposes. Please refer to the original Fashion MNIST dataset license for data usage terms.

---

**Note**: This project is part of CS378: Generative Vision and Computing coursework. For questions or issues, please refer to the course materials or contact the course instructor.
