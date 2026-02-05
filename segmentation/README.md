# Understanding CNNs with PyTorch and FashionMNIST

## 1. Project Overview

This project serves as a practical introduction to **Convolutional Neural Networks (CNNs)** using the **FashionMNIST** dataset. The goal is to build, train, and evaluate a Deep Learning model capable of classifying images of clothing items (T-shirts, trousers, sneakers, etc.) with high accuracy.

We implement this solution using **PyTorch**, a leading deep learning framework, within a Jupyter Notebook environment (`test1.ipynb`).

---

## 2. Concept: Why Convolutional Neural Networks?

Traditional "Feed-Forward" networks (Multi-Layer Perceptrons) treat every pixel in an image as an independent input. This destroys spatial information (e.g., that a pixel is related to its neighbors).

**Convolutional Neural Networks (CNNs)** solve this by using **Filters (Kernels)** that slide over the image to detect features.

### Key Components Used:
*   **Convolution (`Conv2d`)**: Scans the image to extract features like edges, textures, and shapes. It preserves the spatial relationship between pixels.
*   **Activation (`ReLU`)**: Rectified Linear Unit. It introduces non-linearity, allowing the network to learn complex patterns (not just straight lines).
*   **Pooling (`MaxPool2d`)**: Reduces the size of the image representation (downsampling). This makes the computation faster and detecting features less sensitive to exact location (translation invariance).
*   **Fully Connected Layers (`Linear`)**: Used at the end to make a final classification decision based on the high-level features extracted by the convolutional layers.

---

## 3. Technology Stack

*   **Python**: The programming language.
*   **PyTorch (`torch`, `torch.nn`)**: The core deep learning library used for defining the model and automatic differentiation (Backpropagation).
*   **Torchvision**: Helper library providing standard datasets (`FashionMNIST`) and image transformations (`ToTensor`).
*   **Matplotlib**: Used for visualizing the dataset images.

---

## 4. Implementation Details

### A. Data Loading
We use the `FashionMNIST` dataset, which consists of 60,000 training images and 10,000 test images.
*   **Images**: 28x28 grayscale (1 channel).
*   **Classes**: 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
*   **DataLoader**: Wraps the dataset to provide **batching** (loading `64` images at a time) and **shuffling** (randomizing order for better training).

### B. Model Architecture (`SimpleCNN`)
Our model follows a standard architecture pattern:

1.  **Block 1**:
    *   `Conv2d`: 1 input channel -> 32 filters (3x3 kernel).
    *   `ReLU`
    *   `MaxPool2d`: Reduces dimension by half (28x28 -> 14x14).
2.  **Block 2**:
    *   `Conv2d`: 32 input channels -> 64 filters.
    *   `ReLU`
    *   `MaxPool2d`: Reduces dimension by half (14x14 -> 7x7).
3.  **Classifier Head**:
    *   `Flatten`: Converts 3D feature maps (64 channels x 7 x 7) into a 1D vector.
    *   `Linear`: Hidden layer (3136 -> 512).
    *   `ReLU`
    *   `Linear`: Output layer (512 -> 10 scores).

### C. Training Process
*   **Loss Function**: `CrossEntropyLoss` (standard for multi-class classification).
*   **Optimizer**: `Adam` (Adaptive Moment Estimation), which usually converges faster than standard SGD.
*   **Loop**:
    1.  **Forward Pass**: Compute predictions.
    2.  **Compute Loss**: Compare predictions to actual labels.
    3.  **Backward Pass**: Calculate gradients.
    4.  **Step**: Update model weights.

---

## 5. How to Run

1.  **Environment**: Ensure you have a Python environment with PyTorch installed.
    ```bash
    pip install torch torchvision matplotlib
    ```
2.  **Open Notebook**: Launch Jupyter Notebook or VS Code.
    ```bash
    # If using command line
    jupyter notebook segmentation/test1.ipynb
    ```
3.  **Execute Cells**: Run the cells in order.
    *   It will first download the data.
    *   Then visualize a sample.
    *   Create DataLoaders.
    *   Initialize the Model.
    *   Train for 5 epochs (you should see accuracy increase).

---

## 6. Why This Implementation?

We chose this specific architecture because it balances **simplicity** and **performance**. Only using Linear layers would yield lower accuracy (~85%), whereas this simple CNN can easily reach **~90% accuracy** within just a few epochs, demonstrating the power of spatial feature extraction.
