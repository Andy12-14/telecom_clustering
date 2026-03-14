# Mask R-CNN Rock Climbing Hold Detector

Welcome to the Mask R-CNN Segmentation project. This folder contains the tools and notebooks used to train and evaluate a computer vision model that detects and segments rock climbing holds. This guide will walk you through the primary notebook, `climb_epoch10.ipynb`, detailing every step and function so you can easily understand and run the project.

## Directory Overview
- `climb_epoch10.ipynb`: The primary Jupyter Notebook containing the data loading, model architecture, training loop, and evaluation visualizations for our 10-epoch model.
- `requirements.txt`: The required Python packages for running the Mask R-CNN and handling the COCO dataset.
- `perf.txt`: Contains performance metrics and logs from the training.

---

## The `climb_epoch10.ipynb` Notebook: Step-by-Step

This notebook trains a **Mask R-CNN ResNet-50 FPN** (Feature Pyramid Network) model on a custom COCO-annotated dataset (`Hold Detector.v2i.coco-segmentation`) to recognize climbing holds. Below is a breakdown of the core functions and their purposes.

### 1. The Dataset Class (`ClimbingDataset`)
**Purpose:** To load the images and their corresponding COCO annotations (bounding boxes and masks), and format them for the PyTorch model.
- **`__init__`**: Initializes the COCO API, reads the `_annotations.coco.json` file, and filters out any images that are missing from the directory to prevent crashing.
- **`__getitem__`**: Fetches a single image by its index. It calculates the proper bounding box formulas (`[xmin, ymin, xmax, ymax]`) and converts the annotations and masks into PyTorch validation Tensors.
- **`__len__`**: Returns the total number of valid images loaded in the dataset.

### 2. Data Augmentation & Collation
- **`get_transform(train)`**: A helper function that converts the loaded PIL Images into PyTorch Tensors.
- **`collate_fn(batch)`**: A utility function required by the PyTorch `DataLoader` to properly merge a list of samples (images and targets) into a batch tensor.

### 3. Model Architecture (`get_model_instance_segmentation`)
**Purpose:** To initialize and modify a pre-trained Mask R-CNN model tailored for our specific task.
- We start with the `maskrcnn_resnet50_fpn` pre-trained on the standard COCO dataset.
- We replace the **Box Predictor** (for bounding boxes) with a new `FastRCNNPredictor` that matches our custom numerical classes (2 classes: background + hold).
- We replace the **Mask Predictor** (for instance segmentation masks) with a new `MaskRCNNPredictor` tailored to our 2 classes.

### 4. Training the Model (`train_one_epoch`)
**Purpose:** Executes one full pass over the training dataset.
- Sets the model to training mode.
- Iterates through the `data_loader`, calculating the `loss_dict` which contains classification, bounding box, and mask prediction errors.
- Performs backpropagation (`losses.backward()`) and updates the weights (`optimizer.step()`).
- Returns the average loss for the epoch so we can plot the performance over time.

### 5. Visualizing Predictions (`visualize_test_predictions`)
**Purpose:** Evaluates how well the model learned by visualizing its predictions on unseen test images.
- Picks random images from the test or validation datasets.
- Runs the images through the trained model using `model.eval()`.
- Filters out low-confidence predictions using a parameter `score_threshold` (default is 0.5 or 50% confidence).
- Plots the original image overlaid with a translucent segmentation mask, a solid bounding box, and the confidence score text for every hold detected.

---

## Running the Pipeline (Main Execution Loop)
The main execution block in the notebook ties all the functions together:
1. **Device setup:** Automatically detects and uses the GPU (CUDA) if available, otherwise falls back to the CPU.
2. **Data Loaders:** Instantiates the training, validation, and testing `ClimbingDataset`s and passes them to PyTorch `DataLoader`s.
3. **Optimizer & Scheduler:** Uses Stochastic Gradient Descent (SGD) with a learning rate scheduler that drops the learning rate by 90% every 3 epochs to fine-tune the training.
4. **The Training Loop:** Runs for **10 epochs**. It saves the average loss of each epoch and plots a visual loss curve (`training_loss_curve_epoch10.png`) at the very end. Our model achieved an average loss of roughly ~1.19 by the end of training.
5. **Saving the Model:** The finished weights are saved into `climbing_model-epoch10.pth`.
6. **Testing:** The final blocks run `visualize_test_predictions` to automatically display visual results of the detected climbing holds on a sample of test data!

---

## How to use this project
If you want to use or reproduce this work:
1. Open the CLI and install the environment libraries by running `pip install -r requirements.txt`. This ensures PyTorch and `pycocotools` are available.
2. Ensure the `Hold Detector.v2i.coco-segmentation` dataset is extracted in the proper path (`segmentation/MaskRCNN/Hold Detector.v2i.coco-segmentation`).
3. Run the notebook `climb_epoch10.ipynb` from top to bottom. It will process the dataset, train the model, and evaluate its output.
