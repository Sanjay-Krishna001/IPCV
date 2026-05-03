
# Wildlife Species Detection from Camera Trap Images 🦌🦝

A classical Machine Learning pipeline designed to classify wildlife species from the Caltech Camera Traps (CCT-20) dataset. This project relies entirely on mathematical feature extraction and Support Vector Machines (SVM), intentionally avoiding Deep Learning/CNNs to focus on fundamental Computer Vision techniques.

## 🎯 Project Objective
The goal of this project is to accurately detect and classify specific animal species (or empty frames) from night and day camera trap images. 

**Core Techniques Used:**
*   **HOG (Histogram of Oriented Gradients):** Extracts structural and shape-based features (outlines, edges).
*   **Colour Histograms:** Extracts texture and color distributions across the RGB channels.
*   **Multi-Class SVM:** A Support Vector Machine with an RBF kernel used to classify the concatenated feature vectors.

## 🛠️ Tech Stack
*   **Python 3.10+**
*   **OpenCV (`cv2`)** - Image processing and feature extraction
*   **Scikit-Learn (`sklearn`)** - Model training, scaling, and evaluation
*   **NumPy** - Matrix and array operations
*   **Matplotlib** - Confusion matrix visualization

## 📂 Project Structure
```text
├── annotations.json      # Original dataset annotations (CCT-20)
├── images/               # Raw downloaded camera trap images
├── dataset/              # Dynamically generated 500-image subset sorted by class
├── create_dataset.py     # Parses JSON and organizes a balanced multi-class subset
├── features.py           # Core logic for HOG + Colour Histogram extraction
├── train_model.py        # Trains the SVM, prints metrics, and saves the model
├── predict.py            # Loads the trained model to run inference on new images
├── svm_model.pkl         # Saved SVM model (generated after training)
└── scaler.pkl            # Saved StandardScaler (generated after training)
```

## 🚀 How to Run the Pipeline

Follow these steps in order to process the data, train the model, and run predictions.

### 1. Dataset Preparation
First, isolate a balanced subset of 500 images across specific target species (e.g., bobcat, coyote, raccoon, deer, empty) to prevent memory bottlenecks and class imbalances.
```bash
python create_dataset.py
```

### 2. Feature Extraction & Model Training
This script reads the organized `dataset/` directory, resizes images to 64x64 (for memory optimization), extracts the concatenated HOG+Colour features, and trains the SVM.
```bash
python train_model.py
```
*Note: This script will output the model's overall Accuracy, a detailed Classification Report, and display a visual Confusion Matrix. It also saves `svm_model.pkl` and `scaler.pkl` to your directory.*

### 3. Inference / Prediction
Test the trained model on a new, unseen image. Open `predict.py` and point the `predict_image()` function to your target image, then run:
```bash
python predict.py
```

## 🧠 Overcoming Challenges: Night Vision vs. Feature Extraction
Camera trap images present a unique computer vision challenge: **infrared night photography**. 
Because nighttime images strip away natural colors, a brown deer and a gray raccoon can look identical to a Colour Histogram. To combat this, the pipeline normalizes the color histograms and relies heavily on the explicitly configured `cv2.HOGDescriptor` blocks and cells to map the physical outlines of the animals through the background brush.

## 📝 License
This project is submitted as part of an academic requirement. The Caltech Camera Traps dataset belongs to its respective creators (NPS, USGS, and contributors).
```
