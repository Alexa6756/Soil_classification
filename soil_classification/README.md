 ## ☘️ Soil Image Classification using Deep Learning

This project aims to develop a robust machine learning system for **classifying soil images into one of four soil types**:

🟤 **Alluvial**, 
⚫ **Black**, 
🟫 **Clay**,
🔴 **Red**.

The system utilizes computer vision and deep learning techniques to automatically identify the soil type based on photographic inputs.



## 🧠 Problem Statement

Soil plays a foundational role in supporting agriculture, ecosystems, and infrastructure. Accurately identifying soil types from images enables better:

* 🌾 Crop selection and land management
* 🌍 Environmental monitoring
* 🏗️ Engineering decision-making

The goal is to train a model that can **predict the type of soil from an image** using features such as texture, color, and pattern.



## 🛠️ Solution Overview

This project employs a **Convolutional Neural Network (CNN)** based architecture for image classification, specifically using **EfficientNet-B4**, enhanced with several optimization techniques for improved accuracy and generalization.

### 🔁 Workflow Summary

1. **Data Preparation**:

   * Resize all images to 224×224 pixels
   * Normalize pixel values
   * Apply data augmentation (flip, crop, rotate, color jitter)

2. **Model Architecture**:

   * Pretrained EfficientNet-B4 via timm library
   * Final fully-connected layer adjusted for 4-class classification

3. **Training Procedure**:

   * Stratified K-Fold Cross-Validation
   * Loss: Focal Loss with class weights
   * Optional Label Smoothing
   * Optimizer: Adam or AdamW
   * Learning rate scheduling with ReduceLROnPlateau
   * Early stopping based on validation F1-score

4. **Inference**:

   * Load best-performing model
   * Apply Test-Time Augmentation (TTA) for robustness
   * Predict classes for unseen test images



## 📈 Evaluation Metrics

The model performance is measured using the following metrics:

* 🎯 **Accuracy**
* 🧮 **Precision**
* 🔁 **Recall**
* 📊 **F1-Score**

These metrics are computed both per-class and as macro/weighted averages to assess overall and class-wise performance.



## 📌 Key Notes

* Input image resolutions are standardized to support deep CNNs
* Class imbalance is handled with weighted loss functions
* Data augmentation improves generalization
* Evaluation includes per-class analysis to avoid overfitting on dominant classes


#### A reliable soil classification model contributes directly to sustainable land use, precision agriculture, and efficient resource planning. 🌿


## Create Virtual Environment (optional)
python -m venv venv
source venv/bin/activate  # For Unix
venv\Scripts\activate     # For Windows

## Install Dependencies
pip install -r requirements.txt

## Download Dataset
bash data/download.sh

## Train the Model
jupyter notebook notebooks/training.ipynb

## Run Inference
jupyter notebook notebooks/inference.ipynb

## Requirements

Please refer to the requirements.txt for a complete list of dependencies.

torch==2.1.0
torchvision==0.16.0
timm==0.9.8
albumentations==1.4.0
scikit-learn==1.3.2
pandas==2.1.1
matplotlib==3.8.0
opencv-python==4.8.1.78

## Contact

Team name: Async Minds

Name:Alexa Kurapati
Institution: IIIT Tiruchirappalli
Email:221128@iiitt.ac.in

Name:Sahithi Sirivella
Institution: Rajiv Gandhi University Of Knowledge Technologies
Email:sirivella.sahithi37@gmail.com

## Acknowledgements

This project is developed as part of the Soil Image Classification Challenge by Annam.ai in partnership with IIT Ropar. We acknowledge their initiative to promote real-world applications of artificial intelligence in agriculture and environmental science.



