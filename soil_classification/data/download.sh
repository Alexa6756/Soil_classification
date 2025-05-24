#!/bin/bash

# download.sh
# Script to download and extract the Soil Classification dataset from Kaggle

echo "Starting dataset download..."

# Create data directory if it doesn't exist
mkdir -p data

# Navigate to data directory
cd data

# Download dataset using Kaggle CLI (ensure kaggle API is configured)
kaggle competitions download -c soil-classification

echo "Download completed."

# Unzip the dataset files
unzip -o soil-classification.zip

echo "Extraction completed."

# Remove the zip file to save space
rm soil-classification.zip

echo "Cleanup completed. Dataset is ready in the 'data/' directory."

