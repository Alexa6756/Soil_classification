#!/bin/bash

# Exit the script if any command fails
set -e

echo "Downloading the soil-classification-part-2 dataset from Kaggle..."

# Check if the Kaggle API token exists
if [ ! -f ~/.kaggle/kaggle.json ]; then
  echo "Kaggle API token not found at ~/.kaggle/kaggle.json."
  echo "Please download it from https://www.kaggle.com/account and place it in the specified location."
  exit 1
fi

# Set appropriate permissions for the Kaggle API token
chmod 600 ~/.kaggle/kaggle.json

# Create directory for dataset
mkdir -p ./data/soil_competition-2025

# Download the dataset using Kaggle CLI
kaggle competitions download -c soil-classification-part-2 -p ./data/

# Unzip the downloaded archive
unzip -q ./data/*.zip -d ./data/soil_competition-2025

echo "Dataset has been successfully downloaded and extracted to ./data/soil_competition-2025."

