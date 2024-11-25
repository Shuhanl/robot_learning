#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Navigate to the submodule 'diff-gaussian-rasterization' and install it
echo "Installing diff-gaussian-rasterization..."
cd submodules/diff-gaussian-rasterization
pip install .

# Return to the main directory
cd ../..

# Navigate to the submodule 'simple-knn' and install it
echo "Installing simple-knn..."
cd submodules/simple-knn
pip install .

# Return to the main directory
cd ../..

# Install required packages from the requirements.txt
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete!"

