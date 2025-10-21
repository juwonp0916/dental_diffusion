#!/bin/bash
################################################################################
# Training Script for CBCT Point Cloud Diffusion Model
################################################################################
# This script trains the point cloud diffusion model on dental CBCT data
#
# Requirements:
# - Python 3.11 with conda environment
# - CUDA-capable GPU(s)
# - Data in ./data folder with patient folders (e.g., DBT1_0002L, DBT1_0002U)
#
# Usage:
#   chmod +x run_training.sh  # Make executable (first time only)
#   ./run_training.sh         # Run training
################################################################################

echo "========================================"
echo "CBCT Point Cloud Diffusion Training"
echo "========================================"
echo

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "ERROR: data directory not found!"
    echo "Please ensure your dataset is in the ./data folder"
    exit 1
fi

# Display data info
echo "Checking dataset..."
patient_count=$(ls -d data/DBT* 2>/dev/null | wc -l)
echo "Found $patient_count patient jaw folders in ./data"
echo

################################################################################
# Configuration Parameters
################################################################################

# Batch size (28 teeth x 1024 points) - adjust based on GPU memory
# Recommended: 1 batch per 24GB GPU memory
BATCH_SIZE=4

# Number of workers for data loading
# Recommended: 1 worker per batch as starting point
NUM_WORKERS=4

# Number of training iterations (epochs)
NUM_EPOCHS=1000

# GPU devices to use (comma-separated)
# Current: GPU 6,7 (as set in train_generation.py line 4)
# Modify train_generation.py line 4 to change GPU selection

# Learning rate
LEARNING_RATE=0.0002

# Model save interval (in epochs)
SAVE_INTERVAL=200

# Visualization interval (in epochs)
VIZ_INTERVAL=200

################################################################################

echo "Configuration:"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Workers: $NUM_WORKERS"
echo "  - Epochs: $NUM_EPOCHS"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Save every $SAVE_INTERVAL epochs"
echo "  - Visualize every $VIZ_INTERVAL epochs"
echo
echo "Starting training in 3 seconds..."
sleep 3

# Create output directories if they don't exist
mkdir -p checkpoints
mkdir -p logs
mkdir -p visualizations

# Run training with all arguments
python train_generation.py \
    --bs $BATCH_SIZE \
    --workers $NUM_WORKERS \
    --niter $NUM_EPOCHS \
    --lr $LEARNING_RATE \
    --saveIter $SAVE_INTERVAL \
    --vizIter $VIZ_INTERVAL \
    --path ./data

echo
echo "========================================"
echo "Training completed!"
echo "Check ./checkpoints for saved models"
echo "Check ./visualizations for training progress"
echo "========================================"
