#!/bin/bash
#=============================================================================
# RUN TRAINING — SESI Full Pipeline
#=============================================================================

set -e

PHASE=${1:-"all"}
NGPUS=${2:-"auto"}
EPOCHS=${3:-100}
BATCH_SIZE=${4:-8}

# Auto-detect GPUs
if [ "$NGPUS" == "auto" ]; then
    if command -v nvidia-smi &> /dev/null; then
        NGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
        NGPUS=1
    fi
fi

echo "=============================================="
echo " SESI Full Training Pipeline"
echo "=============================================="
echo " Phase: $PHASE"
echo " GPUs: $NGPUS"
echo " Epochs: $EPOCHS"
echo " Batch size: $BATCH_SIZE (per GPU)"
echo "=============================================="

mkdir -p ./checkpoints

if [ $NGPUS -gt 1 ]; then
    echo "Starting multi-GPU training with torchrun..."
    torchrun --nproc_per_node=$NGPUS --master_port=29500 \
        train_full_pipeline.py \
        --phase $PHASE \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE
else
    echo "Starting single-GPU training..."
    python train_full_pipeline.py \
        --phase $PHASE \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE
fi

echo "=============================================="
echo " Training complete!"
echo "=============================================="
