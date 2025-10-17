#!/bin/bash
# Run YOPO Attack on AdFlush_27 with all configurations
# Cost types: DC, HJC, HCC
# Epsilon values: 5, 10, 20, 40
# Lagrangian: 400
# Sample size: 60000

set -e  # Exit on error

# Data path
DATA_PATH="../../dataset/testing"
OUTPUT_DIR="adflush27_attack_results"

# Attack parameters
LAGRANGIAN=400
SAMPLING_SIZE=60000
NUM_STEPS=100
STEP_SIZE=0.1
NUM_ITERATIONS=10

# Cost types
COST_TYPES=("DC" "HJC" "HCC")

# Epsilon values
EPSILONS=(5 10 20 40)

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run attacks for all combinations
for COST_TYPE in "${COST_TYPES[@]}"; do
    for EPSILON in "${EPSILONS[@]}"; do
        echo "=========================================="
        echo "Running attack: Cost=$COST_TYPE, Epsilon=$EPSILON"
        echo "Lagrangian=$LAGRANGIAN, Sample size=$SAMPLING_SIZE"
        echo "Iterations=$NUM_ITERATIONS"
        echo "=========================================="

        python yopo_adflush27_attack.py \
            --data-path "$DATA_PATH" \
            --cost-type "$COST_TYPE" \
            --epsilon "$EPSILON" \
            --lagrangian "$LAGRANGIAN" \
            --sampling-size "$SAMPLING_SIZE" \
            --num-steps "$NUM_STEPS" \
            --step-size "$STEP_SIZE" \
            --num-iterations "$NUM_ITERATIONS" \
            --output-dir "$OUTPUT_DIR" \
            --constrained

        echo ""
        echo "Completed: Cost=$COST_TYPE, Epsilon=$EPSILON"
        echo ""
    done
done

echo "=========================================="
echo "All attacks completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
