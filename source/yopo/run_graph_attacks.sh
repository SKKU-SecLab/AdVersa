#!/bin/bash
# Run YOPO Attack on AdGraph and WebGraph models
# Matches original YOPO parameters from attack_pipeline_adgraph.sh
# Cost types: DC, HSC, HCC
# Epsilon values: 5, 10, 20, 40
# Lagrangian: 400
# Sample size: 60000 (original YOPO)
# num_steps: epsilon * 10 (original YOPO)

set -e  # Exit on error

# Data path
DATA_PATH="../../dataset/testing"
OUTPUT_DIR="graph_attack_results_web_1001"

# Attack parameters (matching original YOPO)
LAGRANGIAN=400
QUERY_SIZE=100000    # Original YOPO uses 100k query set
SAMPLING_SIZE=40000  # Original YOPO samples 40k from query set
STEP_SIZE=0.1
NUM_ITERATIONS=10

# Model types
MODEL_TYPES=("adgraph" "webgraph")

# Cost types
COST_TYPES=("DC" "HSC" "HCC")

# Epsilon values
EPSILONS=(5 10 20 40)

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run attacks for all combinations
for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    for COST_TYPE in "${COST_TYPES[@]}"; do
        for EPSILON in "${EPSILONS[@]}"; do
            # Calculate num_steps based on epsilon (original YOPO: num_step = epsilon * 10)
            NUM_STEPS=$((EPSILON * 10))
            echo "=========================================="
            echo "Running attack: Model=$MODEL_TYPE, Cost=$COST_TYPE, Epsilon=$EPSILON"
            echo "Lagrangian=$LAGRANGIAN, Query size=$QUERY_SIZE, Sample size=$SAMPLING_SIZE"
            echo "Num steps=$NUM_STEPS (epsilon * 10), Iterations=$NUM_ITERATIONS"
            echo "=========================================="
            python yopo_adgraph_webgraph_attack.py \
                --model-type "$MODEL_TYPE" \
                --data-path "$DATA_PATH" \
                --cost-type "$COST_TYPE" \
                --epsilon "$EPSILON" \
                --lagrangian "$LAGRANGIAN" \
                --query-size "$QUERY_SIZE" \
                --sampling-size "$SAMPLING_SIZE" \
                --num-steps "$NUM_STEPS" \
                --step-size "$STEP_SIZE" \
                --num-iterations "$NUM_ITERATIONS" \
                --output-dir "$OUTPUT_DIR"
            echo ""
            echo "Completed: Model=$MODEL_TYPE, Cost=$COST_TYPE, Epsilon=$EPSILON"
            echo ""
        done
    done
done

echo "=========================================="
echo "All graph attacks completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
