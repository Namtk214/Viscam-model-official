#!/bin/bash
# Exp 1: Vanilla vs VisCam synthetic data — two-stage pipeline (binary + multiclass)
#
# Usage:
#   ./exp1.sh                # run all 2 sources
#   ./exp1.sh vanilla        # run 1 source

set -e
cd "$(dirname "$0")"

PYTHON="/home/nvidia-lab/miniconda3/envs/vcs/bin/python -u"
LOGS="logs/exp1"
RESULTS="results"
mkdir -p "$LOGS" "$RESULTS"

run_one() {
    local src=$1
    echo -e "\n[RUN] Train: $src (two-stage)"
    $PYTHON exp1.py --train-source "$src" \
        2>&1 | tee "$LOGS/${src}_two_stage.log"
}

echo "============================================================"
echo " EXP 1 [ThaiVersion]: Vanilla vs VisCam (two-stage)"
echo " Train: vanilla | viscam    Test: real2"
echo "============================================================"

if [[ $# -eq 1 ]]; then
    run_one "$1"
elif [[ $# -eq 0 ]]; then
    run_one vanilla
    run_one viscam

    echo -e "\n============================================================"
    echo " Generating Excel comparison..."
    echo "============================================================"
    $PYTHON compare.py \
        --dirs outputs/exp1_vanilla_two_stage/binary_model \
                outputs/exp1_viscam_two_stage/binary_model \
        --out "$RESULTS/exp1_compare.xlsx"

    echo -e "\nDone! Results: $RESULTS/exp1_compare.xlsx"
else
    echo "Usage: $0 [train-source]"
    echo "  train-source: vanilla | viscam | both"
    exit 1
fi
