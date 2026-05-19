#!/bin/bash
# Exp 3: Real (real1) vs VisCam Synthetic — two-stage pipeline (binary + multiclass)

set -e
cd "$(dirname "$0")"

PYTHON="/home/nvidia-lab/miniconda3/envs/vcs/bin/python -u"
LOGS="logs/exp3"
RESULTS="results"
mkdir -p "$LOGS" "$RESULTS"

echo "============================================================"
echo " EXP 3 [ThaiVersion]: Real (real1) vs VisCam Synthetic (two-stage)"
echo " Train: real1 | viscam    Test: real2"
echo "============================================================"

echo -e "\n[1/2] Train: viscam"
$PYTHON exp3.py --train-source viscam \
    2>&1 | tee "$LOGS/viscam_two_stage.log"

echo -e "\n[2/2] Train: real1"
$PYTHON exp3.py --train-source real1 \
    2>&1 | tee "$LOGS/real1_two_stage.log"

echo -e "\n============================================================"
echo " Generating Excel comparison..."
echo "============================================================"
$PYTHON compare.py \
    --dirs outputs/exp3_viscam_two_stage/binary_model \
            outputs/exp3_real1_two_stage/binary_model \
    --out "$RESULTS/exp3_compare.xlsx"

echo -e "\nDone! Results: $RESULTS/exp3_compare.xlsx"
