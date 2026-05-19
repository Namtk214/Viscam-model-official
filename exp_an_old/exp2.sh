#!/bin/bash
# Exp 2: TeleFraud vs BothBosu vs VisCam — two-stage pipeline (binary + multiclass)

set -e
cd "$(dirname "$0")"

PYTHON="/home/nvidia-lab/miniconda3/envs/vcs/bin/python -u"
LOGS="logs/exp2"
RESULTS="results"
mkdir -p "$LOGS" "$RESULTS"

echo "============================================================"
echo " EXP 2 [ThaiVersion]: TeleFraud vs BothBosu vs VisCam (two-stage)"
echo " Train: tele28k | bothbosu | viscam    Test: real2"
echo "============================================================"

echo -e "\n[1/3] Train: tele28k"
$PYTHON exp2.py --train-sources tele28k \
    2>&1 | tee "$LOGS/tele28k_two_stage.log"

echo -e "\n[2/3] Train: bothbosu"
$PYTHON exp2.py --train-sources bothbosu \
    2>&1 | tee "$LOGS/bothbosu_two_stage.log"

echo -e "\n[3/3] Train: viscam"
$PYTHON exp2.py --train-sources viscam \
    2>&1 | tee "$LOGS/viscam_two_stage.log"

echo -e "\n============================================================"
echo " Generating Excel comparison..."
echo "============================================================"
$PYTHON compare.py \
    --dirs outputs/exp2_tele28k_two_stage/binary_model \
            outputs/exp2_bothbosu_two_stage/binary_model \
            outputs/exp2_viscam_two_stage/binary_model \
    --out "$RESULTS/exp2_compare.xlsx"

echo -e "\nDone! Results: $RESULTS/exp2_compare.xlsx"
