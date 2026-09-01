#!/bin/bash

# --- Get the project root directory ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )

# --- Experiment Configuration ---
BASE_LOG_DIR="${PROJECT_ROOT}/runs/final_experiment"

PARTICIPANTS_TRAIN="${PROJECT_ROOT}/data/ratings/Participants_control_train.csv"
PARTICIPANTS_TEST="${PROJECT_ROOT}/data/ratings/Participants_control_test.csv"


# Define paths for the saved model weights
SSP_WEIGHTS_PATH="${BASE_LOG_DIR}/ssp_model.pt"
SUP_WEIGHTS_PATH="${BASE_LOG_DIR}/supervised_model.pt"

# Create the main directory for this experiment's runs
mkdir -p "$BASE_LOG_DIR"

# --- Phase 1: Self-Supervised Pre-training (SSP) ---
echo "--- Starting Phase 1: Self-Supervised Pre-training ---"
NPROC=${NPROC:-$(python -c "import torch; print(torch.cuda.device_count())")}
LAUNCHER="python"
if command -v torchrun >/dev/null 2>&1 && [ "$NPROC" -gt 1 ]; then
  LAUNCHER="torchrun --standalone --nproc_per_node=$NPROC"
  MASTER_PORT_ARG=""
else
  MASTER_PORT=$(( 20000 + RANDOM % 10000 ))
  MASTER_PORT_ARG="--master_port $MASTER_PORT"
fi
$LAUNCHER "${PROJECT_ROOT}/src/train.py" \
    --run_ssp \
    --log_dir "${BASE_LOG_DIR}/ssp_logs" \
    --ssp_weights_path "$SSP_WEIGHTS_PATH" \
    --ssp_on_all_data \
    $MASTER_PORT_ARG


# --- Phase 2: Supervised Fine-tuning (SFT) ---
echo -e "\n--- Starting Phase 2: Supervised Fine-tuning ---"
# Prefer torchrun for clean DDP startup/shutdown
# Fallback to plain python if torchrun is unavailable.
NPROC=${NPROC:-$(python -c "import torch; print(torch.cuda.device_count())")}
LAUNCHER="python"
if command -v torchrun >/dev/null 2>&1 && [ "$NPROC" -gt 1 ]; then
  LAUNCHER="torchrun --standalone --nproc_per_node=$NPROC"
  MASTER_PORT_ARG=""
else
  MASTER_PORT=$(( 20000 + RANDOM % 10000 ))
  MASTER_PORT_ARG="--master_port $MASTER_PORT"
fi
$LAUNCHER "${PROJECT_ROOT}/src/train.py" \
    --run_supervised \
    --log_dir "${BASE_LOG_DIR}/supervised_logs" \
    --ssp_weights_path "$SSP_WEIGHTS_PATH" \
    --sup_weights_path "$SUP_WEIGHTS_PATH" \
    --participants_train "$PARTICIPANTS_TRAIN" \
    --participants_test "$PARTICIPANTS_TEST" \
    --use_separate_test_set \
    $MASTER_PORT_ARG


# --- Phase 3: Test-time Adaptation (TTA) and Evaluation ---
echo -e "\n--- Starting Phase 3: Test-time Adaptation and Evaluation ---"
MASTER_PORT=$(( 20000 + RANDOM % 10000 ))
python "${PROJECT_ROOT}/src/train.py" \
    --run_final_evaluation \
    --log_dir "${BASE_LOG_DIR}/final_evaluation_logs" \
    --sup_weights_path "$SUP_WEIGHTS_PATH" \
    --participants_train "$PARTICIPANTS_TRAIN" \
    --participants_test "$PARTICIPANTS_TEST" \
    --use_separate_test_set \
    --master_port "$MASTER_PORT" \
    --adaptation \
