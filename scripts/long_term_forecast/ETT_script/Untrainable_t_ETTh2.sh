#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: ./sweep_etth2.sh [ModelName]
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
model_name="${1:-Untrainable_t}"

# Dataset/config fixed for this sweep
ROOT="./dataset/ETT-small"
DATA_FILE="ETTh2.csv"
DATA_NAME="ETTh2"
FEATURES="M"

SEQ_LEN=720
LABEL_LEN=48
PRED_LENS=(96 192 336 720)

D_LAYERS=1
FACTOR=3
ENC_IN=7
DEC_IN=7
C_OUT=7

TRAIN_EPOCHS=50
PATIENCE=10

# The requested configs (deduplicated):
# Fields: E_LAYERS D_MODEL D_FF DROPOUT LEARNING_RATE N_HEADS CFG_IDX LRM_TAG
CONFIGS=(
"6 16 32  0.20 0.00010000 4 3 1.0"
"6 16 32  0.20 0.00040000 4 2 2.0"
"6 16 32  0.20 0.00020000 4 2 1.0"
"6 16 32  0.20 0.00020000 4 3 2.0"
"6 16 256 0.40 0.00010000 2 1 2.0"
)

echo "Starting targeted sweep for ${DATA_NAME} on model=${model_name}"
echo "Pred lens: ${PRED_LENS[*]} | Seq len: ${SEQ_LEN} | Label len: ${LABEL_LEN}"
echo "Configs: ${#CONFIGS[@]} (fixed hyperparams; sweeping P over ${PRED_LENS[*]})"

cfg_count=0
for cfg in "${CONFIGS[@]}"; do
  cfg_count=$((cfg_count + 1))
  read -r E_LAYERS D_MODEL D_FF DROPOUT LEARNING_RATE N_HEADS CFG_IDX LRM_TAG <<< "${cfg}"

  # Sanity checks (keep your originals)
  if (( D_MODEL < 2 * N_HEADS )); then
    echo "⏭️  Skip cfg#${CFG_IDX} (dm=${D_MODEL}, h=${N_HEADS}) since d_model/n_heads < 2"; continue
  fi
  if (( D_MODEL % N_HEADS != 0 )); then
    echo "⏭️  Skip cfg#${CFG_IDX} (dm=${D_MODEL}, h=${N_HEADS}) since not divisible"; continue
  fi

  for PRED_LEN in "${PRED_LENS[@]}"; do
    GROUP="${DATA_NAME}_S${SEQ_LEN}_L${LABEL_LEN}_P${PRED_LEN}_tuned"

    MODEL_ID="${DATA_NAME}_S${SEQ_LEN}_L${LABEL_LEN}_P${PRED_LEN}"\
"_el${E_LAYERS}_dm${D_MODEL}_dff${D_FF}_do$(printf '%.2f' "${DROPOUT}")"\
"_lr$(printf '%.8f' "${LEARNING_RATE}")_h${N_HEADS}_cfg${CFG_IDX}_lrm${LRM_TAG}"

    echo "============================================================"
    echo "▶️  ${DATA_NAME} cfg#${CFG_IDX}  S=${SEQ_LEN} L=${LABEL_LEN} P=${PRED_LEN}  do=$(printf '%.2f' "${DROPOUT}")  lr=$(printf '%.8f' "${LEARNING_RATE}")"
    echo "   heads=${N_HEADS} dm=${D_MODEL} dff=${D_FF} el=${E_LAYERS} | WANDB_GROUP=${GROUP}"
    echo "→  ${MODEL_ID}"
    echo "============================================================"

    WANDB_GROUP="${GROUP}" python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "${ROOT}/" \
      --data_path "${DATA_FILE}" \
      --model_id "${MODEL_ID}" \
      --model "${model_name}" \
      --data "${DATA_NAME}" \
      --features "${FEATURES}" \
      --seq_len ${SEQ_LEN} \
      --label_len ${LABEL_LEN} \
      --pred_len ${PRED_LEN} \
      --e_layers ${E_LAYERS} \
      --d_layers ${D_LAYERS} \
      --factor ${FACTOR} \
      --enc_in ${ENC_IN} \
      --dec_in ${DEC_IN} \
      --c_out ${C_OUT} \
      --d_model ${D_MODEL} \
      --d_ff ${D_FF} \
      --des "etth2_targeted_sweep" \
      --itr 1 \
      --train_epochs ${TRAIN_EPOCHS} \
      --dropout ${DROPOUT} \
      --patience ${PATIENCE} \
      --learning_rate ${LEARNING_RATE} \
      --n_heads ${N_HEADS}
  done
done

echo "✅ ETTh2 targeted sweep complete."
