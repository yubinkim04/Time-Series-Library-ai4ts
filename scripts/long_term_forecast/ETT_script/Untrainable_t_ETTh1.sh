#!/usr/bin/env bash
set -Eeuo pipefail

# GPU to use (edit if needed)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# --------- Static config ---------
model_name="${1:-Untrainable_t}"

ROOT="./dataset/ETT-small"
DATA_FILE="ETTh1.csv"
DATA_NAME="ETTh1"

FEATURES="M"
SEQ_LEN=96
LABEL_LEN=48

# Non-tuned fixed bits
D_LAYERS=1
FACTOR=3
ENC_IN=7
DEC_IN=7
C_OUT=7

TRAIN_EPOCHS=50
PATIENCE=10

# --------- Prediction lengths (standard 4) ---------
PRED_LENS=(96 192 336 720)

# --------- Top-7 configs: el dm dff dropout lr n_heads ---------
CONFIGS=(
"6 16 256 0.3 0.00005 2"
"6 16 32  0.1 0.0002  4"
"6 16 32  0.1 0.0001  4"
"6 8  128 0.1 0.0002  2"
"5 32 128 0.1 0.0001  16"
"6 32 64  0.1 0.00005 8"
)

echo "Starting top-7 sweep for ${DATA_NAME} on model=${model_name}"
echo "Pred lens: ${PRED_LENS[*]}"
echo "Total runs planned (upper bound): $(( ${#PRED_LENS[@]} * ${#CONFIGS[@]} ))"

cfg_idx=0
for cfg in "${CONFIGS[@]}"; do
  cfg_idx=$((cfg_idx + 1))

  # Parse: EL D_MODEL D_FF DROPOUT LR N_HEADS
  read -r E_LAYERS D_MODEL D_FF DROPOUT LEARNING_RATE N_HEADS <<< "${cfg}"

  # Sanity constraints that would otherwise error out
  if (( D_MODEL < 2 * N_HEADS )); then
    echo "⏭️  Skipping config#${cfg_idx} (dm=${D_MODEL}, h=${N_HEADS}) since d_model/n_heads < 2"
    continue
  fi
  if (( D_MODEL % N_HEADS != 0 )); then
    echo "⏭️  Skipping config#${cfg_idx} (dm=${D_MODEL}, h=${N_HEADS}) since not divisible"
    continue
  fi

  for PRED_LEN in "${PRED_LENS[@]}"; do
    GROUP="${DATA_NAME}_S${SEQ_LEN}_P${PRED_LEN}_top7"
    MODEL_ID="${DATA_NAME}_${SEQ_LEN}_${PRED_LEN}_el${E_LAYERS}_dm${D_MODEL}_dff${D_FF}_do${DROPOUT}_lr${LEARNING_RATE}_h${N_HEADS}"

    echo "============================================================"
    echo "▶️  cfg#${cfg_idx}  PRED_LEN=${PRED_LEN} | WANDB_GROUP=${GROUP}"
    echo "→ Running ${MODEL_ID}"
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
      --des "top7_predlen_sweep" \
      --itr 1 \
      --train_epochs ${TRAIN_EPOCHS} \
      --dropout ${DROPOUT} \
      --patience ${PATIENCE} \
      --learning_rate ${LEARNING_RATE} \
      --n_heads ${N_HEADS}
  done
done

echo "✅ Top-7 × pred_len sweep complete."
