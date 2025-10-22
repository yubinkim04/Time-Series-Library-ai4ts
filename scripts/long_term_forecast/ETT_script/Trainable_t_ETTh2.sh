#!/usr/bin/env bash
set -Eeuo pipefail

# GPU to use (edit if needed)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ---------------- Static config ----------------
model_name="${1:-Trainable_t}"

ROOT="./dataset/ETT-small"
DATA_FILE="ETTh2.csv"
DATA_NAME="ETTh2"

FEATURES="M"

# We'll sweep these because ETTh2 often wants different horizons/context
SEQ_LENS=(96 192 336 720)
LABEL_LENS=(48)

# Non-tuned fixed bits
D_LAYERS=1
FACTOR=3
ENC_IN=7
DEC_IN=7
C_OUT=7

TRAIN_EPOCHS=50
PATIENCE=10

# ---------------- Prediction lengths ----------------
PRED_LENS=(96 192 336 720)

# ---------------- Base configs (copied from your ETTh1 "top-7") ----------------
# Format: e_layers d_model d_ff dropout lr n_heads
BASE_CONFIGS=(
"6 16 256 0.3 0.00005 2"
"6 16 32  0.1 0.0002  4"
"6 16 32  0.1 0.0001  4"
"6 8  128 0.1 0.0002  2"
"5 32 128 0.1 0.0001  16"
"6 32 64  0.1 0.00005 8"
)

# ---------------- Targeted retune knobs ----------------
# Slightly stronger regularization can help ETTh2; try +0 or +0.1
DROPOUT_DELTAS=(0.0 0.1)

# Try LR ×{1, 2, 0.5} around the ETTh1 bests
LR_MULTS=(1.0 2.0 0.5)

echo "Starting targeted sweep for ${DATA_NAME} on model=${model_name}"
echo "Pred lens:      ${PRED_LENS[*]}"
echo "Seq lens grid:  ${SEQ_LENS[*]}"
echo "Label lens grid:${LABEL_LENS[*]}"
echo "LR mults:       ${LR_MULTS[*]}"
echo "Dropout deltas: ${DROPOUT_DELTAS[*]}"

# Upper bound run count (after d_model/head divisibility guards)
echo "Theoretical runs (upper bound): $(( ${#PRED_LENS[@]} * ${#SEQ_LENS[@]} * ${#LABEL_LENS[@]} * ${#BASE_CONFIGS[@]} * ${#LR_MULTS[@]} * ${#DROPOUT_DELTAS[@]} ))"

cfg_idx=0
for cfg in "${BASE_CONFIGS[@]}"; do
  cfg_idx=$((cfg_idx + 1))
  read -r E_LAYERS D_MODEL D_FF BASE_DROPOUT BASE_LR N_HEADS <<< "${cfg}"

  # Sanity: ensure d_model divisible by n_heads and >= 2*heads
  if (( D_MODEL < 2 * N_HEADS )); then
    echo "⏭️  Skip base cfg#${cfg_idx} (dm=${D_MODEL}, h=${N_HEADS}) since d_model/n_heads < 2"
    continue
  fi
  if (( D_MODEL % N_HEADS != 0 )); then
    echo "⏭️  Skip base cfg#${cfg_idx} (dm=${D_MODEL}, h=${N_HEADS}) since not divisible"
    continue
  fi

  for SEQ_LEN in "${SEQ_LENS[@]}"; do
    for LABEL_LEN in "${LABEL_LENS[@]}"; do
      # Guard: label_len should be <= seq_len (common assumption)
      if (( LABEL_LEN > SEQ_LEN )); then
        echo "⏭️  Skip (seq=${SEQ_LEN}, label=${LABEL_LEN})"
        continue
      fi

      for PRED_LEN in "${PRED_LENS[@]}"; do
        for LR_MULT in "${LR_MULTS[@]}"; do
          # Compute LR variant
          # Using awk to avoid bc dependency
          LEARNING_RATE=$(awk -v base="${BASE_LR}" -v m="${LR_MULT}" 'BEGIN { printf "%.8f", base*m }')

          for DDELTA in "${DROPOUT_DELTAS[@]}"; do
            DROPOUT=$(awk -v b="${BASE_DROPOUT}" -v d="${DDELTA}" 'BEGIN { x=b+d; if (x<0) x=0; if (x>0.5) x=0.5; printf "%.2f", x }')

            GROUP="${DATA_NAME}_S${SEQ_LEN}_L${LABEL_LEN}_P${PRED_LEN}_tuned"
            MODEL_ID="${DATA_NAME}_S${SEQ_LEN}_L${LABEL_LEN}_P${PRED_LEN}_el${E_LAYERS}_dm${D_MODEL}_dff${D_FF}_do${DROPOUT}_lr${LEARNING_RATE}_h${N_HEADS}_cfg${cfg_idx}_lrm${LR_MULT}"

            echo "============================================================"
            echo "▶️  cfg#${cfg_idx}  S=${SEQ_LEN} L=${LABEL_LEN} P=${PRED_LEN}  do=${DROPOUT}  lr=${LEARNING_RATE}"
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
      done
    done
  done
done

echo "✅ ETTh2 targeted sweep complete."
