#!/usr/bin/env bash
set -Eeuo pipefail

# GPU to use (edit if needed)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# --------- Static config ---------
model_name="${1:-Trainable_t}"

ROOT="./dataset/ETT-small"
DATA_FILE="ETTh1.csv"
DATA_NAME="ETTh1"

FEATURES="M"
SEQ_LEN=96
LABEL_LEN=48

D_LAYERS=1
FACTOR=3
ENC_IN=7
DEC_IN=7
C_OUT=7

TRAIN_EPOCHS=50
PATIENCE=10

# --------- Search space ---------
# Tuning: e_layers, d_model, d_ff, dropout, learning_rate, n_heads
ELAYERS=(3 4)
DMODEL=(64 32 16 8)
DFF=(256 128 64 32)
DROPOUT=(0.5 0.3 0.1)
LEARNING_RATE=(0.0005 0.0002 0.0001 0.00005 0.00001 )
N_HEADS=(16 8 4 2 1)

# --------- Pred lengths to sweep ---------
PRED_LENS=(96)

echo "Starting grid search for ${DATA_NAME} on model=${model_name}"
echo "Pred lens: ${PRED_LENS[*]}"
echo "Search sizes: |e_layers|=${#ELAYERS[@]} |d_model|=${#DMODEL[@]} |d_ff|=${#DFF[@]} |dropout|=${#DROPOUT[@]} |lr|=${#LEARNING_RATE[@]} |n_heads|=${#N_HEADS[@]}"
total_upper_bound=$(( ${#PRED_LENS[@]} * ${#ELAYERS[@]} * ${#DMODEL[@]} * ${#DFF[@]} * ${#DROPOUT[@]} * ${#LEARNING_RATE[@]} * ${#N_HEADS[@]} ))
echo "Total runs (upper bound): ${total_upper_bound}"

for PRED_LEN in "${PRED_LENS[@]}"; do
  GROUP="${DATA_NAME}_S${SEQ_LEN}_P${PRED_LEN}"
  echo "============================================================"
  echo "▶️  PRED_LEN=${PRED_LEN} | WANDB_GROUP=${GROUP}"
  echo "============================================================"

  for EL in "${ELAYERS[@]}"; do
    for DM in "${DMODEL[@]}"; do
      for DFFV in "${DFF[@]}"; do
        for DO in "${DROPOUT[@]}"; do
          for LR in "${LEARNING_RATE[@]}"; do
            for H in "${N_HEADS[@]}"; do

              # Skip invalid combos where d_model / n_heads < 2
              if (( DM < 2 * H )); then
                echo "⏭️  Skipping dm=${DM}, n_heads=${H} (d_model/n_heads < 2)"
                continue
              fi

              MODEL_ID="${DATA_NAME}_${SEQ_LEN}_${PRED_LEN}_el${EL}_dm${DM}_dff${DFFV}_do${DO}_lr${LR}_h${H}"
              echo "→ Running ${MODEL_ID}"

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
                --e_layers ${EL} \
                --d_layers ${D_LAYERS} \
                --factor ${FACTOR} \
                --enc_in ${ENC_IN} \
                --dec_in ${DEC_IN} \
                --c_out ${C_OUT} \
                --d_model ${DM} \
                --d_ff ${DFFV} \
                --des "grid" \
                --itr 1 \
                --train_epochs ${TRAIN_EPOCHS} \
                --dropout ${DO} \
                --patience ${PATIENCE} \
                --learning_rate ${LR} \
                --n_heads ${H}
            done
          done
        done
      done
    done
  done
done

echo "✅ Grid search complete."
