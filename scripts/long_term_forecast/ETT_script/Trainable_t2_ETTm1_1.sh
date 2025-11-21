#!/usr/bin/env bash
set -Eeuo pipefail

# GPU to use (edit if needed)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# --------- Static config ---------
model_name="${1:-Trainable_t2}"

ROOT="./dataset/ETT-small"
DATA_FILE="ETTm1.csv"
DATA_NAME="ETTm1"

FEATURES="M"
SEQ_LENS=(720 336)
LABEL_LEN=48

# Non-tuned fixed bits
D_LAYERS=1
FACTOR=3
ENC_IN=7
DEC_IN=7
C_OUT=7

TRAIN_EPOCHS=100
PATIENCE=10

# --------- Prediction lengths (standard 4) ---------
PRED_LENS=(96 192 336 720)

# --------- Top-7 configs: el dm dff dropout lr n_heads ---------
CONFIGS=(
"6 16 256 0.3 0.00005 2"
)

# --------- SymplecticPE (SyPE) hyperparameters to sweep ---------
# share_mode ∈ {global, per_head, per_block, per_head_block}
SHARE_MODES=(
  "per_head_block"
)

# nonrope_init: 0 = use RoPE-like base, 1 = use non-RoPE random base
NONROPE_INITS=(1)

echo "Starting top-7 sweep for ${DATA_NAME} on model=${model_name}"
echo "Pred lens: ${PRED_LENS[*]}"
echo "SyPE share_modes: ${SHARE_MODES[*]}"
echo "SyPE nonrope_init flags: ${NONROPE_INITS[*]}"
echo "Total runs planned (upper bound): $(( ${#PRED_LENS[@]} * ${#CONFIGS[@]} * ${#SHARE_MODES[@]} * ${#NONROPE_INITS[@]} ))"

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

  for SEQ_LEN in "${SEQ_LENS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
      for SHARE_MODE in "${SHARE_MODES[@]}"; do
        for NONROPE in "${NONROPE_INITS[@]}"; do

          # Build extra args for SyPE (matches run.py: --share_mode, --nonrope_init)
          EXTRA_ARGS=(
            --share_mode "${SHARE_MODE}"
          )
          if [[ "${NONROPE}" -eq 1 ]]; then
            EXTRA_ARGS+=(--nonrope_init)
          fi

          GROUP="${DATA_NAME}_S${SEQ_LEN}_P${PRED_LEN}_top7_sype-${SHARE_MODE}_nr${NONROPE}"
          MODEL_ID="New_run_${DATA_NAME}_S${SEQ_LEN}_P${PRED_LEN}_el${E_LAYERS}_dm${D_MODEL}_dff${D_FF}_do${DROPOUT}_lr${LEARNING_RATE}_h${N_HEADS}_sype-${SHARE_MODE}_nr${NONROPE}"

          echo "============================================================"
          echo "▶️  cfg#${cfg_idx}  PRED_LEN=${PRED_LEN} | SEQ_LEN=${SEQ_LEN}"
          echo "    SyPE: share_mode=${SHARE_MODE}, nonrope_init=${NONROPE}"
          echo "    WANDB_GROUP=${GROUP}"
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
            --des "top7_predlen_sweep_sype" \
            --itr 1 \
            --train_epochs ${TRAIN_EPOCHS} \
            --dropout ${DROPOUT} \
            --patience ${PATIENCE} \
            --learning_rate ${LEARNING_RATE} \
            --n_heads ${N_HEADS} \
            "${EXTRA_ARGS[@]}"
        done
      done
    done
  done
done

echo "✅ Top-7 × pred_len × SyPE sweep complete."
