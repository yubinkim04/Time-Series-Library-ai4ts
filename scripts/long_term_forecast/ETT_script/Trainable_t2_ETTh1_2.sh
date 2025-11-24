#!/usr/bin/env bash
set -Eeuo pipefail

# GPU to use (edit if needed)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# --------- Static config ---------
model_name="${1:-Trainable_t2}"

ROOT="./dataset/ETT-small"
DATA_FILE="ETTh1.csv"
DATA_NAME="ETTh1"

FEATURES="M"
SEQ_LENS=(192 336 720)
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
  "per_head"
)

# non-RoPE base configs A, B, D, F:
# A: -3.0  0.02 0.02
# B: -3.5  0.05 0.02
# D: -3.0  0.30 0.05
# F: -2.5  0.30 0.05
NONROPE_CONFIGS=(
  "-3.0 0.02 0.02"
  "-3.5 0.05 0.02"
  "-3.0 0.30 0.05"
  "-2.5 0.30 0.05"
)

echo "Starting top-7 sweep for ${DATA_NAME} on model=${model_name}"
echo "Pred lens: ${PRED_LENS[*]}"
echo "SyPE share_modes: ${SHARE_MODES[*]}"
echo "SyPE nonrope configs (log_mean, log_std, rho_std):"
for cfg in "${NONROPE_CONFIGS[@]}"; do
  echo "  ${cfg}"
done

NONROPE_CFG_COUNT=${#NONROPE_CONFIGS[@]}
echo "Total runs planned (upper bound): $(( ${#PRED_LENS[@]} * ${#CONFIGS[@]} * ${#SHARE_MODES[@]} * ${NONROPE_CFG_COUNT} ))"

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
        for NONROPE_CFG in "${NONROPE_CONFIGS[@]}"; do

          # Parse non-RoPE hyperparameters: log_mean log_std rho_std
          read -r NONROPE_LOG_MEAN NONROPE_LOG_STD NONROPE_RHO_STD <<< "${NONROPE_CFG}"

          # Build extra args for SyPE
          EXTRA_ARGS=(
            --share_mode "${SHARE_MODE}"
            --nonrope_init
            --nonrope_log_mean "${NONROPE_LOG_MEAN}"
            --nonrope_log_std "${NONROPE_LOG_STD}"
            --nonrope_rho_std "${NONROPE_RHO_STD}"
          )

          # Use values themselves in group/model names
          GROUP="${DATA_NAME}_S${SEQ_LEN}_P${PRED_LEN}_top7_sype-${SHARE_MODE}_nr_m${NONROPE_LOG_MEAN}_s${NONROPE_LOG_STD}_r${NONROPE_RHO_STD}"
          MODEL_ID="New_run_${DATA_NAME}_S${SEQ_LEN}_P${PRED_LEN}_el${E_LAYERS}_dm${D_MODEL}_dff${D_FF}_do${DROPOUT}_lr${LEARNING_RATE}_h${N_HEADS}_sype-${SHARE_MODE}_nr_m${NONROPE_LOG_MEAN}_s${NONROPE_LOG_STD}_r${NONROPE_RHO_STD}"

          echo "============================================================"
          echo "▶️  cfg#${cfg_idx}  PRED_LEN=${PRED_LEN} | SEQ_LEN=${SEQ_LEN}"
          echo "    SyPE: share_mode=${SHARE_MODE}, nonrope_init=1"
          echo "    SyPE nonrope: log_mean=${NONROPE_LOG_MEAN}, log_std=${NONROPE_LOG_STD}, rho_std=${NONROPE_RHO_STD}"
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
