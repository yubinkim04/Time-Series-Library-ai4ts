export CUDA_VISIBLE_DEVICES=0

model_name=no_rope_abl


# Optimal hyperparameters found from experiments run with Trainable_t on the solar dataset
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id no_rope_abl_solar_720_96_el2_dm64_dff32_do0.3_lr0.0005_h8 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --d_model 64 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 50 \
  --dropout 0.3 \
  --patience 10 \
  --learning_rate 0.0005 \
  --n_heads 8


