#!/bin/bash

RUN_NAME="models/AKI"
DATASET_ROOT="$HOME/projects/Kanzo/datasets/pretraining_datasets"

torchrun --nnodes=1 --nproc_per_node=2 train/train.py \
  --model_family AKI \
  --lm_path microsoft/Phi-3.5-mini-instruct \
  --tokenizer_path microsoft/Phi-3.5-mini-instruct \
  --vision_encoder_path siglip-so400m-patch14-384 \
  --vision_encoder_pretrained google \
  --checkpoint_steps 100000000 \
  --logging_steps 1 \
  --loss next_token_prediction \
  --lr_scheduler cosine \
  --min_lr 0.00001 \
  --weight_decay 0.01 \
  --workers 8 \
  --run_name $RUN_NAME \
  --num_epochs 1000 \
  --warmup_steps 2000 \
  --precision amp_bf16 \
  --fsdp \
  --report_to_tensorboard \
  --log_dir "$HOME/$RUN_NAME/logs" \
  --batch_size_blip3_kale 2 \
  --train_num_samples_blip3_kale 16000\
  --blip3_kale_shards "$DATASET_ROOT/blip3_kale/images/{00000..01000}.tar" \
  # --batch_size_blip3_grounding_50m 2 \
  # --train_num_samples_blip3_grounding_50m 16000\
  # --blip3_grounding_50m_shards "$DATASET_ROOT/blip3_grounding_50m/images/{00000..01000}.tar" \
  # --batch_size_blip3_ocr_200m 2 \
  # --train_num_samples_blip3_ocr_200m 16000\
  # --blip3_ocr_200m_shards "$DATASET_ROOT/blip3_ocr_200m/images/{00000..01000}.tar" \
