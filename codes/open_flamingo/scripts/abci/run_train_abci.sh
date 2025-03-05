#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -P gcf51138
#PBS -j oe
#PBS -N AKI-captions-context128
#PBS -m n
#PBS -V

# load environmental modules
module load cuda/12.6/12.6.1 cudnn/9.5/9.5.1

# load venv
source ~/venv/AKI/bin/activate

cd $PBS_O_WORKDIR

RUN_NAME="models/AKI-captions-context128"
DATASET_ROOT="/groups/gcf51138"

torchrun --nnodes=1 --nproc_per_node=8 train/train.py \
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
  --num_epochs 10 \
  --warmup_steps 2000 \
  --precision amp_bf16 \
  --fsdp \
  --report_to_tensorboard \
  --log_dir "$HOME/MFM/Kanzo_logs/$RUN_NAME/logs" \
  --batch_size_blip3_kale 24 \
  --train_num_samples_blip3_kale 1000000\
  --blip3_kale_shards "$DATASET_ROOT/blip3_kale/images/{00000..31000}.tar" \
  # --batch_size_blip3_grounding_50m 8 \
  # --train_num_samples_blip3_grounding_50m 250000\
  # --blip3_grounding_50m_shards "$DATASET_ROOT/blip3_grounding_50m/images/{00000..21000}.tar" \
  # --batch_size_blip3_ocr_200m 8 \
  # --train_num_samples_blip3_ocr_200m 250000\
  # --blip3_ocr_200m_shards "$DATASET_ROOT/blip3_ocr_200m/images/{00000..21000}.tar" \
  # --resume_from_checkpoint $HOME/Multimodal-Foundation-Models/codes/open_flamingo/models/SigLIP-384-Phi-3.5-mini/checkpoint_2.pt \

  # --batch_size_cc3m 8 \
  # --train_num_samples_cc3m 330000\
  # --cc3m_shards "$DATASET_ROOT/cc3m/{00000..00331}.tar" \
  # --batch_size_cc12m 10 \
  # --train_num_samples_cc12m 200000 \
  # --cc12m_shards "$DATASET_ROOT/cc12m/{00000..01010}.tar" \
