#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs
DEEPSPEED_CFG="/mmfs1/gscratch/krishna/mahtab/mmseek/Qwen2.5-VL/qwen-vl-finetune/scripts/zero3.json"
# ======================
# Path Configuration
# ======================
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"  # [ModelArguments] Pretrained model path
OUTPUT_DIR="./7b_aurora"                   # Directory for saving checkpoints
CACHE_DIR="./cache"                          # [TrainingArguments] Cache directory for models
NEW_TOKENS_FILE_PATH="/mmfs1/gscratch/krishna/mahtab/mmseek/Qwen2.5-VL/New_tokens.txt"
# ======================
# Model Configuration
# ======================
DATASETS="depth_aurora%100"                 # [DataArguments] Dataset with sampling rate

# ======================
# Training Hyperparameters
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_qwen.py \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_llm True \
         --tune_mm_vision False \
         --tune_mm_mlp True \
         --tune_embeddings True \
         --new_tokens_file $NEW_TOKENS_FILE_PATH \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 \
         --per_device_train_batch_size 4 \
         --gradient_accumulation_steps 8 \
         --learning_rate 2e-7 \
         --mm_projector_lr 1e-5 \
         --vision_tower_lr 1e-6 \
         --optim adamw_torch \
         --model_max_length 4096 \
         --data_flatten False \
         --data_sequential True \
         --data_packing False \
         --max_pixels 50176 \
         --min_pixels 784 \
         --base_interval 2 \
         --num_train_epochs 1 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type "cosine" \
         --weight_decay 0 \
         --logging_steps 10 \
         --save_steps 50 \
         --save_total_limit 3 \
         --max_grad_norm 1 \
         --gradient_checkpointing True \
         --dataloader_num_workers 4 \
         --deepspeed $DEEPSPEED_CFG \