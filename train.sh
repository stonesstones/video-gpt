#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=3:00:00
#PBS -P gcg51472
#PBS -W umask=027
#PBS -N out.train_video_gpt
#PBS -j oe

# VideoGPT VQVAE Training Script
# シンプルな学習スクリプト

cd $PBS_O_WORKDIR
source /etc/profile.d/modules.sh
module load cuda/12.1/12.1.1

# 基本設定
RUN_NAME="default"
DATA_PATH="/groups/gcg51472/nuscenes"
SEQUENCE_LENGTH=16
RESOLUTION=128
BATCH_SIZE=128
GPUS=1
MAX_STEPS=200000
LOG_EVERY_N_STEPS=10
VAL_CHECK_INTERVAL=1.0


# VQVAE設定
EMBEDDING_DIM=256
N_CODES=4096
N_HIDDENS=240
N_RES_LAYERS=4

# 出力ディレクトリ
OUTPUT_DIR="./logs"
mkdir -p "$OUTPUT_DIR"

echo "=== VQVAE Training ==="
echo "Data: $DATA_PATH"
echo "Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "Sequence: $SEQUENCE_LENGTH frames"
echo "Batch size: $BATCH_SIZE"
echo "GPUs: $GPUS"
echo "Max steps: $MAX_STEPS"
echo "======================"

# 学習実行
uv run python scripts/train_vqvae.py \
    --data_path "$DATA_PATH" \
    --sequence_length "$SEQUENCE_LENGTH" \
    --resolution "$RESOLUTION" \
    --batch_size "$BATCH_SIZE" \
    --num_workers 8 \
    --gpus "$GPUS" \
    --max_steps "$MAX_STEPS" \
    --embedding_dim "$EMBEDDING_DIM" \
    --n_codes "$N_CODES" \
    --n_hiddens "$N_HIDDENS" \
    --n_res_layers "$N_RES_LAYERS" \
    --downsample 4 4 4 \
    --gradient_clip_val 1 \
    --name "$RUN_NAME" \
    --val_check_interval "$VAL_CHECK_INTERVAL" \
    --log_every_n_steps "$LOG_EVERY_N_STEPS"

echo "Training completed!" 