# !/bin/bash

export CUDA_VISIBLE_DEVICES=0

CKPT_PATH="/path/to/your_ckpt.pth"
PROMPT="a photo of a bench"
IMAGE_PATH="/path/to/your_image.jpg"

OUTPUT_DIR="attn_probe_outputs"
UND_MAX_NEW_TOKENS=128
GEN_IMAGE_TOKEN_NUM_PER_IMAGE=576
TOPK_PROMPT=10
HEATMAP_VMIN=0.0
HEATMAP_VMAX=0.02

mkdir -p "$OUTPUT_DIR"

python und_token_attn_probe.py \
    --ckpt_path "$CKPT_PATH" \
    --prompt "$PROMPT" \
    --image_path "$IMAGE_PATH" \
    --max_new_tokens "$UND_MAX_NEW_TOKENS" \
    --topk_prompt "$TOPK_PROMPT" \
    --heatmap_vmin "$HEATMAP_VMIN" \
    --heatmap_vmax "$HEATMAP_VMAX" \
    --output_prefix "$OUTPUT_DIR/und" &

python gen_token_attn_probe.py \
    --ckpt_path "$CKPT_PATH" \
    --prompt "$PROMPT" \
    --image_token_num_per_image "$GEN_IMAGE_TOKEN_NUM_PER_IMAGE" \
    --topk_prompt "$TOPK_PROMPT" \
    --heatmap_vmin "$HEATMAP_VMIN" \
    --heatmap_vmax "$HEATMAP_VMAX" \
    --output_prefix "$OUTPUT_DIR/gen" &

wait

echo "attention probe 已完成，请检查 $OUTPUT_DIR 目录。"
