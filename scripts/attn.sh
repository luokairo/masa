# !/bin/bash

export CUDA_VISIBLE_DEVICES=0

CKPT_PATH="/path/to/your_ckpt.pth"
PROMPT="roman temple ruin, soft grey and blue natural light, crepuscular rays, intricate, digital painting, illustration, art by greg rutkowski and luis rollo and uang guangjian and gil elvgren, symmetry"
UND_PROMPT="Describe this image"
IMAGE_PATH="/inspire/hdd/project/exploration-topic/public/ent/NIPS/masa/generated_samples/img_11_0.jpg"

OUTPUT_DIR="attn_probe_outputs"
UND_MAX_NEW_TOKENS=128
GEN_IMAGE_TOKEN_NUM_PER_IMAGE=576
TOPK_PROMPT=10
HEATMAP_VMIN=0.0
HEATMAP_VMAX=0.02

mkdir -p "$OUTPUT_DIR"

# python und_token_attn_probe.py \
#     --ckpt_path "$CKPT_PATH" \
#     --prompt "$UND_PROMPT" \
#     --image_path "$IMAGE_PATH" \
#     --max_new_tokens "$UND_MAX_NEW_TOKENS" \
#     --topk_prompt "$TOPK_PROMPT" \
#     --heatmap_vmin "$HEATMAP_VMIN" \
#     --heatmap_vmax "$HEATMAP_VMAX" \
#     --output_prefix "$OUTPUT_DIR/und" &

# python /inspire/hdd/project/exploration-topic/public/ent/NIPS/masa/qwen2.5_und_attn.py \
#     --model_path /inspire/hdd/project/exploration-topic/public/downloaded_ckpts/Qwen2.5-VL-3B-Instruct \
#     --prompt "$UND_PROMPT" \
#     --image_path "$IMAGE_PATH" \
#     --heatmap_vmin "$HEATMAP_VMIN" \
#     --heatmap_vmax "$HEATMAP_VMAX" \
#     --output_prefix "$OUTPUT_DIR/qwen25_probe " &


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