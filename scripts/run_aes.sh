# !/bin/bash

export CUDA_VISIBLE_DEVICES=0

CKPT_PATH="/inspire/hdd/project/exploration-topic/public/ent/NIPS/ckpt/t2i_generation/14208860000.0/ckpt/iter_13999.pth"
PROMPT="a photo of a bench"

# 5. 运行 Python 脚本
# python /inspire/hdd/project/exploration-topic/public/ent/TIFO/generation_inference.py \
#     --ckpt_path "$CKPT_PATH" \
#     --prompt "$PROMPT"

python /inspire/hdd/project/exploration-topic/public/ent/NIPS/masa/in_parallel.py \
    --ckpt_path "$CKPT_PATH" \
    --prompts_file /inspire/hdd/project/exploration-topic/public/ent/TIFO/prompt.txt

echo "生成任务已完成，请检查 generated_samples 目录。"