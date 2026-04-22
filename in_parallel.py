import torch
from transformers import AutoModelForCausalLM
from models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--prompts', type=str, nargs='+', required=False,
                    help='一个或多个 prompt，用空格分隔（每个 prompt 用引号包裹）')
parser.add_argument('--prompts_file', type=str, default=None,
                    help='包含 prompt 列表的文本文件，每行一个 prompt')
args = parser.parse_args()

model_path = "/inspire/hdd/project/exploration-topic/public/downloaded_ckpts/Janus/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, attn_implementation="eager")
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
finetune_ckpt = torch.load(args.ckpt_path)
vl_gpt.load_state_dict(finetune_ckpt)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    save_prefix: str = "img",       # ← 新增：用于区分不同 prompt 的文件名前缀
    temperature: float = 1,
    parallel_size: int = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    # inputs_embeds = mmgpt.text_conductor(inputs_embeds)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None
        )
        hidden_states = outputs.last_hidden_state
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('generated_samples', exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join('generated_samples', f"{save_prefix}_{i}.jpg")
        PIL.Image.fromarray(visual_img[i]).save(save_path)
        print(f"  保存图片：{save_path}")


# ── 读取 prompt 列表 ──────────────────────────────────────────
if args.prompts_file:
    with open(args.prompts_file, 'r', encoding='utf-8') as f:
        prompt_list = [line.strip() for line in f if line.strip()]
else:
    prompt_list = args.prompts

# ── 主循环 ────────────────────────────────────────────────────
for idx, prompt_text in enumerate(prompt_list):
    print(f"\n[{idx+1}/{len(prompt_list)}] 正在生成：{prompt_text}")

    conversation = [
        {"role": "<|User|>", "content": prompt_text},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    formatted_prompt = sft_format + vl_chat_processor.image_start_tag

    generate(
        vl_gpt,
        vl_chat_processor,
        formatted_prompt,
        save_prefix=f"img_{idx}",   # 每个 prompt 用独立前缀，避免覆盖
    )

print("\n全部生成完毕！")