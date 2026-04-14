import os
import os.path as osp
import hashlib
import time
import argparse
import json
import shutil
import glob
import re
import sys

import cv2
import tqdm
import torch
from transformers import AutoModelForCausalLM
from models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image
import numpy as np
from pytorch_lightning import seed_everything

from utils.run_geneval import *

# from conf import HF_TOKEN, HF_HOME

# # set environment variables
# os.environ['HF_TOKEN'] = HF_TOKEN
# os.environ['HF_HOME'] = HF_HOME
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 1,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    import numpy as np
    import torch
    import PIL.Image

    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    outputs = None  # 初始化

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

        next_token = torch.cat(
            [next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1
        ).view(-1)

        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    # ===== decode =====
    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
    )

    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    # ===== 返回 PIL Image =====
    images = [PIL.Image.fromarray(dec[i]) for i in range(parallel_size)]

    # 如果你确定 parallel_size=1，直接返回单张
    if parallel_size == 1:
        return images[0]
    else:
        return images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--metadata_file', type=str, default='/evaluation_gen/gen_eval/prompts/evaluation_metadata.jsonl')
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--load_rewrite_prompt_cache', type=int, default=1, choices=[0,1])
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    
    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    
    prompt_rewrite_cache_file = osp.join('evaluation/gen_eval', 'prompt_rewrite_cache.json')
    if osp.exists(prompt_rewrite_cache_file):
        with open(prompt_rewrite_cache_file, 'r') as f:
            prompt_rewrite_cache = json.load(f)
    else:
        prompt_rewrite_cache = {}

    if args.model_type == 'flux_1_dev':
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
    elif args.model_type == 'flux_1_dev_schnell':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")

    elif 'janus' in args.model_type:
        model_path = "/inspire/hdd/project/exploration-topic/public/downloaded_ckpts/Janus/Janus-Pro-7B"
        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, attn_implementation="eager")
        tokenizer = vl_chat_processor.tokenizer

        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )

        finetune_ckpt = torch.load(args.ckpt_path)
        vl_gpt.load_state_dict(finetune_ckpt)
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

        
    for index, metadata in enumerate(metadatas):
        seed_everything(args.seed)
        outpath = os.path.join(args.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)
        prompt = metadata['prompt']
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        tau = args.tau
        cfg = args.cfg

        images = []

        if 'janus' in args.model_type:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": prompt,
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=vl_chat_processor.sft_format,
                system_prompt="",
            )
            prompt = sft_format + vl_chat_processor.image_start_tag
        else:
            raise ValueError
        for sample_j in range(args.n_samples):
            print(f"Generating {sample_j+1} of {args.n_samples}, prompt={prompt}")
            t1 = time.time()

            if 'janus' in args.model_type:
                image = generate(
                    vl_gpt,
                    vl_chat_processor,
                    prompt,
                )
            else:
                raise ValueError

            t2 = time.time()
            print(f'{args.model_type} infer one image takes {t2-t1:.2f}s')
            images.append(image)
        for i, image in enumerate(images):
            save_file = os.path.join(sample_path, f"{i:05}.jpg")
            if 'infinity' in args.model_type:
                cv2.imwrite(save_file, image.cpu().numpy())
            else:
                image.save(save_file)


