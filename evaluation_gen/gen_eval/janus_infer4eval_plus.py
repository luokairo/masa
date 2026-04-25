import argparse
import json
import os
import os.path as osp
import time
from typing import List, Union

import cv2
import numpy as np
import PIL.Image
import torch
from pytorch_lightning import seed_everything

from models import MultiModalityCausalLM, VLChatProcessor
from utils.run_geneval import add_common_arguments

from generation_inference_plus import (
    build_attention_latent_prefix,
    build_generation_prompt,
    generate as generate_plus,
    load_model as load_plus_model,
)


DEFAULT_JANUS_MODEL_PATH = "/inspire/hdd/project/exploration-topic/public/downloaded_ckpts/Janus/Janus-Pro-7B"


def scalar_cfg(cfg: Union[float, List[float]]) -> float:
    if isinstance(cfg, list):
        if len(cfg) != 1:
            raise ValueError(f"Janus plus eval expects a single cfg value, got {cfg}.")
        return float(cfg[0])
    return float(cfg)


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    raw_prompt: str,
    temperature: float = 1.0,
    parallel_size: int = 1,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    device: torch.device = torch.device("cuda"),
    disable_plus: bool = False,
    latent_topk: int = 4,
    latent_attention_layers: int = 4,
    latent_repeats: int = 1,
    latent_cfg_mode: str = "shared",
    keep_filler_tokens: bool = False,
):
    latent_embeds = None
    latent_debug = None
    if not disable_plus:
        latent_embeds, latent_debug = build_attention_latent_prefix(
            mmgpt=mmgpt,
            vl_chat_processor=vl_chat_processor,
            prompt=raw_prompt,
            device=device,
            parallel_size=parallel_size,
            topk=latent_topk,
            attention_layers=latent_attention_layers,
            repeats=latent_repeats,
            cfg_mode=latent_cfg_mode,
            keep_filler_tokens=keep_filler_tokens,
        )

    images, _ = generate_plus(
        mmgpt=mmgpt,
        vl_chat_processor=vl_chat_processor,
        prompt=prompt,
        latent_embeds=latent_embeds,
        temperature=temperature,
        parallel_size=parallel_size,
        cfg_weight=cfg_weight,
        image_token_num_per_image=image_token_num_per_image,
        img_size=img_size,
        patch_size=patch_size,
        device=device,
    )
    pil_images = [PIL.Image.fromarray(images[i]) for i in range(images.shape[0])]
    return pil_images[0] if parallel_size == 1 else pil_images, latent_debug


def save_latent_debug(outpath: str, latent_debug) -> None:
    if latent_debug is None:
        return
    with open(os.path.join(outpath, "latent_debug.json"), "w", encoding="utf-8") as fp:
        json.dump(latent_debug, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="/evaluation_gen/gen_eval/prompts/evaluation_metadata.jsonl",
    )
    parser.add_argument("--rewrite_prompt", type=int, default=0, choices=[0, 1])
    parser.add_argument("--load_rewrite_prompt_cache", type=int, default=1, choices=[0, 1])
    parser.add_argument("--ckpt_path", type=str, required=True)

    parser.add_argument("--janus_model_path", type=str, default=DEFAULT_JANUS_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        help="Must be eager for the plus attention-search prefix.",
    )
    parser.add_argument("--disable_plus", action="store_true")
    parser.add_argument("--latent_topk", type=int, default=4)
    parser.add_argument("--latent_attention_layers", type=int, default=4)
    parser.add_argument("--latent_repeats", type=int, default=1)
    parser.add_argument("--semantic_repeats", type=int, default=None)
    parser.add_argument(
        "--latent_cfg_mode",
        type=str,
        default="shared",
        choices=["conditional_only", "shared"],
    )
    parser.add_argument("--keep_filler_tokens", action="store_true")
    parser.add_argument("--save_latent_debug", action="store_true")
    args = parser.parse_args()

    if args.semantic_repeats is not None:
        args.latent_repeats = args.semantic_repeats

    args.cfg = list(map(float, args.cfg.split(",")))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    cfg_weight = scalar_cfg(args.cfg)

    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    if "janus" not in args.model_type:
        raise ValueError("janus_infer4eval_plus.py only supports Janus model_type.")

    args.model_path = args.janus_model_path
    args.dtype = "bfloat16" if args.bf16 else "float16"

    vl_gpt, vl_chat_processor = load_plus_model(args)
    device = torch.device(args.device)

    for index, metadata in enumerate(metadatas):
        seed_everything(args.seed)
        outpath = os.path.join(args.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        raw_prompt = metadata["prompt"]
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{raw_prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        prompt = build_generation_prompt(vl_chat_processor, raw_prompt)

        latent_embeds = None
        latent_debug = None
        if not args.disable_plus:
            latent_embeds, latent_debug = build_attention_latent_prefix(
                mmgpt=vl_gpt,
                vl_chat_processor=vl_chat_processor,
                prompt=raw_prompt,
                device=device,
                parallel_size=1,
                topk=args.latent_topk,
                attention_layers=args.latent_attention_layers,
                repeats=args.latent_repeats,
                cfg_mode=args.latent_cfg_mode,
                keep_filler_tokens=args.keep_filler_tokens,
            )
            print(
                "[plus] latent embedding-prefix slots: "
                f"{latent_debug['latent_slots']} "
                f"(topk={args.latent_topk}, repeats={args.latent_repeats}, "
                f"cfg_mode={args.latent_cfg_mode})"
            )
            if args.save_latent_debug:
                save_latent_debug(outpath, latent_debug)

        images = []
        for sample_j in range(args.n_samples):
            print(f"Generating {sample_j + 1} of {args.n_samples}, prompt={prompt}")
            t1 = time.time()

            image_array, _ = generate_plus(
                mmgpt=vl_gpt,
                vl_chat_processor=vl_chat_processor,
                prompt=prompt,
                latent_embeds=latent_embeds,
                temperature=args.tau,
                parallel_size=1,
                cfg_weight=cfg_weight,
                device=device,
            )
            image = PIL.Image.fromarray(image_array[0])

            t2 = time.time()
            print(f"{args.model_type}_plus infer one image takes {t2 - t1:.2f}s")
            images.append(image)

        for i, image in enumerate(images):
            save_file = os.path.join(sample_path, f"{i:05}.jpg")
            if "infinity" in args.model_type:
                cv2.imwrite(save_file, image.cpu().numpy())
            else:
                image.save(save_file)
