# Copyright (c) 2023-2024 DeepSeek.
#
# Training-free inference-time latent KV-cache augmentation for Janus generation.

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import PIL.Image
import torch
from transformers import AutoConfig, AutoModelForCausalLM
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None

from models import MultiModalityCausalLM, VLChatProcessor


DEFAULT_MODEL_PATH = "/inspire/hdd/project/exploration-topic/public/downloaded_ckpts/Janus/Janus-Pro-7B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--outdir", type=str, default="generated_samples_plus")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg_weight", type=float, default=5.0)
    parser.add_argument("--parallel_size", type=int, default=1)
    parser.add_argument("--image_token_num_per_image", type=int, default=576)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        help="Use eager because latent search needs output_attentions.",
    )

    parser.add_argument(
        "--disable_plus",
        action="store_true",
        help="Run the same generation loop without latent KV augmentation.",
    )
    parser.add_argument(
        "--latent_topk",
        type=int,
        default=16,
        help="Number of prompt-token embedding slots selected by understanding attention search.",
    )
    parser.add_argument(
        "--latent_attention_layers",
        type=int,
        default=4,
        help="Average the last N layers' attentions for latent token search.",
    )
    parser.add_argument(
        "--latent_repeats",
        type=int,
        default=1,
        help="Repeat selected latent embedding slots to strengthen the semantic prefix.",
    )
    parser.add_argument(
        "--semantic_repeats",
        type=int,
        default=None,
        help="Deprecated alias of --latent_repeats.",
    )
    parser.add_argument(
        "--latent_cfg_mode",
        type=str,
        default="shared",
        choices=["conditional_only", "shared"],
        help="Use selected latent prefix only for CFG conditional rows, or share it to both rows.",
    )
    parser.add_argument(
        "--keep_filler_tokens",
        action="store_true",
        help="Keep articles/function words in latent search candidates.",
    )
    parser.add_argument(
        "--save_latent_debug",
        action="store_true",
        help="Save selected token ids/scores for analysis. This is not used as generation text.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_generation_prompt(vl_chat_processor: VLChatProcessor, prompt: str) -> str:
    conversation = [
        {"role": "<|User|>", "content": prompt},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return sft_format + vl_chat_processor.image_start_tag


def build_understanding_prompt(vl_chat_processor: VLChatProcessor, prompt: str) -> str:
    query = (
        "Search the image prompt for visual semantics. Focus on objects, attributes, "
        "counts, spatial relations, colors, style, lighting, and actions. Ignore "
        "punctuation and filler words.\n\n"
        f"Image prompt: {prompt}\n\n"
        "Visual semantic slots:"
    )
    conversation = [
        {"role": "<|User|>", "content": query},
        {"role": "<|Assistant|>", "content": ""},
    ]
    return vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )


def strip_edge_special_ids(token_ids: Sequence[int], tokenizer) -> List[int]:
    special_ids = {
        token_id
        for token_id in (
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
        )
        if token_id is not None
    }
    ids = list(token_ids)
    while ids and ids[0] in special_ids:
        ids.pop(0)
    while ids and ids[-1] in special_ids:
        ids.pop()
    return ids


def find_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> Optional[Tuple[int, int]]:
    if not pattern or len(pattern) > len(sequence):
        return None
    pattern = list(pattern)
    for start in range(0, len(sequence) - len(pattern) + 1):
        if list(sequence[start : start + len(pattern)]) == pattern:
            return start, start + len(pattern)
    return None


def find_prompt_span(input_ids: Sequence[int], prompt: str, tokenizer) -> Tuple[int, int]:
    prompt_ids = strip_edge_special_ids(tokenizer.encode(prompt), tokenizer)
    span = find_subsequence(input_ids, prompt_ids)
    if span is not None:
        return span

    # Tokenizers sometimes change whitespace at the prompt boundary.
    prompt_ids = strip_edge_special_ids(tokenizer.encode(" " + prompt), tokenizer)
    span = find_subsequence(input_ids, prompt_ids)
    if span is not None:
        return span

    # Conservative fallback: search all non-special tokens except the final assistant cue.
    return 0, max(1, len(input_ids) - 1)


FILLER_TOKENS = {
    "a",
    "an",
    "the",
    "of",
    "and",
    "or",
    "to",
    "with",
    "in",
    "on",
    "at",
    "by",
    "for",
    "from",
    "is",
    "are",
    "be",
}


def decode_token_text(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=True)
    return text.replace("▁", "").replace("Ġ", "").strip()


def is_content_token(tokenizer, token_id: int, keep_filler_tokens: bool) -> bool:
    text = decode_token_text(tokenizer, token_id)
    if not keep_filler_tokens and text.lower() in FILLER_TOKENS:
        return False
    return bool(text) and any(ch.isalnum() for ch in text)


def aggregate_query_attention(
    attentions: Sequence[torch.Tensor],
    num_layers: int,
    attn_implementation: Optional[str] = None,
) -> torch.Tensor:
    if not attentions:
        raise RuntimeError(
            "No attentions were returned. The inner language model is probably not using "
            f"eager attention. Current config: {attn_implementation!r}. "
            "Run with --attn_implementation eager."
        )
    selected = attentions[-max(1, num_layers) :]
    scores = []
    for attn in selected:
        # attn: [batch, heads, query_len, key_len]. Use the final query token.
        scores.append(attn[0, :, -1, :].float().mean(dim=0))
    return torch.stack(scores, dim=0).mean(dim=0)


def select_latent_positions(
    scores: torch.Tensor,
    input_ids: Sequence[int],
    span: Tuple[int, int],
    tokenizer,
    topk: int,
    repeats: int,
    keep_filler_tokens: bool,
) -> Tuple[List[int], List[Dict[str, object]]]:
    start, end = span
    candidate_positions = [
        pos
        for pos in range(start, end)
        if is_content_token(tokenizer, int(input_ids[pos]), keep_filler_tokens)
    ]
    if not candidate_positions:
        candidate_positions = list(range(start, end))

    if not candidate_positions:
        raise RuntimeError("No candidate prompt tokens found for latent attention search.")

    candidate_scores = scores[torch.tensor(candidate_positions, device=scores.device)]
    k = min(max(1, topk), len(candidate_positions))
    top_indices = torch.topk(candidate_scores, k=k).indices.detach().cpu().tolist()
    top_positions = sorted(candidate_positions[i] for i in top_indices)

    repeats = max(1, repeats)
    positions = []
    for _ in range(repeats):
        positions.extend(top_positions)

    debug_rows = []
    score_cpu = scores.detach().float().cpu()
    for pos in top_positions:
        debug_rows.append(
            {
                "position": int(pos),
                "token_id": int(input_ids[pos]),
                "token": decode_token_text(tokenizer, int(input_ids[pos])),
                "score": float(score_cpu[pos]),
            }
        )
    return positions, debug_rows


def past_key_values_to_legacy(past_key_values):
    if isinstance(past_key_values, (tuple, list)):
        return past_key_values
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    raise TypeError(
        "Unsupported past_key_values type for KV slicing: "
        f"{type(past_key_values).__name__}. Expected tuple/list or a cache with "
        "to_legacy_cache()."
    )


def past_key_values_to_model_cache(past_key_values):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, (tuple, list)) and DynamicCache is not None:
        return DynamicCache.from_legacy_cache(past_key_values)
    return past_key_values


def slice_past_key_values(past_key_values, positions: Sequence[int]):
    past_key_values = past_key_values_to_legacy(past_key_values)
    selected = torch.tensor(positions, dtype=torch.long, device=past_key_values[0][0].device)
    sliced_layers = []
    for layer_past in past_key_values:
        key, value = layer_past[:2]
        key = key.index_select(dim=2, index=selected)
        value = value.index_select(dim=2, index=selected)
        if len(layer_past) > 2:
            sliced_layers.append((key, value, *layer_past[2:]))
        else:
            sliced_layers.append((key, value))
    return tuple(sliced_layers)


def make_cfg_tokens(
    vl_chat_processor: VLChatProcessor,
    text: str,
    parallel_size: int,
    device: torch.device,
) -> torch.LongTensor:
    input_ids = torch.LongTensor(vl_chat_processor.tokenizer.encode(text)).to(device)
    tokens = torch.empty((parallel_size * 2, input_ids.numel()), dtype=torch.long, device=device)
    tokens[:, :] = input_ids
    if input_ids.numel() > 2:
        tokens[1::2, 1:-1] = vl_chat_processor.pad_id
    return tokens


def prefill_from_tokens(
    mmgpt: MultiModalityCausalLM,
    tokens: torch.LongTensor,
    past_key_values=None,
):
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    return prefill_from_embeds(mmgpt, inputs_embeds, past_key_values=past_key_values)


def prefill_from_embeds(
    mmgpt: MultiModalityCausalLM,
    inputs_embeds: torch.Tensor,
    past_key_values=None,
):
    past_key_values = past_key_values_to_model_cache(past_key_values)
    return mmgpt.language_model.model(
        inputs_embeds=inputs_embeds,
        use_cache=True,
        past_key_values=past_key_values,
    )


def expand_latent_past_for_cfg(
    latent_past,
    neutral_past,
    parallel_size: int,
    cfg_mode: str,
):
    latent_past = past_key_values_to_legacy(latent_past)
    neutral_past = past_key_values_to_legacy(neutral_past)
    expanded_layers = []
    for latent_layer, neutral_layer in zip(latent_past, neutral_past):
        latent_key, latent_value = latent_layer[:2]
        neutral_key, neutral_value = neutral_layer[:2]

        key_rows = []
        value_rows = []
        for _ in range(parallel_size):
            key_rows.append(latent_key)
            value_rows.append(latent_value)
            if cfg_mode == "shared":
                key_rows.append(latent_key)
                value_rows.append(latent_value)
            elif cfg_mode == "conditional_only":
                key_rows.append(neutral_key)
                value_rows.append(neutral_value)
            else:
                raise ValueError(f"Unsupported latent_cfg_mode: {cfg_mode}")

        key = torch.cat(key_rows, dim=0)
        value = torch.cat(value_rows, dim=0)
        if len(latent_layer) > 2:
            expanded_layers.append((key, value, *latent_layer[2:]))
        else:
            expanded_layers.append((key, value))
    return tuple(expanded_layers)


def build_cfg_latent_embeds(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    selected_token_ids: Sequence[int],
    parallel_size: int,
    cfg_mode: str,
    device: torch.device,
) -> torch.Tensor:
    latent_tokens = torch.LongTensor(selected_token_ids).unsqueeze(0).to(device)
    neutral_tokens = torch.full_like(latent_tokens, fill_value=vl_chat_processor.pad_id)

    embedding = mmgpt.language_model.get_input_embeddings()
    latent_embeds = embedding(latent_tokens)
    neutral_embeds = embedding(neutral_tokens)

    rows = []
    for _ in range(parallel_size):
        rows.append(latent_embeds)
        if cfg_mode == "shared":
            rows.append(latent_embeds)
        elif cfg_mode == "conditional_only":
            rows.append(neutral_embeds)
        else:
            raise ValueError(f"Unsupported latent_cfg_mode: {cfg_mode}")
    return torch.cat(rows, dim=0)


@torch.inference_mode()
def build_attention_latent_prefix(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    device: torch.device,
    parallel_size: int,
    topk: int,
    attention_layers: int,
    repeats: int,
    cfg_mode: str,
    keep_filler_tokens: bool,
):
    tokenizer = vl_chat_processor.tokenizer
    understanding_prompt = build_understanding_prompt(vl_chat_processor, prompt)
    input_ids_list = tokenizer.encode(understanding_prompt)
    input_ids = torch.LongTensor(input_ids_list).unsqueeze(0).to(device)

    outputs = mmgpt.language_model.model(
        input_ids=input_ids,
        use_cache=True,
        output_attentions=True,
    )
    current_attn_impl = getattr(mmgpt.language_model.config, "_attn_implementation", None)
    scores = aggregate_query_attention(
        outputs.attentions,
        num_layers=attention_layers,
        attn_implementation=current_attn_impl,
    )
    span = find_prompt_span(input_ids_list, prompt, tokenizer)
    positions, debug_rows = select_latent_positions(
        scores=scores,
        input_ids=input_ids_list,
        span=span,
        tokenizer=tokenizer,
        topk=topk,
        repeats=repeats,
        keep_filler_tokens=keep_filler_tokens,
    )
    selected_token_ids = [int(input_ids_list[pos]) for pos in positions]
    latent_embeds = build_cfg_latent_embeds(
        mmgpt=mmgpt,
        vl_chat_processor=vl_chat_processor,
        selected_token_ids=selected_token_ids,
        parallel_size=parallel_size,
        cfg_mode=cfg_mode,
        device=device,
    )
    debug = {
        "latent_mode": "embedding_prefix",
        "prompt_span": [int(span[0]), int(span[1])],
        "latent_topk": int(topk),
        "latent_repeats": int(max(1, repeats)),
        "latent_slots": int(len(positions)),
        "latent_cfg_mode": cfg_mode,
        "keep_filler_tokens": bool(keep_filler_tokens),
        "selected_tokens": debug_rows,
    }
    return latent_embeds, debug


def prefill_generation_context(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    parallel_size: int,
    device: torch.device,
    latent_embeds: Optional[torch.Tensor] = None,
):
    past_key_values = None
    if latent_embeds is not None:
        latent_outputs = prefill_from_embeds(mmgpt, latent_embeds)
        past_key_values = latent_outputs.past_key_values

    prompt_tokens = make_cfg_tokens(
        vl_chat_processor=vl_chat_processor,
        text=prompt,
        parallel_size=parallel_size,
        device=device,
    )
    return prefill_from_tokens(mmgpt, prompt_tokens, past_key_values=past_key_values)


def sample_next_image_token(
    mmgpt: MultiModalityCausalLM,
    hidden_states: torch.Tensor,
    cfg_weight: float,
    temperature: float,
) -> torch.LongTensor:
    logits = mmgpt.gen_head(hidden_states[:, -1, :])
    logit_cond = logits[0::2, :]
    logit_uncond = logits[1::2, :]
    logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    latent_embeds: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    parallel_size: int = 1,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, torch.LongTensor]:
    if device is None:
        device = torch.device("cuda")

    outputs = prefill_generation_context(
        mmgpt=mmgpt,
        vl_chat_processor=vl_chat_processor,
        prompt=prompt,
        parallel_size=parallel_size,
        device=device,
        latent_embeds=latent_embeds,
    )

    generated_tokens = torch.zeros(
        (parallel_size, image_token_num_per_image),
        dtype=torch.long,
        device=device,
    )

    for step in range(image_token_num_per_image):
        next_token = sample_next_image_token(
            mmgpt=mmgpt,
            hidden_states=outputs.last_hidden_state,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )
        generated_tokens[:, step] = next_token.squeeze(dim=-1)

        if step == image_token_num_per_image - 1:
            break

        cfg_next_token = next_token.repeat_interleave(2, dim=0).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(cfg_next_token).unsqueeze(dim=1)
        outputs = mmgpt.language_model.model(
            inputs_embeds=img_embeds,
            use_cache=True,
            past_key_values=past_key_values_to_model_cache(outputs.past_key_values),
        )

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    images = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return images, generated_tokens


def set_attention_config(config, attn_implementation: str) -> None:
    configs = [config]
    language_config = getattr(config, "language_config", None)
    if language_config is not None:
        configs.append(language_config)

    for cfg in configs:
        for attr in ("_attn_implementation", "attn_implementation"):
            try:
                setattr(cfg, attr, attn_implementation)
            except Exception:
                pass


def force_attention_implementation(mmgpt: MultiModalityCausalLM, attn_implementation: str) -> None:
    """Nested Janus configs do not always receive from_pretrained attn kwargs."""
    modules = [
        mmgpt,
        getattr(mmgpt, "language_model", None),
        getattr(getattr(mmgpt, "language_model", None), "model", None),
    ]
    for module in modules:
        if module is None:
            continue
        if hasattr(module, "set_attn_implementation"):
            try:
                module.set_attn_implementation(attn_implementation)
            except TypeError:
                module.set_attn_implementation(attn_implementation, check_device_map=False)
            except Exception:
                pass

    configs = []
    for module in modules:
        config = getattr(module, "config", None)
        if config is not None:
            configs.append(config)
            language_config = getattr(config, "language_config", None)
            if language_config is not None:
                configs.append(language_config)

    for config in configs:
        set_attention_config(config, attn_implementation)


def load_model(args: argparse.Namespace) -> Tuple[MultiModalityCausalLM, VLChatProcessor]:
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
        args.model_path,
        attn_implementation=args.attn_implementation,
    )

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    set_attention_config(config, args.attn_implementation)
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    vl_gpt.load_state_dict(ckpt)

    force_attention_implementation(vl_gpt, args.attn_implementation)

    dtype = dtype_from_name(args.dtype)
    device = torch.device(args.device)
    vl_gpt = vl_gpt.to(dtype).to(device).eval()
    return vl_gpt, vl_chat_processor


def save_outputs(
    images: np.ndarray,
    outdir: str,
    latent_debug: Optional[Dict[str, object]],
    save_latent_debug: bool,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    for i in range(images.shape[0]):
        save_path = os.path.join(outdir, f"img_{i}.jpg")
        PIL.Image.fromarray(images[i]).save(save_path)

    if save_latent_debug and latent_debug is not None:
        with open(os.path.join(outdir, "latent_debug.json"), "w", encoding="utf-8") as f:
            json.dump(latent_debug, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if args.semantic_repeats is not None:
        args.latent_repeats = args.semantic_repeats

    set_seed(args.seed)
    device = torch.device(args.device)

    vl_gpt, vl_chat_processor = load_model(args)
    generation_prompt = build_generation_prompt(vl_chat_processor, args.prompt)

    latent_embeds = None
    latent_debug = None
    if not args.disable_plus:
        latent_embeds, latent_debug = build_attention_latent_prefix(
            mmgpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            prompt=args.prompt,
            device=device,
            parallel_size=args.parallel_size,
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

    images, _ = generate(
        mmgpt=vl_gpt,
        vl_chat_processor=vl_chat_processor,
        prompt=generation_prompt,
        latent_embeds=latent_embeds,
        temperature=args.temperature,
        parallel_size=args.parallel_size,
        cfg_weight=args.cfg_weight,
        image_token_num_per_image=args.image_token_num_per_image,
        img_size=args.img_size,
        patch_size=args.patch_size,
        device=device,
    )
    save_outputs(
        images=images,
        outdir=args.outdir,
        latent_debug=latent_debug,
        save_latent_debug=args.save_latent_debug,
    )


if __name__ == "__main__":
    main()
