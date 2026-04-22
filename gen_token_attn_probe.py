import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from models import MultiModalityCausalLM, VLChatProcessor


MODEL_PATH = "/inspire/hdd/project/exploration-topic/public/downloaded_ckpts/Janus/Janus-Pro-7B"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Probe attention positions during image-token generation. "
            "The script reports which previous tokens each generated image token "
            "depends on, and whether the original prompt tokens dominate."
        )
    )
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image_token_num_per_image", type=int, default=576)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg_weight", type=float, default=5.0)
    parser.add_argument("--parallel_size", type=int, default=1)
    parser.add_argument("--topk_prompt", type=int, default=10)
    parser.add_argument("--output_prefix", type=str, default="gen_attn_probe")
    parser.add_argument("--heatmap_vmin", type=float, default=None)
    parser.add_argument("--heatmap_vmax", type=float, default=None)
    return parser.parse_args()


def ensure_output_dir(prefix: str):
    out_dir = os.path.dirname(prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def encode_text(tokenizer, text: str) -> List[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def find_subsequence(
    sequence: Sequence[int],
    subsequence: Sequence[int],
    start_hint: int = 0,
) -> Optional[Tuple[int, int]]:
    if not subsequence or len(subsequence) > len(sequence):
        return None

    last_start = len(sequence) - len(subsequence)
    candidate_ranges = [
        range(max(0, start_hint), last_start + 1),
        range(0, max(0, start_hint)),
    ]
    for candidate_range in candidate_ranges:
        for start in candidate_range:
            if list(sequence[start : start + len(subsequence)]) == list(subsequence):
                return start, start + len(subsequence)
    return None


def locate_prompt_span(
    input_ids: Sequence[int],
    tokenizer,
    prompt_text: str,
) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
    candidates = []
    for text in (
        prompt_text,
        prompt_text.strip(),
        "\n" + prompt_text,
        "\n" + prompt_text.strip(),
        " " + prompt_text,
    ):
        if text and text not in candidates:
            candidates.append(text)

    for candidate in candidates:
        candidate_ids = encode_text(tokenizer, candidate)
        span = find_subsequence(input_ids, candidate_ids, start_hint=0)
        if span is not None:
            return span, candidate
    return None, None


def format_token(tokenizer, token_id: int) -> str:
    try:
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        if isinstance(token, list):
            token = token[0]
    except Exception:
        token = None

    if token is None:
        try:
            token = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        except Exception:
            token = str(int(token_id))

    return (
        str(token)
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )


def get_source_token_info(
    position: int,
    input_ids: Sequence[int],
    generated_ids: Sequence[int],
    input_length: int,
    tokenizer,
) -> Dict[str, object]:
    if position < input_length:
        token_id = int(input_ids[position])
        source = "initial_prompt"
        rel_index = position
    else:
        rel_index = position - input_length
        token_id = int(generated_ids[rel_index])
        source = "generated_image_prefix"

    return {
        "position": int(position),
        "relative_index": int(rel_index),
        "token_id": token_id,
        "token": format_token(tokenizer, token_id),
        "source": source,
    }


def classify_position(
    position: int,
    input_length: int,
    prompt_span: Optional[Tuple[int, int]],
) -> str:
    if position >= input_length:
        return "generated_image_prefix"
    if prompt_span is not None and prompt_span[0] <= position < prompt_span[1]:
        return "prompt_text"
    return "prompt_wrapper"


def extract_step_attention(step_attentions) -> np.ndarray:
    per_layer = []
    for layer_attention in step_attentions:
        query_index = layer_attention.shape[2] - 1
        layer_vector = layer_attention[0, :, query_index, :].float().mean(dim=0)
        per_layer.append(layer_vector.detach().cpu())
    return torch.stack(per_layer, dim=0).mean(dim=0).numpy()


def effective_token_count(weights: np.ndarray) -> float:
    total = float(weights.sum())
    sq_sum = float(np.square(weights).sum())
    if total <= 0 or sq_sum <= 0:
        return 0.0
    return (total * total) / sq_sum


def top_prompt_entries(
    prompt_weights: np.ndarray,
    prompt_span: Optional[Tuple[int, int]],
    input_ids: Sequence[int],
    tokenizer,
    topk: int,
) -> List[Dict[str, object]]:
    if prompt_span is None or prompt_weights.size == 0:
        return []

    k = min(topk, prompt_weights.shape[0])
    order = np.argsort(prompt_weights)[::-1][:k]
    entries = []
    for rank, prompt_idx in enumerate(order, start=1):
        abs_pos = prompt_span[0] + int(prompt_idx)
        token_id = int(input_ids[abs_pos])
        entries.append(
            {
                "rank": rank,
                "prompt_index": int(prompt_idx),
                "absolute_position": abs_pos,
                "token_id": token_id,
                "token": format_token(tokenizer, token_id),
                "attention": float(prompt_weights[prompt_idx]),
            }
        )
    return entries


def save_heatmap(
    matrix: np.ndarray,
    output_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
    vlines: Optional[List[Tuple[float, str, str]]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    plt.figure(figsize=(14, 8))
    plt.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label="Average attention")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if vlines:
        for x, color, label in vlines:
            plt.axvline(x=x, color=color, linestyle="--", linewidth=1, label=label)
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        plt.legend(unique.values(), unique.keys(), loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_prompt_bar(
    prompt_scores: np.ndarray,
    prompt_span: Optional[Tuple[int, int]],
    input_ids: Sequence[int],
    tokenizer,
    output_path: str,
):
    if prompt_span is None or prompt_scores.size == 0:
        return

    labels = [
        f"{idx}:{format_token(tokenizer, input_ids[prompt_span[0] + idx])}"
        for idx in range(prompt_scores.shape[0])
    ]
    plt.figure(figsize=(max(12, prompt_scores.shape[0] * 0.28), 6))
    plt.bar(np.arange(prompt_scores.shape[0]), prompt_scores, color="#ff7f0e")
    plt.xticks(np.arange(prompt_scores.shape[0]), labels, rotation=90, fontsize=8)
    plt.ylabel("Mean attention over steps")
    plt.xlabel("Prompt token index")
    plt.title("Generation: mean attention assigned to prompt tokens")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_csv(rows: List[Dict[str, object]], output_path: str):
    if not rows:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_run_text_conductor(model, inputs_embeds):
    if hasattr(model, "text_conductor") and callable(model.text_conductor):
        return model.text_conductor(inputs_embeds)
    return inputs_embeds


def main():
    args = parse_args()
    ensure_output_dir(args.output_prefix)

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
        MODEL_PATH, attn_implementation="eager"
    )
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, attn_implementation="eager"
    )
    # finetune_ckpt = torch.load(args.ckpt_path)
    # vl_gpt.load_state_dict(finetune_ckpt)
    vl_gpt.set_attn_implementation("eager")
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    conversation = [
        {
            "role": "<|User|>",
            "content": args.prompt,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag

    input_ids = encode_text(tokenizer, prompt)
    input_length = len(input_ids)
    prompt_span, matched_prompt_text = locate_prompt_span(
        input_ids=input_ids,
        tokenizer=tokenizer,
        prompt_text=args.prompt,
    )

    token_tensor = torch.zeros(
        (args.parallel_size * 2, input_length), dtype=torch.int, device="cuda"
    )
    input_id_tensor = torch.LongTensor(input_ids).cuda()
    for i in range(args.parallel_size * 2):
        token_tensor[i, :] = input_id_tensor
        if i % 2 != 0:
            token_tensor[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(token_tensor)

    generated_ids: List[int] = []
    full_rows = []
    prompt_rows = []
    per_step_rows = []
    prompt_top_rows = []

    if prompt_span is not None:
        prompt_length = prompt_span[1] - prompt_span[0]
        prompt_top_counter = np.zeros(prompt_length, dtype=np.int64)
        prompt_average_vector = np.zeros(prompt_length, dtype=np.float64)
    else:
        prompt_length = 0
        prompt_top_counter = None
        prompt_average_vector = None

    prompt_is_top_count = 0
    prompt_max_group_count = 0
    generated_prefix_max_group_count = 0
    outputs = None

    with torch.inference_mode():
        for step_idx in range(args.image_token_num_per_image):
            outputs = vl_gpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=True,
                past_key_values=outputs.past_key_values if outputs is not None else None,
            )
            hidden_states = outputs.last_hidden_state
            attention_vector = extract_step_attention(outputs.attentions)
            full_rows.append(attention_vector)

            if prompt_span is not None:
                prompt_vector = attention_vector[prompt_span[0] : prompt_span[1]]
                prompt_rows.append(prompt_vector)
                prompt_average_vector += prompt_vector
            else:
                prompt_vector = np.array([], dtype=np.float64)

            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            guided_logits = logit_uncond + args.cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(guided_logits / args.temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            token_id = int(next_token[0, 0].item())
            generated_ids.append(token_id)

            top_source_pos = int(np.argmax(attention_vector))
            top_source_weight = float(attention_vector[top_source_pos])
            top_source_group = classify_position(
                position=top_source_pos,
                input_length=input_length,
                prompt_span=prompt_span,
            )
            top_source_info = get_source_token_info(
                position=top_source_pos,
                input_ids=input_ids,
                generated_ids=generated_ids,
                input_length=input_length,
                tokenizer=tokenizer,
            )

            prompt_mass = float(prompt_vector.sum()) if prompt_span is not None else 0.0
            prompt_wrapper_mass = float(attention_vector[:input_length].sum()) - prompt_mass
            generated_prefix_mass = float(attention_vector[input_length:].sum())

            group_masses = {
                "prompt_text": prompt_mass,
                "prompt_wrapper": prompt_wrapper_mass,
                "generated_image_prefix": generated_prefix_mass,
            }
            max_group = max(group_masses, key=group_masses.get)

            if top_source_group == "prompt_text":
                prompt_is_top_count += 1
            if max_group == "prompt_text":
                prompt_max_group_count += 1
            if max_group == "generated_image_prefix":
                generated_prefix_max_group_count += 1

            prompt_top1_mass = float(prompt_vector.max()) if prompt_vector.size else 0.0
            prompt_top5_mass = (
                float(np.sort(prompt_vector)[-min(5, prompt_vector.size) :].sum())
                if prompt_vector.size
                else 0.0
            )
            prompt_eff_count = effective_token_count(prompt_vector)
            prompt_eff_fraction = (
                prompt_eff_count / prompt_vector.size if prompt_vector.size else 0.0
            )

            top_prompt = top_prompt_entries(
                prompt_weights=prompt_vector,
                prompt_span=prompt_span,
                input_ids=input_ids,
                tokenizer=tokenizer,
                topk=args.topk_prompt,
            )
            if prompt_top_counter is not None and prompt_vector.size:
                prompt_top_counter[int(np.argmax(prompt_vector))] += 1

            for item in top_prompt:
                prompt_top_rows.append(
                    {
                        "step": step_idx + 1,
                        "generated_image_token_id": token_id,
                        **item,
                    }
                )

            per_step_rows.append(
                {
                    "step": step_idx + 1,
                    "generated_image_token_id": token_id,
                    "source_length": int(attention_vector.shape[0]),
                    "top_source_pos": top_source_pos,
                    "top_source_group": top_source_group,
                    "top_source_token_id": top_source_info["token_id"],
                    "top_source_token": top_source_info["token"],
                    "top_source_weight": top_source_weight,
                    "max_group": max_group,
                    "prompt_text_mass": prompt_mass,
                    "prompt_wrapper_mass": prompt_wrapper_mass,
                    "generated_image_prefix_mass": generated_prefix_mass,
                    "prompt_top1_ratio": (prompt_top1_mass / prompt_mass) if prompt_mass > 0 else 0.0,
                    "prompt_top5_ratio": (prompt_top5_mass / prompt_mass) if prompt_mass > 0 else 0.0,
                    "prompt_effective_token_count": prompt_eff_count,
                    "prompt_effective_fraction": prompt_eff_fraction,
                }
            )

            next_token = torch.cat(
                [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
            ).view(-1)
            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

    step_count = len(generated_ids)

    max_source_length = max(row.shape[0] for row in full_rows)
    full_heatmap = np.full((step_count, max_source_length), np.nan, dtype=np.float32)
    for idx, row in enumerate(full_rows):
        full_heatmap[idx, : row.shape[0]] = row

    if prompt_rows:
        prompt_heatmap = np.stack(prompt_rows, axis=0)
        prompt_mean = prompt_heatmap.mean(axis=0)
    else:
        prompt_heatmap = None
        prompt_mean = None

    if prompt_average_vector is not None and step_count > 0:
        prompt_average_vector = prompt_average_vector / step_count

    prompt_token_rows = []
    if prompt_span is not None and prompt_mean is not None:
        for prompt_idx in range(prompt_length):
            abs_pos = prompt_span[0] + prompt_idx
            prompt_token_rows.append(
                {
                    "prompt_index": prompt_idx,
                    "absolute_position": abs_pos,
                    "token_id": int(input_ids[abs_pos]),
                    "token": format_token(tokenizer, int(input_ids[abs_pos])),
                    "mean_attention": float(prompt_mean[prompt_idx]),
                    "normalized_prompt_share": (
                        float(prompt_mean[prompt_idx] / prompt_mean.sum())
                        if prompt_mean.sum() > 0
                        else 0.0
                    ),
                    "times_top1_within_prompt": int(prompt_top_counter[prompt_idx]),
                }
            )

    summary = {
        "mode": "generation",
        "matched_prompt_text": matched_prompt_text,
        "input_length": input_length,
        "generated_image_token_count": step_count,
        "prompt_span": (
            {"start": prompt_span[0], "end_exclusive": prompt_span[1]}
            if prompt_span is not None
            else None
        ),
        "prompt_found": prompt_span is not None,
        "fraction_top_source_in_prompt": (prompt_is_top_count / step_count) if step_count else 0.0,
        "fraction_max_group_prompt": (prompt_max_group_count / step_count) if step_count else 0.0,
        "fraction_max_group_generated_prefix": (generated_prefix_max_group_count / step_count) if step_count else 0.0,
        "mean_prompt_mass": float(np.mean([row["prompt_text_mass"] for row in per_step_rows])) if per_step_rows else 0.0,
        "mean_prompt_wrapper_mass": float(np.mean([row["prompt_wrapper_mass"] for row in per_step_rows])) if per_step_rows else 0.0,
        "mean_generated_prefix_mass": float(np.mean([row["generated_image_prefix_mass"] for row in per_step_rows])) if per_step_rows else 0.0,
        "mean_prompt_top1_ratio": float(np.mean([row["prompt_top1_ratio"] for row in per_step_rows])) if per_step_rows else 0.0,
        "mean_prompt_top5_ratio": float(np.mean([row["prompt_top5_ratio"] for row in per_step_rows])) if per_step_rows else 0.0,
        "mean_prompt_effective_token_count": float(np.mean([row["prompt_effective_token_count"] for row in per_step_rows])) if per_step_rows else 0.0,
        "mean_prompt_effective_fraction": float(np.mean([row["prompt_effective_fraction"] for row in per_step_rows])) if per_step_rows else 0.0,
    }

    save_heatmap(
        matrix=full_heatmap,
        output_path=f"{args.output_prefix}_full_heatmap.png",
        title="Generation: image-token attention over all previous positions",
        xlabel="Source token position",
        ylabel="Generated image token step",
        vmin=args.heatmap_vmin,
        vmax=args.heatmap_vmax,
        # vlines=[
        #     (input_length - 0.5, "lime", "end of initial prompt"),
        # ]
        # + (
        #     [
        #         (prompt_span[0] - 0.5, "yellow", "prompt start"),
        #         (prompt_span[1] - 0.5, "yellow", "prompt end"),
        #     ]
        #     if prompt_span is not None
        #     else []
        # ),
    )

    if prompt_heatmap is not None:
        save_heatmap(
            matrix=prompt_heatmap,
            output_path=f"{args.output_prefix}_prompt_heatmap.png",
            title="Generation: attention restricted to prompt tokens",
            xlabel="Prompt token index",
            ylabel="Generated image token step",
            vmin=args.heatmap_vmin,
            vmax=args.heatmap_vmax,
        )
        save_prompt_bar(
            prompt_scores=prompt_mean,
            prompt_span=prompt_span,
            input_ids=input_ids,
            tokenizer=tokenizer,
            output_path=f"{args.output_prefix}_prompt_token_bar.png",
        )

    write_csv(per_step_rows, f"{args.output_prefix}_per_step.csv")
    if prompt_top_rows:
        write_csv(prompt_top_rows, f"{args.output_prefix}_prompt_top_tokens.csv")
    if prompt_token_rows:
        write_csv(prompt_token_rows, f"{args.output_prefix}_prompt_token_summary.csv")

    with open(f"{args.output_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
