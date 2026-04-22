import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Probe attention positions during Qwen2.5-VL understanding generation. "
            "The outputs are aligned with und_token_attn_probe.py for model comparison."
        )
    )
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--topk_prompt", type=int, default=10)
    parser.add_argument("--output_prefix", type=str, default="qwen25_und_attn_probe")
    parser.add_argument("--heatmap_vmin", type=float, default=None)
    parser.add_argument("--heatmap_vmax", type=float, default=None)
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
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
    start_hint: int,
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
        span = find_subsequence(input_ids, candidate_ids, start_hint=start_hint)
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
        source = "context"
        rel_index = position
    else:
        rel_index = position - input_length
        token_id = int(generated_ids[rel_index])
        source = "generated_prefix"

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
    image_span: Tuple[int, int],
    prompt_span: Optional[Tuple[int, int]],
) -> str:
    if position >= input_length:
        return "generated_prefix"
    if image_span[0] <= position < image_span[1]:
        return "image_block"
    if prompt_span is not None and prompt_span[0] <= position < prompt_span[1]:
        return "prompt_text"
    return "context_other"


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
    plt.bar(np.arange(prompt_scores.shape[0]), prompt_scores, color="#1f77b4")
    plt.xticks(np.arange(prompt_scores.shape[0]), labels, rotation=90, fontsize=8)
    plt.ylabel("Mean attention over steps")
    plt.xlabel("Prompt token index")
    plt.title("Qwen2.5-VL understanding: mean attention assigned to prompt tokens")
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


def collect_image_related_token_ids(model, tokenizer) -> set:
    candidate_ids = set()
    config = getattr(model, "config", None)
    if config is not None:
        for attr in (
            "image_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
            "video_token_id",
        ):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value >= 0:
                candidate_ids.add(int(value))

    for token in (
        "<|image_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|video_pad|>",
        "<image>",
    ):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            token_id = None
        if isinstance(token_id, int) and token_id >= 0:
            candidate_ids.add(token_id)

    return candidate_ids


def infer_image_span(
    input_ids: Sequence[int],
    tokenizer,
    candidate_image_ids: set,
) -> Tuple[int, int]:
    image_positions = []
    for idx, token_id in enumerate(input_ids):
        token = format_token(tokenizer, int(token_id)).lower()
        if (
            int(token_id) in candidate_image_ids
            or "image" in token
            or "vision" in token
        ):
            image_positions.append(idx)

    if image_positions:
        return min(image_positions), max(image_positions) + 1

    return 0, 0


def main():
    args = parse_args()
    ensure_output_dir(args.output_prefix)

    processor_kwargs = {}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels

    processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs)
    tokenizer = processor.tokenizer

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        attn_implementation="eager",
        device_map="auto",
    ).eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image_path},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    model_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = model_inputs.to("cuda")

    input_ids = model_inputs["input_ids"][0].detach().cpu().tolist()
    input_length = len(input_ids)

    candidate_image_ids = collect_image_related_token_ids(model, tokenizer)
    image_span = infer_image_span(
        input_ids=input_ids,
        tokenizer=tokenizer,
        candidate_image_ids=candidate_image_ids,
    )

    prompt_start_hint = image_span[1] if image_span[1] > 0 else 0
    prompt_span, matched_prompt_text = locate_prompt_span(
        input_ids=input_ids,
        tokenizer=tokenizer,
        prompt_text=args.prompt,
        start_hint=prompt_start_hint,
    )

    with torch.inference_mode():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,
            use_cache=True,
        )

    full_sequence = outputs.sequences[0].detach().cpu().tolist()
    generated_ids = full_sequence[input_length:]
    attentions = outputs.attentions
    step_count = min(len(generated_ids), len(attentions))

    full_rows = []
    prompt_rows = []
    per_step_rows = []
    prompt_top_rows = []
    prompt_top_counter = None
    prompt_average_vector = None

    if prompt_span is not None:
        prompt_length = prompt_span[1] - prompt_span[0]
        prompt_top_counter = np.zeros(prompt_length, dtype=np.int64)
        prompt_average_vector = np.zeros(prompt_length, dtype=np.float64)
    else:
        prompt_length = 0

    prompt_is_top_count = 0
    image_is_top_count = 0
    max_group_prompt_count = 0
    max_group_image_count = 0

    for step_idx in range(step_count):
        attention_vector = extract_step_attention(attentions[step_idx])
        full_rows.append(attention_vector)

        if prompt_span is not None:
            prompt_vector = attention_vector[prompt_span[0] : prompt_span[1]]
            prompt_rows.append(prompt_vector)
            prompt_average_vector += prompt_vector
        else:
            prompt_vector = np.array([], dtype=np.float64)

        top_source_pos = int(np.argmax(attention_vector))
        top_source_weight = float(attention_vector[top_source_pos])
        top_source_group = classify_position(
            position=top_source_pos,
            input_length=input_length,
            image_span=image_span,
            prompt_span=prompt_span,
        )
        top_source_info = get_source_token_info(
            position=top_source_pos,
            input_ids=input_ids,
            generated_ids=generated_ids,
            input_length=input_length,
            tokenizer=tokenizer,
        )

        image_mass = float(attention_vector[image_span[0] : image_span[1]].sum())
        prompt_mass = float(prompt_vector.sum()) if prompt_span is not None else 0.0
        context_other_mask = np.ones_like(attention_vector, dtype=bool)
        if image_span[1] > image_span[0]:
            context_other_mask[image_span[0] : image_span[1]] = False
        if prompt_span is not None:
            context_other_mask[prompt_span[0] : prompt_span[1]] = False
        context_other_mask[input_length:] = False
        context_other_mass = float(attention_vector[context_other_mask].sum())
        generated_prefix_mass = float(attention_vector[input_length:].sum())

        group_masses = {
            "image_block": image_mass,
            "prompt_text": prompt_mass,
            "context_other": context_other_mass,
            "generated_prefix": generated_prefix_mass,
        }
        max_group = max(group_masses, key=group_masses.get)

        if top_source_group == "prompt_text":
            prompt_is_top_count += 1
        if top_source_group == "image_block":
            image_is_top_count += 1
        if max_group == "prompt_text":
            max_group_prompt_count += 1
        if max_group == "image_block":
            max_group_image_count += 1

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

        generated_token_id = int(generated_ids[step_idx])
        generated_token = format_token(tokenizer, generated_token_id)

        for item in top_prompt:
            prompt_top_rows.append(
                {
                    "step": step_idx + 1,
                    "generated_token_id": generated_token_id,
                    "generated_token": generated_token,
                    **item,
                }
            )

        per_step_rows.append(
            {
                "step": step_idx + 1,
                "generated_token_id": generated_token_id,
                "generated_token": generated_token,
                "source_length": int(attention_vector.shape[0]),
                "top_source_pos": top_source_pos,
                "top_source_group": top_source_group,
                "top_source_token_id": top_source_info["token_id"],
                "top_source_token": top_source_info["token"],
                "top_source_weight": top_source_weight,
                "max_group": max_group,
                "image_mass": image_mass,
                "prompt_mass": prompt_mass,
                "context_other_mass": context_other_mass,
                "generated_prefix_mass": generated_prefix_mass,
                "prompt_top1_ratio": (prompt_top1_mass / prompt_mass) if prompt_mass > 0 else 0.0,
                "prompt_top5_ratio": (prompt_top5_mass / prompt_mass) if prompt_mass > 0 else 0.0,
                "prompt_effective_token_count": prompt_eff_count,
                "prompt_effective_fraction": prompt_eff_fraction,
            }
        )

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
        "mode": "understanding",
        "model_type": "Qwen2.5-VL",
        "model_path": args.model_path,
        "matched_prompt_text": matched_prompt_text,
        "input_length": input_length,
        "generated_token_count": step_count,
        "image_span": {"start": image_span[0], "end_exclusive": image_span[1]},
        "prompt_span": (
            {"start": prompt_span[0], "end_exclusive": prompt_span[1]}
            if prompt_span is not None
            else None
        ),
        "prompt_found": prompt_span is not None,
        "fraction_top_source_in_prompt": (prompt_is_top_count / step_count) if step_count else 0.0,
        "fraction_top_source_in_image": (image_is_top_count / step_count) if step_count else 0.0,
        "fraction_max_group_prompt": (max_group_prompt_count / step_count) if step_count else 0.0,
        "fraction_max_group_image": (max_group_image_count / step_count) if step_count else 0.0,
        "mean_prompt_mass": float(np.mean([row["prompt_mass"] for row in per_step_rows])) if per_step_rows else 0.0,
        "mean_image_mass": float(np.mean([row["image_mass"] for row in per_step_rows])) if per_step_rows else 0.0,
        "mean_generated_prefix_mass": float(np.mean([row["generated_prefix_mass"] for row in per_step_rows])) if per_step_rows else 0.0,
        "mean_prompt_top1_ratio": float(np.mean([row["prompt_top1_ratio"] for row in per_step_rows])) if per_step_rows else 0.0,
        "mean_prompt_top5_ratio": float(np.mean([row["prompt_top5_ratio"] for row in per_step_rows])) if per_step_rows else 0.0,
        "mean_prompt_effective_token_count": float(np.mean([row["prompt_effective_token_count"] for row in per_step_rows])) if per_step_rows else 0.0,
        "mean_prompt_effective_fraction": float(np.mean([row["prompt_effective_fraction"] for row in per_step_rows])) if per_step_rows else 0.0,
    }

    save_heatmap(
        matrix=full_heatmap,
        output_path=f"{args.output_prefix}_full_heatmap.png",
        title="Qwen2.5-VL: generated token attention over all previous positions",
        xlabel="Source token position",
        ylabel="Generated token step",
        vmin=args.heatmap_vmin,
        vmax=args.heatmap_vmax,
    )

    if prompt_heatmap is not None:
        save_heatmap(
            matrix=prompt_heatmap,
            output_path=f"{args.output_prefix}_prompt_heatmap.png",
            title="Qwen2.5-VL: attention restricted to prompt tokens",
            xlabel="Prompt token index",
            ylabel="Generated token step",
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

    decoded_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDecoded answer:\n{decoded_answer}")


if __name__ == "__main__":
    main()
