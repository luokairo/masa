import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
from models import MultiModalityCausalLM, VLChatProcessor
import argparse

# =========================
# 参数
# =========================
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--cutoff', type=float, default=0.3)   # 低频范围
parser.add_argument('--smooth', action="store_true")        # 是否使用平滑滤波
args = parser.parse_args()

model_path = "/inspire/hdd/project/exploration-topic/public/downloaded_ckpts/Janus/Janus-Pro-7B"

# =========================
# 加载模型
# =========================
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
    model_path, attn_implementation="eager"
)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    attn_implementation="eager"
)

vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# =========================
# 构造 prompt
# =========================
conversation = [
    {"role": "User", "content": args.prompt},
    {"role": "Assistant", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)

prompt = sft_format + vl_chat_processor.image_start_tag

# =========================
# 频域分解（正确版本）
# =========================
def freq_decompose(x, cutoff_ratio=0.25, smooth=False):
    """
    x: [N, D]
    return: low, high (same shape)
    """
    x = x.float()

    # FFT
    x_fft = torch.fft.fft(x, dim=-1)
    D = x.shape[-1]

    freqs = torch.fft.fftfreq(D, device=x.device)

    if smooth:
        # 平滑滤波（推荐做研究用）
        sharpness = 10
        mask_low = torch.sigmoid(-sharpness * (freqs.abs() - cutoff_ratio))
    else:
        # 硬阈值滤波
        mask_low = (freqs.abs() <= cutoff_ratio).float()

    mask_high = 1.0 - mask_low

    mask_low = mask_low[None, :]
    mask_high = mask_high[None, :]

    # 低频
    x_low = torch.fft.ifft(x_fft * mask_low, dim=-1).real

    # 高频
    x_high = torch.fft.ifft(x_fft * mask_high, dim=-1).real

    return x_low, x_high


def cosine_sim(a, b):
    a = a.mean(dim=0)
    b = b.mean(dim=0)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


# =========================
# 绘图
# =========================
def plot_results(sim_ll, sim_hh, sim_lh, sim_hl):
    layers = np.arange(1, len(sim_ll) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(layers, sim_ll, label="Text_low ↔ Image_low", linewidth=2)
    plt.plot(layers, sim_hh, label="Text_high ↔ Image_high", linewidth=2)

    plt.plot(layers, sim_lh, label="Text_low ↔ Image_high", linestyle="--")
    plt.plot(layers, sim_hl, label="Text_high ↔ Image_low", linestyle="--")

    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.title("Frequency-based Cross-modal Alignment")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("freq_alignment_full-0.3.png", dpi=300)
    plt.close()

    print("Saved to freq_alignment_full-0.1.png")


# =========================
# 主函数
# =========================
@torch.inference_mode()
def run_analysis():
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).cuda()
    text_length = input_ids.shape[1]

    attention_mask = torch.ones_like(input_ids)

    # Step 1: generate image tokens
    outputs = vl_gpt.language_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=576,
        do_sample=True,
        temperature=1.0,
    )

    # Step 2: forward full sequence
    full_ids = outputs[0]

    fwd = vl_gpt.language_model.model(
        input_ids=full_ids.unsqueeze(0),
        attention_mask=torch.ones(1, full_ids.shape[0], device=full_ids.device),
        output_hidden_states=True,
        return_dict=True,
    )

    hidden_states_all = fwd.hidden_states

    sim_ll, sim_hh, sim_lh, sim_hl = [], [], [], []

    for layer_id, h in enumerate(hidden_states_all[1:]):
        h = h[0]

        text_h  = h[:text_length]
        image_h = h[text_length:]

        if image_h.shape[0] == 0:
            continue

        text_low,  text_high  = freq_decompose(
            text_h, cutoff_ratio=args.cutoff, smooth=args.smooth
        )
        image_low, image_high = freq_decompose(
            image_h, cutoff_ratio=args.cutoff, smooth=args.smooth
        )

        # ✔ sanity check（可删）
        recon_error = (text_low + text_high - text_h).abs().mean().item()

        sim_ll.append(cosine_sim(text_low,  image_low))
        sim_hh.append(cosine_sim(text_high, image_high))
        sim_lh.append(cosine_sim(text_low,  image_high))
        sim_hl.append(cosine_sim(text_high, image_low))

        print(
            f"Layer {layer_id+1}: "
            f"LL={sim_ll[-1]:.3f}, "
            f"HH={sim_hh[-1]:.3f}, "
            f"LH={sim_lh[-1]:.3f}, "
            f"HL={sim_hl[-1]:.3f}"
        )

    plot_results(sim_ll, sim_hh, sim_lh, sim_hl)


# =========================
# 运行
# =========================
if __name__ == "__main__":
    run_analysis()