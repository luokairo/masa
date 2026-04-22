import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

dir_path = "/inspire/hdd/project/exploration-topic/public/ent/NIPS/ckpt/t2i_generation/14209368000.0/grad_stats"

# =========================
# 1. 收集所有梯度（按layer聚合）
# =========================
files = [f for f in os.listdir(dir_path) if f.endswith(".pt")]

def get_step(f):
    return int(re.findall(r"step_(\d+)", f)[0])

files = sorted(files, key=get_step)

layer_grads_t2i = {}
layer_grads_i2t = {}

for f in files:
    data = torch.load(os.path.join(dir_path, f), map_location="cpu")

    for l in data["t2i"]:
        g1_list = data["t2i"][l]
        g2_list = data["i2t"][l]

        layer_grads_t2i.setdefault(l, [])
        layer_grads_i2t.setdefault(l, [])

        layer_grads_t2i[l].extend(g1_list)
        layer_grads_i2t[l].extend(g2_list)

layers = sorted(layer_grads_t2i.keys(), key=lambda x: int(x))

# =========================
# 2. 正确的冲突分析（sample-level + float32）
# =========================

layer_conflict_mean = []
layer_conflict_var = []
layer_conflict_topk = []

# 可选：存heatmap
heatmap_data = []

for l in layers:
    # ✅ 转 float32（关键）
    G1 = torch.stack(layer_grads_t2i[l]).float()  # [N, D]
    G2 = torch.stack(layer_grads_i2t[l]).float()  # [N, D]

    # ===== debug（你可以看一眼数值）=====
    print(f"[Layer {l}] mean abs:", G1.abs().mean().item())

    # ===== sample-level conflict =====
    # sign conflict（最稳）
    conflict_matrix = (G1 * G2 < 0).float()   # [N, D]

    # ===== 每个维度冲突概率 =====
    conflict_prob = conflict_matrix.mean(dim=0)   # [D]

    # ===== 统计 =====
    layer_conflict_mean.append(conflict_prob.mean().item())
    layer_conflict_var.append(conflict_prob.var().item())

    # ===== Top-K =====
    k = int(0.05 * conflict_prob.numel())
    topk_vals, _ = torch.topk(conflict_prob, k)
    layer_conflict_topk.append(topk_vals.mean().item())

    # ===== heatmap（降采样防止太大）=====
    heatmap_data.append(conflict_prob[:1024].cpu().numpy())  # 只取前1024维

# =========================
# 3. 画图
# =========================

layers_idx = np.arange(len(layers))

# ---- 图1：平均冲突 ----
plt.figure()
plt.plot(layers_idx, layer_conflict_mean, marker='o')
plt.xlabel("Layer")
plt.ylabel("Mean Conflict Probability")
plt.title("Average Conflict (Sign-based)")
plt.grid()
plt.savefig("conflict_mean.png", dpi=200)

# ---- 图2：冲突分布（方差）----
plt.figure()
plt.plot(layers_idx, layer_conflict_var, marker='o')
plt.xlabel("Layer")
plt.ylabel("Conflict Variance")
plt.title("Conflict Variance Across Dimensions")
plt.grid()
plt.savefig("conflict_var.png", dpi=200)

# ---- 图3：Top-K ----
plt.figure()
plt.plot(layers_idx, layer_conflict_topk, marker='o')
plt.xlabel("Layer")
plt.ylabel("Top 5% Conflict")
plt.title("Severe Conflict Dimensions")
plt.grid()
plt.savefig("conflict_topk.png", dpi=200)

# =========================
# 4. Heatmap（非常关键）
# =========================

heatmap = np.stack(heatmap_data)  # [L, D_sample]

plt.figure(figsize=(10, 6))
plt.imshow(heatmap, aspect='auto')
plt.colorbar(label="Conflict Probability")
plt.xlabel("Dimension (sampled)")
plt.ylabel("Layer")
plt.title("Conflict Heatmap (Layer × Dimension)")
plt.savefig("conflict_heatmap.png", dpi=200)

# =========================
# 5. 冲突分布（直方图）
# =========================

# 取最后一层做分析
last_l = layers[-1]
G1 = torch.stack(layer_grads_t2i[last_l]).float()
G2 = torch.stack(layer_grads_i2t[last_l]).float()

prod = (G1 * G2).flatten().cpu().numpy()

plt.figure()
plt.hist(prod, bins=100)
plt.xlabel("g1 * g2")
plt.ylabel("Frequency")
plt.title(f"Gradient Product Distribution (Layer {last_l})")
plt.grid()
plt.savefig("conflict_hist.png", dpi=200)

print("Done.")