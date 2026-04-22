import json
import os
from collections import defaultdict, Counter
from tqdm import tqdm

file_path = "/inspire/hdd/project/exploration-topic/public/ent/dataset/und/sharegpt4v/sharegpt4v_instruct_gpt4-vision_cap100k.json"

# 图片根目录（关键）
image_root = "/inspire/hdd/project/exploration-topic/public/ent/dataset/und/sharegpt4v/data"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

total_counter = Counter()
missing_counter = Counter()

missing_examples = defaultdict(list)

for sample in tqdm(data):
    image_rel_path = sample.get("image", "")
    
    if not image_rel_path:
        continue
    
    source = image_rel_path.split("/")[0]
    total_counter[source] += 1
    
    # 拼接完整路径
    full_path = os.path.join(image_root, image_rel_path)
    
    if not os.path.exists(full_path):
        missing_counter[source] += 1
        
        # 每个source记录前3个缺失样本
        if len(missing_examples[source]) < 3:
            missing_examples[source].append(full_path)

# ===== 打印结果 =====
print("\n===== Missing Image Report =====\n")

for source in total_counter:
    total = total_counter[source]
    missing = missing_counter[source]
    exist = total - missing
    
    print(f"{source}:")
    print(f"  total   = {total}")
    print(f"  exist   = {exist}")
    print(f"  missing = {missing} ({missing/total:.2%})")
    
    if missing > 0:
        print("  examples:")
        for p in missing_examples[source]:
            print(f"    {p}")
    
    print()