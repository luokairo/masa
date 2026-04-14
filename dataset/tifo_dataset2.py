import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import linecache
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Tuple
import json
from torchvision import transforms
from models import VLChatProcessor
from torch.utils.data import Dataset, DataLoader
import datasets
from tqdm import tqdm
import re

from datasets import load_dataset
import glob
import os
from PIL import Image
from io import BytesIO

def clean_text(text: str):
    # 1. 去掉前后带换行的 <image>
    text = re.sub(r"\n?<image>\n?", "\n", text)

    # 2. 再兜底去掉残留 <image>
    text = re.sub(r"<image>", "", text)

    # 3. 压缩多余换行
    # text = re.sub(r"\n+", "\n", text)

    return text.strip()


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class TextToImageDataset(Dataset):
    def __init__(
        self,
        model_path,
        data_path,
    ):
        self.gen_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        # only for blip3o

        data_files = glob.glob(data_path)

        t2i_metadata = load_dataset(
            "webdataset",
            data_files=data_files,
            cache_dir='/inspire/hdd/project/exploration-topic/public/ent/download_dataset/cache/blip3o',
            split="train",
            num_proc=64
        )
        if isinstance(t2i_metadata, datasets.DatasetDict):
            if "train" in t2i_metadata:
                t2i_metadata = t2i_metadata["train"]
            else:
                # 如果没有 train，就取第一个 split
                first_split = list(t2i_metadata.keys())[0]
                t2i_metadata = t2i_metadata[first_split]
        t2i = t2i_metadata
        print(len(t2i))

        self.dataset = t2i

    def __getitem__(self, idx):
        curdata = self.dataset[idx]

        caption = curdata['txt']
        prompt = caption
        image = curdata["jpg"].convert("RGB")
        image = self.gen_transform(image)

        conversation = [
            {
                "role": "<|User|>",
                "content":prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.vl_chat_processor.image_start_tag
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').squeeze(0)
        if random.random() < 0.1:
            input_ids[1:-1] = self.vl_chat_processor.pad_id
            judge = 0
        else:
            judge = 1
        
        front=[0, len(input_ids)]
        end=[len(input_ids), len(input_ids)+576]

        return {"input_ids": input_ids, "image": image, "task_type": 0, "front": front, "end": end, "use_attn": judge}
    
    def __len__(self):
        return len(self.dataset)
    
class ShareGPT4V_I2T(Dataset):
    def __init__(
        self,
        model_path,
        data_path,
        image_root = "/inspire/hdd/project/exploration-topic/public/ent/dataset/und/sharegpt4v/data",
        sample_ratio: float = 0.45,
        seed: int = 42
    ):
        self.data_path = data_path
        self.image_root = image_root
        self.sample_ratio = sample_ratio
        self.seed = seed

        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        rng = random.Random(seed)

        print("Loading JSON...")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 原始 total 统计
        total_counter: Counter = Counter()

        # 过滤后可用样本池
        valid_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        print("Checking image existence...")
        for sample in tqdm(data):
            image_rel = sample.get("image", "")
            if not image_rel:
                continue

            source = image_rel.split("/")[0]
            total_counter[source] += 1

            full_path = os.path.join(image_root, image_rel)
            if os.path.exists(full_path):

                # valid_by_source[source].append(sample)
                # ⭐ 关键：clean conversations
                new_sample = sample.copy()
                new_conversations = []

                for turn in sample["conversations"]:
                    new_turn = turn.copy()
                    new_turn["value"] = clean_text(new_turn["value"])
                    new_conversations.append(new_turn)

                new_sample["conversations"] = new_conversations

                valid_by_source[source].append(new_sample)

        # 采样
        self.samples: List[Dict[str, Any]] = []
        self.stats: Dict[str, Dict[str, int]] = {}

        print("\nResampling by original total ratio...")
        for source in sorted(total_counter.keys()):
            total = total_counter[source]
            exist = len(valid_by_source[source])

            target_keep = int(total * sample_ratio)
            keep = min(exist, target_keep)

            if keep > 0:
                sampled = rng.sample(valid_by_source[source], keep)
                self.samples.extend(sampled)
            else:
                sampled = []

            self.stats[source] = {
                "total": total,
                "exist": exist,
                "target_keep": target_keep,
                "final_keep": keep,
            }

            print(
                f"{source}: total={total}, exist={exist}, "
                f"target_keep={target_keep}, final_keep={keep}"
            )

        print(f"\nFinal dataset size: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        curdata = self.samples[idx]

        texts = curdata['conversations']
        image_rel = curdata["image"]
        image_path = os.path.join(self.image_root, image_rel)
        image = Image.open(image_path).convert('RGB')
        image = self.vl_chat_processor.image_processor([image])['pixel_values'].squeeze(0)

        all_input_ids,all_labels = [],[]
        end = []
        length = 0
        front = None

        num_turns = len(texts) // 2

        for turn_idx in range(num_turns):
            human_text = texts[turn_idx * 2]['value']
            gpt_text   = texts[turn_idx * 2 + 1]['value']

            conversation = [
                {
                    "role": "<|User|>",
                    "content": "<image_placeholder>\n" + human_text if turn_idx == 0 else human_text,
                },
                {
                    "role": "<|Assistant|>",
                    "content": "",  # 故意为空，只取prompt部分
                },
            ]

            sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.vl_chat_processor.sft_format,
                system_prompt="",
            )

            # 处理图像的token
            if turn_idx == 0:
                input_ids1 = self.tokenizer.encode(sft_format, return_tensors='pt').squeeze(0)
                input_ids1 = torch.LongTensor(input_ids1)

                image_token_mask: torch.BoolTensor = input_ids1 == self.vl_chat_processor.image_id
                index = image_token_mask.nonzero()[0]

                # 为图像token设置位置
                all_input_ids.append(input_ids1[:index])
                all_input_ids.append(self.vl_chat_processor.image_start_id * torch.ones((1,), dtype=torch.long))
                all_input_ids.append(self.vl_chat_processor.image_id * torch.ones((self.vl_chat_processor.num_image_tokens,), dtype=torch.long))
                all_input_ids.append(self.vl_chat_processor.image_end_id * torch.ones((1,), dtype=torch.long))
                all_input_ids.append(input_ids1[index + 1:])

                prompt_len = len(input_ids1) + 1 + self.vl_chat_processor.num_image_tokens
                labels1 = torch.full((prompt_len,), -100, dtype=torch.long)
                all_labels.append(labels1)

                front = [index.item(), index.item() + self.vl_chat_processor.num_image_tokens]
                length += prompt_len

                # 后续轮次为纯文本
            else:
                input_ids1 = self.tokenizer.encode(sft_format, return_tensors='pt').squeeze(0)
                labels1 = torch.full(input_ids1.shape, -100, dtype=torch.long)
                all_input_ids.append(input_ids1)
                all_labels.append(labels1)
                length += len(input_ids1)

            # ---------- 所有轮次：填入gpt answer ----------
            answer = gpt_text
            if turn_idx == num_turns - 1:
                answer += self.tokenizer.eos_token

            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False, return_tensors='pt').squeeze(0)

            temp_end = [length, length + len(answer_ids)]
            end.append(temp_end)
            length += len(answer_ids)

            all_input_ids.append(answer_ids)
            all_labels.append(answer_ids) 
        
        input_ids = torch.cat(all_input_ids, dim=0)
        labels = torch.cat(all_labels, dim=0)

        return {"input_ids":input_ids, "image":image, "labels": labels, "task_type":1, "front": front, "end": end, "use_attn": 1}




class ImageToTextDataset(Dataset):
    def __init__(
        self,
        model_path,
        data_path,
    ):
        self.gen_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        i2t_metadata = datasets.load_from_disk(data_path)
        i2t_metadata = i2t_metadata['train'] if isinstance(i2t_metadata, datasets.DatasetDict) else i2t_metadata

        print(len(i2t_metadata))
        self.dataset = i2t_metadata

    def __getitem__(self, idx):
        curdata = self.dataset[idx]

        texts = curdata['conversations']
        image_path = curdata['image']

        image = Image.open(image_path).convert('RGB')
        image = self.vl_chat_processor.image_processor([image])['pixel_values'].squeeze(0)
        all_input_ids,all_labels = [],[]
        end = []
        length = 0
        for id, text in enumerate(texts):
            if id%2==0:
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": "",
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                if id==0:
                    conversation[0]['content'] = "<image_placeholder>\n" + text['value']
                else:
                    conversation[0]['content'] = text['value']
            if id%2==1:
                sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                    conversations=conversation,
                    sft_format=self.vl_chat_processor.sft_format,
                    system_prompt="",
                )
                if id==1:
                    input_ids1 = self.tokenizer.encode(sft_format, return_tensors='pt').squeeze(0)
                    input_ids1 = torch.LongTensor(input_ids1)
                    image_token_mask: torch.BoolTensor = input_ids1 == self.vl_chat_processor.image_id
                    index = image_token_mask.nonzero()[0]
                    labels1 = torch.ones(len(input_ids1)+1+self.vl_chat_processor.num_image_tokens, dtype=input_ids1.dtype) * -100

                    all_input_ids.append(input_ids1[:index])
                    all_input_ids.append(self.vl_chat_processor.image_start_id * torch.ones((1), dtype=torch.long))
                    all_input_ids.append(self.vl_chat_processor.image_id * torch.ones((self.vl_chat_processor.num_image_tokens,), dtype=torch.long))
                    all_input_ids.append(self.vl_chat_processor.image_end_id * torch.ones((1), dtype=torch.long))
                    all_input_ids.append(input_ids1[index+1:])

                    all_labels.append(labels1)

                    front = [index, index + self.vl_chat_processor.num_image_tokens]
                    length = length + len(input_ids1) + 1 + self.vl_chat_processor.num_image_tokens
                else:
                    prompt = sft_format
                    input_ids1 = self.tokenizer.encode(prompt, return_tensors='pt').squeeze(0)
                    labels1 = torch.ones(input_ids1.shape, dtype=input_ids1.dtype) * -100
                    all_input_ids.append(input_ids1)
                    all_labels.append(labels1)
                    length = length + len(input_ids1) 
                answer = text['value']
                if id==len(texts)-1:
                    answer += self.tokenizer.eos_token
                temp_end = [length, -1]
                labels = self.tokenizer.encode(answer, add_special_tokens=False, return_tensors='pt').squeeze(0)
                length = length + len(labels) 
                temp_end[-1] = length

                end.append(temp_end)
                all_input_ids.append(labels)
                all_labels.append(labels)

        input_ids = torch.cat(all_input_ids,dim=0)
        labels = torch.cat(all_labels, dim=0)
        return {"input_ids":input_ids, "image":image, "labels": labels, "task_type":1, "front": front, "end": end, "use_attn": 1}

    def __len__(self):
        return len(self.dataset)

def my_collate_fn(batch):
    return batch

def TextToImageDataloader(cfg, tasks=[0,1]):
    probs = []
    dataloaders = []

    dataset1 = TextToImageDataset(
        model_path=cfg.model.processor_path,
        data_path=cfg.dataloader.gen_data_path,
    )
    sampler1 = torch.utils.data.distributed.DistributedSampler(dataset1)
    loader1 = DataLoader(
        dataset1,
        batch_size=cfg.dataloader.train.task1.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
        sampler=sampler1,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        drop_last=True,
        collate_fn=my_collate_fn
    )
    dataloaders.append(loader1)
    probs.append(cfg.dataloader.train.task1.sample_ratio)

    dataset2 = ShareGPT4V_I2T(
        model_path=cfg.model.processor_path,
        data_path=cfg.dataloader.und_data_path
    )
    sampler2 = torch.utils.data.distributed.DistributedSampler(dataset2)
    loader2 = DataLoader(
        dataset2,
        batch_size=cfg.dataloader.train.task2.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
        sampler=sampler2,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        drop_last=True,
        collate_fn=my_collate_fn
    )
    dataloaders.append(loader2)
    probs.append(cfg.dataloader.train.task2.sample_ratio)

    probs = [p / sum(probs) for p in probs]
    if len(dataloaders)==1:
        return dataloaders[0], probs[0]

    return dataloaders, probs

