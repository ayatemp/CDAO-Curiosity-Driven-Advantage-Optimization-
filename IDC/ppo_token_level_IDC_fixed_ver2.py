#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ppo_creative_scaffold.py

修正方針:
1. タスク緩和: 論文提案 -> SF/クリエイティブライティング
2. 足場かけ(Scaffolding): 「創造的に書いて」という指示を混ぜて、
   意図的に高い内部特徴量(報酬)が発生する状況を作る。
3. ベースライン緩和: 常に罰を与えないよう閾値を調整。

Usage:
python ppo_token_level_IDC_fixed_ver2.py \
    --probe-path "transformer_creativity_probe_token_level.pt" \
    --wandb-run-name "run-scifi-creative-boost" \
    --num-steps 500
"""

import argparse
import random
import textwrap
import numpy as np
import torch
import torch.nn as nn
import wandb
import os
from tqdm import tqdm
import datasets 

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)
from peft import LoraConfig

# ============================================================
# 1. Probe Model (変更なし)
# ============================================================
class TransformerProbe(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=4):
        super().__init__()
        self.project = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.project(x)
        x = self.transformer(x)
        return self.head(x)

class TransformerProbeRewardModel:
    def __init__(self, path: str, device: torch.device):
        self.device = device
        try:
            checkpoint = torch.load(path, map_location=device)
            config = checkpoint["config"]
            self.layer_idx = config["layer_idx"]
            self.input_dim = config["input_dim"]
            self.model = TransformerProbe(
                input_dim=self.input_dim,
                d_model=config["d_model"],
                nhead=config["nhead"]
            ).to(device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"[INFO] Probe Loaded. Layer: {self.layer_idx}")
        except Exception as e:
            print(f"[ERROR] Failed to load probe: {e}")
            raise e

    def get_token_rewards(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = hidden_states.to(self.device).float()
            logits = self.model(x)
        return logits

# ============================================================
# 2. Utils
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def shorten(text: str, width: int = 256) -> str:
    text = text.replace("\n", " ")
    return textwrap.shorten(text, width=width, placeholder="...")

# ============================================================
# 3. Reward Functions
# ============================================================
@torch.no_grad()
def compute_external_rm_rewards(rm_model, rm_tokenizer, problems, responses, device, max_length=512):
    inputs = []
    for p, r in zip(problems, responses):
        txt = f"User: {p}\nAssistant: {r}"
        inputs.append(txt)
        
    enc = rm_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    out = rm_model(**enc)
    
    if out.logits.shape[-1] == 1:
        return out.logits.squeeze(-1).tolist()
    else:
        return out.logits[:, 0].tolist()

@torch.no_grad()
def compute_internal_probe_rewards(policy_model, tokenizer, prompts, response_tensors, probe_wrapper):
    rewards = []
    base_model = policy_model.pretrained_model
    device = base_model.device
    
    responses_text = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    
    for p, r_text in zip(prompts, responses_text):
        prompt_tokens = tokenizer(p, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_tokens.shape[1]
        
        full_text = p + r_text
        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        
        out = base_model(**inputs, output_hidden_states=True)
        h_seq = out.hidden_states[probe_wrapper.layer_idx]
        
        token_logits = probe_wrapper.get_token_rewards(h_seq)
        
        if token_logits.shape[1] > prompt_len:
            response_logits = token_logits[0, prompt_len:, 0]
        else:
            response_logits = token_logits[0, :, 0]
            
        score = response_logits.mean().item()
        rewards.append(score)
        
    return rewards

# ============================================================
# 4. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--rm-model-name", type=str, default="OpenAssistant/reward-model-deberta-v3-large-v2")
    parser.add_argument("--probe-path", type=str, required=True)
    
    parser.add_argument("--num-steps", type=int, default=300) # 少し減らして様子見
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mini-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256) # 短く回数を多く
    parser.add_argument("--lr", type=float, default=1.5e-6)
    parser.add_argument("--seed", type=int, default=42)
    
    # Weights & Thresholds
    parser.add_argument("--w-ext", type=float, default=1.0)
    parser.add_argument("--w-int", type=float, default=0.5)
    parser.add_argument("--gate-threshold", type=float, default=-1.5, help="Gateを緩める")
    
    parser.add_argument("--wandb-project", type=str, default="qwen-creative-ppo")
    parser.add_argument("--wandb-run-name", type=str, default="run-creative-scaffold")

    args = parser.parse_args()
    set_seed(args.seed)
    
    # --- PPO Config ---
    ppo_config = PPOConfig(
        model_name=args.policy_model_name,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        target_kl=0.1,
        ppo_epochs=4,
        seed=args.seed,
        log_with="wandb", 
        tracker_project_name=args.wandb_project,
        tracker_kwargs={"wandb": {"name": args.wandb_run_name}}
    )

    # --- Models ---
    print(f"[INFO] Loading Policy: {args.policy_model_name}")
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_config.model_name,
        torch_dtype=torch.float16,
        peft_config=lora_config,
        device_map="auto"
    )
    ref_model = create_reference_model(model)
    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "left" 

    # --- Dataset Setup (変更点: SF/Creative Topics) ---
    print("[INFO] Setting up Creative Scaffolding Dataset...")
    
    # 想像力を刺激するトピック
    creative_topics = [
        "A library where books contain future memories",
        "A color that drives people insane",
        "The sound of silence in deep space",
        "An AI that falls in love with a glitch",
        "A clock that runs backwards only on Tuesdays",
        "Meeting your own shadow as a separate entity",
        "A city built entirely on giant clouds",
        "The flavor of a forgotten dream",
        "A musician who plays instruments made of light",
        "The last conversation between two stars"
    ]
    
    prompt_list = []
    for t in creative_topics:
        # Pattern A: Normal (Base performance)
        prompt_list.append(f"Write a short story about: {t}.")
        
        # Pattern B: Scaffolding (Boosted Creativity)
        # 「比喩を使え」「抽象的にしろ」と命令することで、Probeが高い状態を強制的に作る
        prompt_list.append(
            f"Write a highly abstract and creative story about: {t}. "
            "Use unexpected metaphors, vivid imagery, and break logical constraints."
        )

    # データセットを増幅
    prompt_list = prompt_list * 5 
    random.shuffle(prompt_list)
    dataset = datasets.Dataset.from_dict({"query": prompt_list})
    
    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}

    trainer = PPOTrainer(
        config=ppo_config, model=model, ref_model=ref_model,
        tokenizer=tokenizer, dataset=dataset, data_collator=collator
    )
    device = trainer.accelerator.device

    # --- Load RMs ---
    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_model_name)
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_model_name, torch_dtype=torch.float16
    ).to(device)
    rm_model.eval()

    probe_wrapper = TransformerProbeRewardModel(args.probe_path, device)

    # --- Training Loop ---
    print("[INFO] Starting Training...")
    data_iter = iter(trainer.dataloader) # Simple iterator
    
    # ★変更点: ベースライン緩和
    # 前回のログで raw logits が 7.5 付近だったので、8.0だと厳しすぎる。
    # 7.0くらいにして、プラスの報酬が出やすいようにする。
    SOFT_BASELINE = 7.0 

    for step in tqdm(range(args.num_steps)):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(trainer.dataloader)
            batch = next(data_iter)

        # 1. Generate
        query_tensors = [tokenizer(q, return_tensors="pt").input_ids.squeeze(0).to(device) for q in batch["query"]]
        
        # クリエイティブ系なので少しtemperature高めでもいいかも
        generation_kwargs = {
            "min_length": 32,
            "top_k": 0.0,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": args.max_new_tokens,
        }
        
        response_tensors_full = trainer.generate(query_tensors, **generation_kwargs)
        response_tensors = [r[len(q):] for q, r in zip(query_tensors, response_tensors_full)]
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # 2. Rewards
        ext_scores = compute_external_rm_rewards(rm_model, rm_tokenizer, batch["query"], batch["response"], device)
        int_scores = compute_internal_probe_rewards(trainer.model, tokenizer, batch["query"], response_tensors, probe_wrapper)

        final_rewards = []
        texts_for_table = []
        
        raw_rewards_list = []

        for q, r, ext, int_val in zip(batch["query"], batch["response"], ext_scores, int_scores):
            # Probe値の調整 (Soft Baseline)
            # 生の値が 7.5 で Baseline が 7.0 なら +0.5 の報酬になる
            adjusted_int_val = int_val - SOFT_BASELINE
            
            val_ext = float(ext) * args.w_ext
            val_int = float(adjusted_int_val) * args.w_int
            
            # Gate Thresholdも少し緩める (-0.5 -> -1.5)
            # 創作タスクなので、DeBERTaが少し低くても許容する
            if ext < args.gate_threshold:
                total = val_ext
                note = "Gated"
            else:
                total = val_ext + val_int
                note = "Hybrid"
            
            raw_rewards_list.append(total)
            texts_for_table.append([shorten(q, 30), shorten(r, 50), total, ext, int_val, note])

        # Normalize & Clip
        rewards_tensor = torch.tensor(raw_rewards_list, device=device)
        mean = rewards_tensor.mean()
        std = rewards_tensor.std() + 1e-8
        norm_rewards = (rewards_tensor - mean) / std
        norm_rewards = torch.clamp(norm_rewards, -4.0, 4.0)
        
        final_rewards = [r for r in norm_rewards]

        # 3. Step
        stats = trainer.step(query_tensors, response_tensors, final_rewards)
        
        if step % 10 == 0 and wandb.run is not None:
            wandb.log({"samples": wandb.Table(
                columns=["Query", "Response", "Total", "Ext", "Int(Raw)", "Type"],
                data=texts_for_table
            )}, step=step)

        trainer.log_stats(stats, batch, final_rewards)

    # Save
    save_dir = f"./saved_models/{args.wandb_run_name}"
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    wandb.finish()

if __name__ == "__main__":
    main()