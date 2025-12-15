#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ppo_token_level_IDC_fixed.py

Strategy: "The Pincer Attack" (Dense Creativity + Sparse Quality)
 - Each token gets a reward from the Probe (Creativity).
 - The final token gets a reward from DeBERTa (Quality).
 - This forces the model to be creative step-by-step while maintaining global coherence.

Usage:
python ppo_token_level_IDC_fixed.py \
    --probe-path "transformer_creativity_probe_token_level.pt" \
    --wandb-run-name "run-dense-reward-final" \
    --num-steps 300 \
    --w-ext 1.0 \
    --w-int 1.0 \
    --max-new-tokens 512
"""

import argparse
import random
import textwrap
import numpy as np
import torch
import torch.nn as nn
import os
import time
from tqdm import tqdm
import datasets 
import wandb # Manual import

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Probe Classes
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

class TransformerProbeRewardModel(nn.Module):
    def __init__(self, path: str, device: torch.device):
        super().__init__()
        self.device = device
        try:
            checkpoint = torch.load(path, map_location=device)
            config = checkpoint["config"]
            self.layer_idx = config["layer_idx"]
            self.model = TransformerProbe(
                input_dim=config["input_dim"], d_model=config["d_model"], nhead=config["nhead"]
            ).to(device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            for param in self.model.parameters(): param.requires_grad = False
            print(f"[INFO] Token-Level Probe Loaded: Layer {self.layer_idx}")
        except Exception as e:
            print(f"[ERROR] Failed to load probe '{path}': {e}")
            raise e

    def get_token_rewards(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = hidden_states.to(self.device)
            logits = self.model(x)
        return logits

# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def shorten(text: str, width: int = 256) -> str:
    text = text.replace("\n", " ")
    return textwrap.shorten(text, width=width, placeholder="...")

@torch.no_grad()
def compute_external_rm_rewards(rm_model, rm_tokenizer, problems, responses, device, max_length=512):
    inputs = []
    for p, r in zip(problems, responses):
        txt = f"User: {shorten(p)}\nAssistant: {r}"
        inputs.append(txt)
    enc = rm_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    out = rm_model(**enc)
    if out.logits.shape[-1] == 1: return out.logits.squeeze(-1).tolist()
    else: return out.logits[:, 0].tolist()

@torch.no_grad()
def compute_internal_probe_rewards_dense(policy_model, tokenizer, prompts, response_tensors, probe_wrapper):
    """
    各トークンごとの報酬（Probability）をリストとして返す。
    平均化はしない。
    """
    rewards_list = []
    device = policy_model.pretrained_model.device
    responses_text = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    
    for p, r_text in zip(prompts, responses_text):
        prompt_tokens = tokenizer(p, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_tokens.shape[1]
        
        full_text = p + r_text
        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        
        out = policy_model.pretrained_model(**inputs, output_hidden_states=True)
        target_layer = probe_wrapper.layer_idx
        h_seq = out.hidden_states[target_layer].float()
        
        # Logit -> Sigmoid (0.0~1.0)
        token_logits = probe_wrapper.get_token_rewards(h_seq)
        token_probs = torch.sigmoid(token_logits) # [1, Seq, 1]
        
        # Response部分抽出
        if token_probs.shape[1] > prompt_len:
            response_probs = token_probs[0, prompt_len:, 0]
        else:
            response_probs = token_probs[0, :, 0]
            
        rewards_list.append(response_probs.cpu()) # CPUに移してリストへ
        
    return rewards_list

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--rm-model-name", type=str, default="OpenAssistant/reward-model-deberta-v3-large-v2")
    parser.add_argument("--probe-path", type=str, default="transformer_creativity_probe_token_level.pt")
    
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mini-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--init-kl-coef", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    
    # ここでバランス調整
    parser.add_argument("--w-ext", type=float, default=1.0) # RMの重み
    parser.add_argument("--w-int", type=float, default=1.0) # Probeの重み
    
    parser.add_argument("--wandb-project", type=str, default="qwen-creative-ppo")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    ppo_config = PPOConfig(
        model_name=args.policy_model_name,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        target_kl=0.1,
        init_kl_coef=args.init_kl_coef,
        ppo_epochs=4,
        seed=args.seed,
        log_with="wandb",
        tracker_project_name=args.wandb_project,
        tracker_kwargs={"wandb": {"name": args.wandb_run_name}}
    )

    print(f"[INFO] Loading Policy Model: {args.policy_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "left"

    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_config.model_name, torch_dtype=torch.float16, peft_config=lora_config, device_map="auto"
    )
    ref_model = create_reference_model(model)
    
    # Dataset
    base_inst = "You are an expert researcher. Propose a novel and concrete research idea.\nOutput ONLY in the following format:\n\nTitle: <concise title>\nAbstract: <150-200 word abstract>\n"
    topics = ["Mixture of Experts", "State Space Models", "Sparse Attention", "KV-Cache Optimization", "Continual Learning", "Federated Learning", "Quantization", "Knowledge Distillation", "LoRA", "RLHF", "DPO", "Chain of Thought", "Multi-Agent", "Tool use", "RAG", "Synthetic data", "Multilingual"]
    perspectives = ["efficiency", "interpretability", "robustness", "reasoning", "cost"]
    prompt_list = [f"Draft a research proposal about {t} focusing on {p}.\n{base_inst}" for t in topics for p in perspectives]
    
    dataset = datasets.Dataset.from_dict({"query": prompt_list})
    def collator(data): return {key: [d[key] for d in data] for key in data[0]}

    trainer = PPOTrainer(config=ppo_config, model=model, ref_model=ref_model, tokenizer=tokenizer, dataset=dataset, data_collator=collator)
    device = trainer.accelerator.device

    print(f"[INFO] Loading Rewards...")
    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_model_name)
    rm_model = AutoModelForSequenceClassification.from_pretrained(args.rm_model_name, torch_dtype=torch.float16).to(device)
    rm_model.eval()
    probe_wrapper = TransformerProbeRewardModel(args.probe_path, device)

    print(f"[INFO] Starting Training for {args.num_steps} steps...")
    data_iter = iter(trainer.dataloader)

    for step in tqdm(range(args.num_steps)):
        try: batch = next(data_iter)
        except StopIteration:
            data_iter = iter(trainer.dataloader)
            batch = next(data_iter)
        
        if not batch or "query" not in batch or len(batch["query"]) == 0: continue

        # Generation
        query_tensors = [tokenizer(q, return_tensors="pt").input_ids.squeeze(0).to(device) for q in batch["query"]]

        generation_kwargs = {
            "min_length": 50,
            "top_k": 0.0,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.85,
            "repetition_penalty": 1.1
        }

        all_response_tensors = trainer.generate(query_tensors, **generation_kwargs)

        response_tensors = []
        for i, full_seq in enumerate(all_response_tensors):
            q_len = len(query_tensors[i])
            r_tensor = full_seq[q_len:] 
            response_tensors.append(r_tensor)
        if len(response_tensors) == 0: continue

        # Rewards Calculation
        batch_responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # 1. Ext (Sparse)
        ext_scores = compute_external_rm_rewards(rm_model, rm_tokenizer, batch["query"], batch_responses, device)
        
        # 2. Int (Dense) - List[Tensor]
        int_scores_list = compute_internal_probe_rewards_dense(trainer.model, tokenizer, batch["query"], response_tensors, probe_wrapper)

        # 3. Combine (The Sandwich)
        final_rewards = []
        stats_ext = []
        stats_int = []
        texts_for_log = []

        # ★★★ 修正箇所: ここで batch_query ではなく batch["query"] を使う
        for i in range(len(batch["query"])):
            # Dense Reward (0.0~1.0) * w_int
            dense_r = args.w_int * int_scores_list[i].to(device)
            
            # Sparse Reward (-5.0~+5.0) * w_ext
            sparse_r = args.w_ext * ext_scores[i]
            
            # 合成: 全トークンに創造性報酬を与え、最後のトークンに品質報酬を足す
            per_token_reward = dense_r.clone()
            per_token_reward[-1] += sparse_r
            
            final_rewards.append(per_token_reward)
            
            # ログ用
            stats_ext.append(ext_scores[i])
            stats_int.append(int_scores_list[i].mean().item()) # 平均値で記録
            
            # テキストログには合計スコアを表示
            texts_for_log.append([step, batch["query"][i], batch_responses[i], per_token_reward.sum().item(), ext_scores[i], int_scores_list[i].mean().item()])

        # PPO Step
        stats = trainer.step(query_tensors, response_tensors, final_rewards)

        # Inject Custom Stats
        stats["env/reward_mean_total"] = np.mean([r.sum().item() for r in final_rewards]) # Sum of rewards
        stats["env/reward_mean_ext"] = np.mean(stats_ext)
        stats["env/reward_mean_int"] = np.mean(stats_int)
        
        # Manual Table Logging (via accelerator)
        if step % 10 == 0:
            try:
                if trainer.accelerator.is_main_process:
                    table = wandb.Table(columns=["Step", "Query", "Response", "TotalReward", "Ext(Sparse)", "Int(DenseAvg)"], data=texts_for_log)
                    trainer.accelerator.log({"game_log": table}, step=step)
            except: pass

        batch["response"] = batch_responses
        trainer.log_stats(stats, batch, final_rewards)

        print("-" * 50)
        print(f"[Step {step}] Monitor")
        print(f"Stats: TotalSum={stats['env/reward_mean_total']:.2f} | Ext={np.mean(stats_ext):.2f} | IntAvg={np.mean(stats_int):.2f}")
        print(f"Response: {texts_for_log[0][2][:100]}...") 
        print("-" * 50)

    # Save
    save_dir = f"./saved_models/{args.wandb_run_name}" if args.wandb_run_name else "./saved_models/ppo_token_level"
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("🔔 学習が終了しました！")

if __name__ == "__main__":
    main()