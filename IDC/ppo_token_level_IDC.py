#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ppo_token_level_final_fixed.py

修正内容:
 1. W&Bログが表示されない問題を修正 (TRL標準の log_stats を使用)
 2. 出力文章が短くなる問題を修正 (min_length設定)
 3. Token-Level Probe と Hybrid Reward のロジックは維持

Usage:
python ppo_token_level_IDC.py \
    --probe-path "transformer_creativity_probe_token_level.pt" \
    --wandb-run-name "run-research-token-final-v2" \
    --num-steps 300 \
    --w-ext 1.0 \
    --w-int 0.05 \
    --gate-threshold -0.5 \
    --batch-size 32 \
    --mini-batch-size 8 \
    --lr 1.41e-5 \
    --init-kl-coef 0.05 \
    --max-new-tokens 512
"""

import argparse
import random
import textwrap
import numpy as np
import torch
import torch.nn as nn
import wandb
import os
import time
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

# グローバル設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 1. Probe Model Architecture (Token-Level)
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
        # x: [Batch, Seq, Input_Dim]
        x = self.project(x)
        x = self.transformer(x)
        return self.head(x) # -> [Batch, Seq, 1]

class TransformerProbeRewardModel(nn.Module):
    def __init__(self, path: str, device: torch.device):
        super().__init__()
        self.device = device
        try:
            checkpoint = torch.load(path, map_location=device)
            config = checkpoint["config"]
            
            self.layer_idx = config["layer_idx"]
            self.model = TransformerProbe(
                input_dim=config["input_dim"],
                d_model=config["d_model"],
                nhead=config["nhead"]
            ).to(device)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            
            for param in self.model.parameters():
                param.requires_grad = False
                
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
# 2. ユーティリティ
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
# 3. 報酬計算関数
# ============================================================
@torch.no_grad()
def compute_external_rm_rewards(rm_model, rm_tokenizer, problems, responses, device, max_length=512):
    inputs = []
    for p, r in zip(problems, responses):
        txt = f"User: {shorten(p)}\nAssistant: {r}"
        inputs.append(txt)
    enc = rm_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    out = rm_model(**enc)
    if out.logits.shape[-1] == 1:
        return out.logits.squeeze(-1).tolist()
    else:
        return out.logits[:, 0].tolist()

@torch.no_grad()
def compute_internal_probe_rewards(policy_model, tokenizer, prompts, response_tensors, probe_wrapper):
    """
    修正版: Logitをそのまま使わず、Sigmoidを通して 0.0~1.0 に正規化する。
    """
    rewards = []
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
        
        # Probe推論 (Logits)
        token_logits = probe_wrapper.get_token_rewards(h_seq) # [1, Seq, 1]
        
        # ★★★ ここが修正点: Logit -> Probability (0.0~1.0) ★★★
        # +8.0 -> 0.999, -3.0 -> 0.047
        token_probs = torch.sigmoid(token_logits)
        
        # Response部分だけ抽出
        if token_probs.shape[1] > prompt_len:
            response_probs = token_probs[0, prompt_len:, 0]
        else:
            response_probs = token_probs[0, :, 0]
            
        # 平均「確率」を報酬とする (これで 0.0〜1.0 の範囲に収まる)
        score = response_probs.mean().item()
        rewards.append(score)
        
    return rewards

# ============================================================
# 4. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    # Models
    parser.add_argument("--policy-model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--rm-model-name", type=str, default="OpenAssistant/reward-model-deberta-v3-large-v2")
    parser.add_argument("--probe-path", type=str, default="transformer_creativity_probe_token_level.pt")
    
    # PPO Params
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mini-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.41e-5)
    parser.add_argument("--init-kl-coef", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    
    # Hybrid Reward
    parser.add_argument("--w-ext", type=float, default=1.0)
    parser.add_argument("--w-int", type=float, default=0.05)
    parser.add_argument("--gate-threshold", type=float, default=-0.5)
    
    # W&B
    parser.add_argument("--wandb-project", type=str, default="qwen-creative-ppo")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)
    
    # Config (修正：log_with="wandb" を使用してTRLに任せる)
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
        log_with="wandb",  # ★ここを修正（None -> wandb）
        tracker_project_name=args.wandb_project,
        tracker_kwargs={"wandb": {"name": args.wandb_run_name}}
    )

    # Load Policy
    print(f"[INFO] Loading Policy Model: {args.policy_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "left"

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
    
    # Dataset
    base_inst = "You are an expert researcher. Propose a novel and concrete research idea.\nOutput ONLY in the following format:\n\nTitle: <concise title>\nAbstract: <150-200 word abstract>\n"
    topics = ["Mixture of Experts", "State Space Models", "Sparse Attention", "KV-Cache Optimization", "Continual Learning", "Federated Learning", "Quantization", "Knowledge Distillation", "LoRA", "RLHF", "DPO", "Chain of Thought", "Multi-Agent", "Tool use", "RAG", "Synthetic data", "Multilingual"]
    perspectives = ["efficiency", "interpretability", "robustness", "reasoning", "cost"]
    
    prompt_list = []
    for t in topics:
        for p in perspectives:
            prompt_list.append(f"Draft a research proposal about {t} focusing on {p}.\n{base_inst}")
    
    dataset = datasets.Dataset.from_dict({"query": prompt_list})
    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator
    )
    device = trainer.accelerator.device

    # Load Rewards
    print(f"[INFO] Loading Extrinsic RM: {args.rm_model_name}")
    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_model_name)
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_model_name, torch_dtype=torch.float16
    ).to(device)
    rm_model.eval()

    print(f"[INFO] Loading Token-Level Probe: {args.probe_path}")
    probe_wrapper = TransformerProbeRewardModel(args.probe_path, device)

    # Loop
    print(f"[INFO] Starting Training for {args.num_steps} steps...")
    data_iter = iter(trainer.dataloader)

    for step in tqdm(range(args.num_steps)):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(trainer.dataloader)
            batch = next(data_iter)
        
        if not batch or "query" not in batch or len(batch["query"]) == 0:
            continue

        # --- 1. Generation ---
        query_tensors = [
            tokenizer(q, return_tensors="pt").input_ids.squeeze(0).to(device)
            for q in batch["query"]
        ]

        # ★ 修正: min_length を設定してある程度長い文章を出力させる
        generation_kwargs = {
            "min_length": 64,       # 最低でも64トークンは喋らせる
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.95,
        }

        all_response_tensors = trainer.generate(query_tensors, **generation_kwargs)

        response_tensors = []
        for i, full_seq in enumerate(all_response_tensors):
            q_len = len(query_tensors[i])
            r_tensor = full_seq[q_len:] 
            response_tensors.append(r_tensor)

        if len(response_tensors) == 0: continue

        # --- 2. Reward Calculation ---
        batch_responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        ext_scores = compute_external_rm_rewards(
            rm_model, rm_tokenizer, batch["query"], batch_responses, device
        )
        int_scores = compute_internal_probe_rewards(
            trainer.model, tokenizer, batch["query"], response_tensors, probe_wrapper
        )

        final_rewards = []
        stats_ext = []
        stats_int = []
        gated_count = 0
        texts_for_log = []

        for q, r, ext, int_val in zip(batch["query"], batch_responses, ext_scores, int_scores):
            if ext < args.gate_threshold:
                total = float(ext)
                gated_count += 1
            else:
                total = float(ext) + (args.w_int * float(int_val))
            
            final_rewards.append(torch.tensor(total, device=device))
            stats_ext.append(ext)
            stats_int.append(int_val)
            texts_for_log.append([q, r, total, ext, int_val])

        # --- 3. PPO Step ---
        stats = trainer.step(query_tensors, response_tensors, final_rewards)

        # --- 4. Logging (TRL Standard + Manual Columns) ---
        # 独自指標をstatsに追加して、TRLに一緒にログを取らせる
        stats["env/reward_mean_total"] = np.mean([r.item() for r in final_rewards])
        stats["env/reward_mean_ext"] = np.mean(stats_ext)
        stats["env/reward_mean_int"] = np.mean(stats_int)
        stats["env/gated_ratio"] = gated_count / len(batch["query"])
        
        # テキストログ (Table)
        if step % 10 == 0 and wandb.run is not None:
            table = wandb.Table(
                columns=["Query", "Response", "Total", "Ext", "Int"],
                data=texts_for_log
            )
            wandb.log({"game_log": table}, step=step)

        # ★ ここが重要： log_stats を呼ぶことでW&Bに正常に送信される
        # (batchにresponseを追加しておくとTRLが内部でログを取りやすい)
        batch["response"] = batch_responses
        trainer.log_stats(stats, batch, final_rewards)

        # ターミナル監視用
        if step % 5 == 0:
            print("-" * 50)
            print(f"[Step {step}] Total={stats['env/reward_mean_total']:.2f} | Ext={np.mean(stats_ext):.2f} | Int={np.mean(stats_int):.2f}")
            print(f"Response (Snippet): {batch_responses[0][:80]}...") 
            print("-" * 50)

    # --------------------------------------------------------
    # Save Model
    # --------------------------------------------------------
    save_dir = f"./saved_models/{args.wandb_run_name}" if args.wandb_run_name else "./saved_models/ppo_token_level"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"[INFO] Created directory: {save_dir}")

    print(f"\n[INFO] Saving model to {save_dir} ...")
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    wandb.finish()
    
    print("🔔 学習が終了しました！")
    for _ in range(3):
        print('\a')
        time.sleep(0.5)

if __name__ == "__main__":
    main()