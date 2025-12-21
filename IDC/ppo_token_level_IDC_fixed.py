#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ppo_token_level_IDC_fixed.py

研究用: Token-Level Probeを用いた内部特徴量ベースのPPO学習スクリプト
修正履歴:
 v1: W&Bログ修正, min_length追加
 v2: ★内部報酬の計算ロジック変更 (Sigmoid廃止 -> Raw Logitsの平均)
     これにより、外部報酬とスケール感を合わせ、かつ長文稼ぎを防ぐ。

Usage:
python ppo_token_level_IDC_fixed.py \
    --probe-path "transformer_creativity_probe_token_level.pt" \
    --wandb-run-name "run-research-token-logits-mean" \
    --num-steps 500 \
    --w-ext 1.0 \
    --w-int 0.1 \
    --gate-threshold -1.0
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
            
            # 勾配計算を無効化
            for param in self.model.parameters():
                param.requires_grad = False
                
            print(f"[INFO] Token-Level Probe Loaded successfully.")
            print(f"       Target Layer: {self.layer_idx}, Input Dim: {self.input_dim}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load probe '{path}': {e}")
            raise e

    def get_token_rewards(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [Batch, Seq, Dim]
        Returns: [Batch, Seq, 1] (Raw Logits)
        """
        with torch.no_grad():
            x = hidden_states.to(self.device).float()
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
    """外部報酬モデル（DeBERTa等）による評価"""
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
    """
    内部特徴量から創造性を評価する (Raw Logits Mean版)
    """
    rewards = []
    base_model = policy_model.pretrained_model
    device = base_model.device
    
    responses_text = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    
    for p, r_text in zip(prompts, responses_text):
        # 1. Prompt長さを特定
        prompt_tokens = tokenizer(p, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_tokens.shape[1]
        
        # 2. 全文を入力してHidden Statesを取得
        full_text = p + r_text
        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        
        out = base_model(**inputs, output_hidden_states=True)
        
        # Probeのターゲット層
        target_layer = probe_wrapper.layer_idx
        h_seq = out.hidden_states[target_layer] # [1, Seq, Dim]
        
        # 3. Probe推論 (Logits)
        token_logits = probe_wrapper.get_token_rewards(h_seq) # [1, Seq, 1]
        
        # ★★★ 変更点: Sigmoidを廃止 ★★★
        # Raw Logitsをそのまま使うことで、負の値（ペナルティ）も表現でき、
        # 外部報酬とのスケール感も合わせやすくなる。
        
        # 4. Response部分だけ抽出
        if token_logits.shape[1] > prompt_len:
            response_logits = token_logits[0, prompt_len:, 0]
        else:
            response_logits = token_logits[0, :, 0]
            
        # 5. 平均を取る (SumではなくMean)
        # 長文稼ぎを防ぐため、トークンあたりの「創造性密度」をスコアとする
        score = response_logits.mean().item()
        
        rewards.append(score)
        
    return rewards

# ============================================================
# 4. Main Execution
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    # Models
    parser.add_argument("--policy-model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--rm-model-name", type=str, default="OpenAssistant/reward-model-deberta-v3-large-v2")
    parser.add_argument("--probe-path", type=str, required=True, help="Path to the .pt probe file")
    
    # PPO Params
    parser.add_argument("--num-steps", type=int, default=500, help="Total optimization steps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mini-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-new-tokens", type=int, default=256, help="Minimum generated tokens")
    parser.add_argument("--lr", type=float, default=1.5e-6)
    parser.add_argument("--init-kl-coef", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    
    # Hybrid Reward Parameters
    parser.add_argument("--w-ext", type=float, default=1.0, help="Weight for External RM")
    # Raw Logitsを使うようになったため、デフォルトの重みを少し上げても良いかもしれない
    # (値のスケールが確率ではなくLogitsになったため、影響力が変動する可能性あり)
    parser.add_argument("--w-int", type=float, default=0.5, help="Weight for Internal Probe (Raw Logits)")
    parser.add_argument("--gate-threshold", type=float, default=-0.5, help="Reward threshold to activate Probe")
    
    # W&B
    parser.add_argument("--wandb-project", type=str, default="qwen-creative-ppo")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)
    
    # --- A. PPO Config ---
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

    # --- B. Load Models ---
    print(f"[INFO] Loading Policy Model: {args.policy_model_name}")
    
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "left" 

    # --- C. Dataset Setup ---
    base_inst = "Output ONLY in the following format:\nTitle: <concise title>\nAbstract: <250-400 word abstract>"
    topics = ["Mixture of Experts", "State Space Models", "Sparse Attention", "KV-Cache Optimization", "Continual Learning", "Federated Learning", "Quantization", "Knowledge Distillation", "LoRA", "RLHF", "DPO", "Chain of Thought", "Multi-Agent", "Tool use", "RAG", "Synthetic data", "Multilingual"]
    
    prompt_list = []
    for _ in range(5): 
        for t in topics:
            prompt_list.append(f"Propose a groundbreaking research idea about {t}. {base_inst}")
            prompt_list.append(f"Suggest a novel method to improve {t}. {base_inst}")

    random.shuffle(prompt_list)
    dataset = datasets.Dataset.from_dict({"query": prompt_list})
    
    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}

    # --- D. Initialize Trainer ---
    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator
    )
    device = trainer.accelerator.device

    # --- E. Load Reward Models ---
    print(f"[INFO] Loading Extrinsic RM: {args.rm_model_name}")
    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_model_name)
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_model_name, torch_dtype=torch.float16
    ).to(device)
    rm_model.eval()

    print(f"[INFO] Loading Internal Probe: {args.probe_path}")
    probe_wrapper = TransformerProbeRewardModel(args.probe_path, device)

    # --- F. Training Loop ---
    print(f"[INFO] Starting PPO Training for {args.num_steps} steps...")
    
    def infinite_dataloader(dataloader):
        while True:
            for batch in dataloader:
                yield batch
    
    data_iter = infinite_dataloader(trainer.dataloader)

    for step in tqdm(range(args.num_steps)):
        batch = next(data_iter)
        if not batch: continue

        # 1. Generation
        query_tensors = [
            tokenizer(q, return_tensors="pt").input_ids.squeeze(0).to(device)
            for q in batch["query"]
        ]

        generation_kwargs = {
            "min_length": args.min_new_tokens,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.90,
        }

        response_tensors_full = trainer.generate(query_tensors, **generation_kwargs)
        
        response_tensors = []
        for i, full_seq in enumerate(response_tensors_full):
            q_len = len(query_tensors[i])
            r_tensor = full_seq[q_len:] 
            response_tensors.append(r_tensor)

        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # 2. Reward Calculation
        # (A) External
        ext_scores = compute_external_rm_rewards(
            rm_model, rm_tokenizer, batch["query"], batch["response"], device
        )
        
        # (B) Internal (Raw Logits Mean)
        int_scores = compute_internal_probe_rewards(
            trainer.model, tokenizer, batch["query"], response_tensors, probe_wrapper
        )

        # (C) Hybrid Synthesis
        final_rewards = []
        
        # ログ用リスト: Raw(生の値) と Weighted(実際に足された値) を分けて記録
        log_total = []
        log_ext_raw, log_ext_weighted = [], []
        log_int_raw, log_int_weighted = [], []

        AGGRESSIVE_BASELINE = 8.0
        
        texts_for_table = []

        for q, r, ext, int_val in zip(batch["query"], batch["response"], ext_scores, int_scores):
            # まず重みを計算
            val_ext_weighted = float(ext) * args.w_ext

            int_val = int_val - AGGRESSIVE_BASELINE
            val_int_weighted = float(int_val) * args.w_int
            
            # 外部報酬が悪すぎる場合は、内部報酬を加算しない（ゲート処理）
            if ext < args.gate_threshold:
                total_reward = val_ext_weighted
                actual_int_contrib = 0.0 # Gatedされたので貢献は0
                note = "Gated"
            else:
                total_reward = val_ext_weighted + val_int_weighted
                actual_int_contrib = val_int_weighted
                note = "Hybrid"
            
            # final_rewards.append(torch.tensor(total_reward, device=device))
            raw_rewards_list.append(raw_total)

            
            # ログ保存
            log_total.append(total_reward)
            
            log_ext_raw.append(ext)
            log_ext_weighted.append(val_ext_weighted)
            
            log_int_raw.append(int_val)
            log_int_weighted.append(actual_int_contrib) # Gated時は0が入る

            texts_for_table.append([shorten(q, 30), shorten(r, 50), total_reward, ext, int_val, note])

        # 3. PPO Step
        rewards_tensor = torch.tensor(raw_rewards_list, device=device)
        
        mean = rewards_tensor.mean()
        std = rewards_tensor.std() + 1e-8 # ゼロ除算防止
        
        # 正規化: (x - mean) / std
        # これにより、0.01の差が明確な差としてモデルに伝わる
        normalized_rewards_tensor = (rewards_tensor - mean) / std
        
        # クリップ (-4 ~ +4) で外れ値を抑える
        normalized_rewards_tensor = torch.clamp(normalized_rewards_tensor, -4.0, 4.0)
        
        # リストに戻してPPOに渡す
        final_rewards = [r for r in normalized_rewards_tensor]

        # 3. PPO Step
        stats = trainer.step(query_tensors, response_tensors, final_rewards)
        
        if step % 10 == 0 and wandb.run is not None:
            table = wandb.Table(
                columns=["Query", "Response", "Total", "Ext", "Int", "Type"],
                data=texts_for_table
            )
            wandb.log({"samples": table}, step=step)

        trainer.log_stats(stats, batch, final_rewards)

        if step % 5 == 0:
            # ★ 変更箇所: Weighted (Raw) の形式で表示
            print(f"\n[Step {step}] Total: {avg_total:.3f} | Ext: {avg_ext_w:.3f} ({avg_ext_raw:.3f}) | Int: {avg_int_w:.3f} ({avg_int_raw:.3f})")

    # --- G. Save ---
    save_dir = f"./saved_models/{args.wandb_run_name}" if args.wandb_run_name else "./saved_models/ppo_token_level"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print(f"\n[INFO] Saving model to {save_dir} ...")
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    wandb.finish()
    print("🔔 Done!")

if __name__ == "__main__":
    main()