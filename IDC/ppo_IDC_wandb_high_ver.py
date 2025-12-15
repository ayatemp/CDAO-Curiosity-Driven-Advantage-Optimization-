#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ppo_transformer_probe_hybrid.py

Hybrid Reward PPO:
 - Policy: Qwen/Qwen2.5-7B-Instruct (with LoRA)
 - Extrinsic Reward: DeBERTa (Quality/Safety)
 - Intrinsic Reward: Transformer Probe (Creativity/Context-Aware)
 - Strategy: Gated Hybrid (Quality Check -> Creativity Bonus)

Usage:
python ppo_transformer_probe_hybrid.py \
    --probe-path "transformer_creativity_probe.pt" \
    --wandb-run-name "run-creative-probe-v1" \
    --w-ext 1.0 \
    --w-int 0.3 \
    --gate-threshold -1.5


python ppo_IDC_wandb_high_ver.py \
    --probe-path "transformer_creativity_probe_enhanced.pt" \
    --wandb-run-name "run-research-enhanced-v1" \
    --num-steps 500 \
    --w-ext 1.0 \
    --w-int 0.3 \
    --gate-threshold -0.5 \
    --batch-size 32 \
    --mini-batch-size 8 \
    --lr 1.41e-5 \
    --init-kl-coef 0.05
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

# グローバルでデバイスを定義
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 1. Probe Model Architecture (Must match saved model)
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
        # Global Average Pooling
        x = x.mean(dim=1) 
        return self.head(x)

class TransformerProbeRewardModel(nn.Module):
    """
    保存されたProbeを読み込み、報酬（Logit）を計算するラッパー
    """
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
            
            # PPO中にProbeが更新されないように凍結
            for param in self.model.parameters():
                param.requires_grad = False
                
            print(f"[INFO] Probe Loaded: Layer {self.layer_idx}, Dim {config['input_dim']}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load probe '{path}': {e}")
            raise e

    def get_reward(self, hidden_states: torch.Tensor) -> float:
        """
        hidden_states: [Seq, Dim] (1つのサンプルのシーケンス)
        """
        with torch.no_grad():
            # バッチ次元を追加 [1, Seq, Dim]
            x = hidden_states.unsqueeze(0).to(self.device)
            logit = self.model(x).item()
        return logit

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
def compute_external_rm_rewards(
    rm_model, rm_tokenizer,
    problems, responses,
    device,
    max_length=512
):
    """外部 RM (DeBERTa) の報酬を計算 (Quality)"""
    inputs = []
    for p, r in zip(problems, responses):
        # Chat形式やInstruct形式に合わせて調整
        txt = f"User: {shorten(p)}\nAssistant: {r}"
        inputs.append(txt)
        
    enc = rm_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    out = rm_model(**enc)
    
    # Logits (通常は [Batch, 1] または [Batch, 2])
    if out.logits.shape[-1] == 1:
        return out.logits.squeeze(-1).tolist()
    else:
        return out.logits[:, 0].tolist()

@torch.no_grad()
def compute_internal_probe_rewards(
    policy_model, tokenizer,
    prompts, response_tensors,
    probe_wrapper
):
    """
    生成されたテキストのHiddenStatesを取得し、Probeで創造性スコアを算出する
    """
    rewards = []
    device = policy_model.pretrained_model.device
    
    # Tensor -> Text
    responses_text = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    
    for p, r_text in zip(prompts, responses_text):
        # プロンプト+応答をモデルに入力してHidden Stateを取得
        full_text = p + r_text
        inp = tokenizer(full_text, return_tensors="pt").to(device)
        
        # Forward Pass (Gradient不要)
        out = policy_model.pretrained_model(**inp, output_hidden_states=True)
        
        # 指定層のHidden Stateを取得
        target_layer = probe_wrapper.layer_idx
        
        # シーケンス全体を取得 (ProbeがTransformerなので文脈が必要)
        # [1, Seq, Dim] -> [Seq, Dim]
        h_seq = out.hidden_states[target_layer].squeeze(0).float()
        
        # Probe推論
        score = probe_wrapper.get_reward(h_seq)
        rewards.append(score)
        
    return rewards

# ============================================================
# 4. メイン処理
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    # Models
    parser.add_argument("--policy-model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--rm-model-name", type=str, default="OpenAssistant/reward-model-deberta-v3-large-v2")
    parser.add_argument("--probe-path", type=str, default="transformer_creativity_probe.pt", help="Path to saved probe")

    # PPO Params
    parser.add_argument("--num-steps", type=int, default=200, help="Total optimization steps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mini-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.41e-5)
    parser.add_argument("--init-kl-coef", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # Hybrid Reward Params
    parser.add_argument("--w-ext", type=float, default=1.0, help="外部報酬(品質)の重み")
    parser.add_argument("--w-int", type=float, default=0.3, help="内部報酬(創造性)の重み")
    parser.add_argument("--gate-threshold", type=float, default=-1.5, help="足切りライン。これ以下の品質なら創造性ボーナス無効")

    # W&B
    parser.add_argument("--wandb-project", type=str, default="qwen-creative-ppo")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    # --------------------------------------------------------
    # PPO Configuration
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Model Loading (with LoRA)
    # --------------------------------------------------------
    print(f"[INFO] Loading Policy Model: {args.policy_model_name}")
    
    # トークナイザーのロードと設定 (ここを修正)
    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "left" # 学習時の重要設定

    # LoRA Config
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    # Policy Model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_config.model_name,
        torch_dtype=torch.float16,
        peft_config=lora_config,
        device_map="auto"
    )
    
    # Reference Model
    ref_model = create_reference_model(model)
    
    # データセット作成
    base_inst = (
        "You are an expert researcher. Propose a novel and concrete research idea.\n"
        "Output ONLY in the following format:\n\n"
        "Title: <concise title>\n"
        "Abstract: <150-200 word abstract>\n"
    )
    topics = [
        "Mixture of Experts", "State Space Models", "Sparse Attention",
        "KV-Cache Optimization", "Continual Learning", "Federated Learning",
        "Quantization", "Knowledge Distillation", "LoRA", "RLHF", "DPO", 
        "Chain of Thought", "Multi-Agent", "Tool use", "RAG", 
        "Synthetic data", "Multilingual"
    ]
    perspectives = ["efficiency", "interpretability", "robustness", "reasoning", "cost"]
    
    prompt_list = []
    for t in topics:
        for p in perspectives:
            txt = f"Draft a research proposal about {t} focusing on {p}.\n{base_inst}"
            prompt_list.append(txt)
    
    print(f"[INFO] Generated {len(prompt_list)} prompts.")
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
    
    # デバイスの取得 (accelerate経由が安全)
    device = trainer.accelerator.device

    # --------------------------------------------------------
    # Reward Models Loading
    # --------------------------------------------------------
    print(f"[INFO] Loading Extrinsic RM: {args.rm_model_name}")
    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_model_name)
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_model_name, torch_dtype=torch.float16
    ).to(device)
    rm_model.eval()

    print(f"[INFO] Loading Intrinsic Probe: {args.probe_path}")
    probe_wrapper = TransformerProbeRewardModel(args.probe_path, device)

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
    print(f"[INFO] Starting Training for {args.num_steps} steps...")
    
    # DataLoaderイテレータ
    data_iter = iter(trainer.dataloader)

    for step in tqdm(range(args.num_steps)):
        # データの取得 (安全策を追加)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(trainer.dataloader)
            batch = next(data_iter)
        
        # 安全策: バッチが空ならスキップ (IndexError回避)
        if not batch or "query" not in batch or len(batch["query"]) == 0:
            print("[WARNING] Empty batch detected. Skipping step.")
            continue

        query_tensors = []
        response_tensors = []
        
        # --- 1. Generation (Rollout) ---
        for query in batch["query"]:
            # attention_maskを取得
            inputs = tokenizer(query, return_tensors="pt").to(device)
            query_tensor = inputs.input_ids.squeeze(0)
            attention_mask = inputs.attention_mask
            
            query_tensors.append(query_tensor)
            
            generation_kwargs = {
                "min_length": -1,
                "top_k": 0.0,
                "top_p": 1.0,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
                "max_new_tokens": args.max_new_tokens,
                "temperature": 0.95,
                "attention_mask": attention_mask, # 警告対策
            }
            
            response = trainer.generate(query_tensor, **generation_kwargs).squeeze(0)
            response_tensors.append(response[len(query_tensor):])

        # 安全策: 何らかの理由でリストが空ならスキップ
        if len(query_tensors) == 0:
            continue

        # --- 2. Reward Calculation ---
        batch_responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # A. Extrinsic (Quality)
        ext_scores = compute_external_rm_rewards(
            rm_model, rm_tokenizer, batch["query"], batch_responses, device
        )
        
        # B. Intrinsic (Creativity Probe)
        int_scores = compute_internal_probe_rewards(
            trainer.model, tokenizer, batch["query"], response_tensors, probe_wrapper
        )

        # C. Hybrid & Gating
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

        # --- 3. PPO Update ---
        # ここで IndexError が出ないよう、空リストでないことを前提とする
        stats = trainer.step(query_tensors, response_tensors, final_rewards)

        # --- 4. Logging ---
        stats["env/reward_mean_total"] = np.mean([r.item() for r in final_rewards])
        stats["env/reward_mean_ext"] = np.mean(stats_ext)
        stats["env/reward_mean_int"] = np.mean(stats_int)
        stats["env/gated_ratio"] = gated_count / len(batch["query"])
        
        if step % 10 == 0 and wandb.run is not None:
            table = wandb.Table(
                columns=["Query", "Response", "Total", "Ext(Qual)", "Int(Creat)"],
                data=texts_for_log
            )
            wandb.log({"game_log": table}, step=step)

        # ★修正: batchにresponseを追加 (trlのログ要件)
        batch["response"] = batch_responses
        
        trainer.log_stats(stats, batch, final_rewards)
        
        if step % 5 == 0:
            print(f"\n[Step {step}] Total: {stats['env/reward_mean_total']:.2f} | "
                  f"Ext: {np.mean(stats_ext):.2f} | Int: {np.mean(stats_int):.2f}")

    # --------------------------------------------------------
    # Save Model
    # --------------------------------------------------------
    save_dir = f"./saved_models/{args.wandb_run_name}" if args.wandb_run_name else "./saved_models/ppo_final"
    
    # 【修正箇所】ディレクトリが存在しない場合は作成する
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"[INFO] Created directory: {save_dir}")

    print(f"\n[INFO] Saving model to {save_dir} ...")
    
    # モデルとトークナイザーの保存
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    wandb.finish()
    
    print("🔔 学習が終了しました！")
    for _ in range(3):
        print('\a')
        time.sleep(0.5)

if __name__ == "__main__":
    main()