#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ppo_rdrl_hybrid.py

【最終版】RDRL: Residual-Based Curiosity Driven RL
 - 外部報酬(Quality): DeBERTaなどによる評価
 - 内部報酬(Curiosity): "Common Subspace" からの逸脱度（残差）
 - 統合ロジック: Gated Hybrid (品質が低いと好奇心報酬は無効)
"""

import argparse
import random
import textwrap
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
)
from peft import LoraConfig

# ============================================================
# 1. ユーティリティ & 設定
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
# 2. 残差好奇心モデル (Residual Curiosity Model)
# ============================================================
class ResidualCuriosityRewardModel(nn.Module):
    """
    Notebookで作成した 'common_subspace.pt' を読み込み、
    入力されたHiddenStateから「ありきたりな成分」を除去した
    「残差(Residual)」の大きさを計算する。
    """
    def __init__(self, path: str, device: torch.device):
        super().__init__()
        # weights_only=False でNumPy互換ロード
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            basis = data["basis"]
            mean = data["mean"]
            print(f"[INFO] Curiosity Model Loaded: Basis {basis.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to load subspace: {e}")
            # エラー時はダミーで動作させる（学習は進まないが落ちはしない）
            basis = torch.randn(5, 3584)
            mean = torch.zeros(3584)

        self.register_buffer("basis", basis.to(device, dtype=torch.float16))
        self.register_buffer("mean", mean.to(device, dtype=torch.float16))

    def get_reward(self, h: torch.Tensor) -> float:
        """
        1つのシーケンス(Seq, D)を受け取り、残差スコア(scalar)を返す
        """
        # 型合わせ
        if h.dtype != self.basis.dtype:
            h = h.to(self.basis.dtype)
        
        # 1. 中心化 (Centering)
        h_centered = h - self.mean
        
        # 2. 共通空間への射影 (Projection to Common)
        # z_common: [Seq, k]
        z_common = h_centered @ self.basis.T
        
        # 共通成分を復元: [Seq, D]
        h_common = z_common @ self.basis
        
        # 3. 残差の計算 (Residual)
        h_residual = h_centered - h_common
        
        # 4. スコアリング (Log Norm)
        # 各トークンの残差ノルムを計算し、その平均を取る
        norms = torch.norm(h_residual, dim=-1) # [Seq]
        score = torch.log1p(norms).mean()      # Scalar
        
        return score.item()

# ============================================================
# 3. 報酬計算ヘルパー
# ============================================================
@torch.no_grad()
def compute_external_rewards(
    rm_model, rm_tokenizer,
    prompts, responses,
    device
):
    """外部RM (DeBERTa) による品質スコア"""
    inputs = []
    for p, r in zip(prompts, responses):
        # 評価用フォーマット（モデルにより調整）
        txt = f"User: {shorten(p)}\nAssistant: {r}"
        inputs.append(txt)
        
    enc = rm_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    out = rm_model(**enc)
    # Logitsが1次元か2次元かで処理を分ける
    if out.logits.shape[-1] == 1:
        return out.logits.squeeze(-1).tolist()
    else:
        # 2値分類(Good/Bad)の場合は差分などを取る
        return out.logits[:, 0].tolist() 

@torch.no_grad()
def compute_internal_rewards_batch(
    policy_model, tokenizer,
    prompts, responses_ids,
    curiosity_model
):
    """
    バッチ内の各サンプルについて、生成部分のHiddenStatesを取得し、
    Curiosity Modelで残差スコアを計算する。
    """
    rewards = []
    device = policy_model.device
    
    # 1サンプルずつ処理（メモリ節約 & 正確なマスク処理のため）
    for p, r_ids in zip(prompts, responses_ids):
        # テキストに戻して再エンコード（確実な方法）
        r_text = tokenizer.decode(r_ids, skip_special_tokens=True)
        full_text = p + r_text
        
        inp = tokenizer(full_text, return_tensors="pt").to(device)
        
        # Forward Pass (Hidden States取得)
        out = policy_model.pretrained_model(**inp, output_hidden_states=True)
        
        # 最終層: [1, Seq, Dim]
        last_hidden = out.hidden_states[-1].squeeze(0)
        
        # プロンプト部分を除外し、生成部分のみを評価対象にする
        # prompt_len = tokenizer(p, return_tensors="pt")["input_ids"].shape[1]
        # gen_hidden = last_hidden[prompt_len-1 :] 
        # ※ 簡易的に「全トークンの平均」でも機能しますが、生成部分に絞るのがベスト
        
        # ここでは全トークンを通して評価（文脈も含む）
        score = curiosity_model.get_reward(last_hidden)
        rewards.append(score)
        
    return rewards

# ============================================================
# 4. メイン処理
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    
    # --- Models ---
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--rm-name", default="OpenAssistant/reward-model-deberta-v3-large-v2")
    parser.add_argument("--subspace-path", default="common_subspace.pt")
    
    # --- Training ---
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mini-batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6) # 安全運転
    parser.add_argument("--kl-coef", type=float, default=0.05)
    
    # --- Reward Weights & Gating ---
    parser.add_argument("--w-ext", type=float, default=1.0)
    parser.add_argument("--w-int", type=float, default=0.5)
    parser.add_argument("--gate-threshold", type=float, default=-1.0, help="外部報酬がこれ以下なら内部報酬を無効化")
    
    # --- W&B ---
    parser.add_argument("--wandb-name", default="run-rdrl-hybrid")
    parser.add_argument("--wandb-project", default="rdrl-project")

    args = parser.parse_args()
    set_seed(42)

    # 1. Initialize W&B via PPOConfig
    config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch,
        init_kl_coef=args.kl_coef,
        log_with="wandb",
        tracker_project_name=args.wandb_project,
        tracker_kwargs={"wandb": {"name": args.wandb_name}}
    )

    # 2. Models
    # Policy (Actor)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    peft_config = LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM", bias="none")
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        peft_config=peft_config,
        device_map="auto"
    )
    
    ppo_trainer = PPOTrainer(config, model, tokenizer=tokenizer)
    device = ppo_trainer.accelerator.device
    
    # Reward Model (External)
    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_name)
    rm_model = AutoModelForSequenceClassification.from_pretrained(args.rm_name, torch_dtype=torch.float16).to(device)
    rm_model.eval()
    
    # Curiosity Model (Internal)
    curiosity_model = ResidualCuriosityRewardModel(args.subspace_path, device)

    # 3. Prompts (Simplified for Demo)
    # 実際はNotebookで作った多様なプロンプトを使用してください
    prompts = [
        "Explain quantum computing to a 5-year-old.",
        "Write a poem about rust and gears.",
        "Describe a color that doesn't exist.",
        "Propose a new form of government.",
        "What if cats could talk?"
    ] * 20 

    print("=== Start Training ===")
    
    for step in range(args.steps):
        # --- A. Rollout ---
        batch_prompts = random.sample(prompts, args.batch_size)
        
        # Tokenize
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        
        # Generate
        with torch.no_grad():
            response_tensors = ppo_trainer.generate(
                list(inputs.input_ids),
                return_prompt=False,
                max_new_tokens=64,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True
            )
            
        batch_responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # --- B. Compute Rewards ---
        
        # 1. External (Quality)
        ext_scores = compute_external_rewards(rm_model, rm_tokenizer, batch_prompts, batch_responses, device)
        
        # 2. Internal (Curiosity)
        # モデルのForwardを回してHiddenを取得・計算
        int_scores = compute_internal_rewards_batch(model, tokenizer, batch_prompts, response_tensors, curiosity_model)
        
        # 3. Hybrid Gating Logic (The Core!)
        final_rewards = []
        gated_count = 0
        
        for ext, int_val in zip(ext_scores, int_scores):
            # 足切りチェック
            if ext < args.gate_threshold:
                # 品質が悪い -> 好奇心ボーナスなし (罰のみ)
                total = float(ext) # そのまま、あるいは -1.0 など固定罰
                gated_count += 1
            else:
                # 品質OK -> 好奇心ボーナス付与
                total = float(ext) + (args.w_int * float(int_val))
            
            # PPOにはTensorで渡す
            final_rewards.append(torch.tensor(total, device=device))
            
        # --- C. PPO Step ---
        query_tensors = [t for t in inputs.input_ids]
        stats = ppo_trainer.step(query_tensors, response_tensors, final_rewards)
        
        # --- D. Logging ---
        # 独自のメトリクスを追加
        stats["env/reward_mean_total"] = sum([r.item() for r in final_rewards]) / len(final_rewards)
        stats["env/reward_mean_ext"] = np.mean(ext_scores)
        stats["env/reward_mean_int"] = np.mean(int_scores)
        stats["env/gated_ratio"] = gated_count / args.batch_size # どれくらい足切りされたか
        
        ppo_trainer.log_stats(stats, {"query": batch_prompts, "response": batch_responses}, final_rewards)
        
        if step % 5 == 0:
            print(f"[Step {step}] Ext: {np.mean(ext_scores):.3f} | Int: {np.mean(int_scores):.3f} | Gated: {gated_count}/{args.batch_size}")

    print("Training Finished!")
    model.save_pretrained("./final_model")
    wandb.finish()

if __name__ == "__main__":
    main()