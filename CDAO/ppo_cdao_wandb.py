#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ppo_rdrl_research_curiosity.py

RDRL: Residual-Based Curiosity Driven RL (Research Idea Generation)
 - 外部報酬: DeBERTa (品質担保)
 - 内部報酬: Residual Curiosity (共通成分を除去した独自性)
 - 統合: Gated Hybrid (品質が低いと好奇心報酬は無効)


python ppo_cdao_wandb.py \
    --num-steps 200 \
    --wandb-run-name "run-research-hybrid-01" \
    --w-ext 1.0 \
    --w-int 0.5 \
    --gate-threshold -1.0
"""

import argparse
import random
import textwrap
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os

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
# 1. ユーティリティ
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
    'common_subspace.pt' を読み込み、
    入力されたHiddenStateから「ありきたりな成分」を除去した
    「残差(Residual)」の大きさを計算する。
    """
    def __init__(self, path: str, device: torch.device):
        super().__init__()
        try:
            # weights_only=False でNumPy互換ロード
            data = torch.load(path, map_location="cpu", weights_only=False)
            basis = data["basis"]
            mean = data["mean"]
            print(f"[INFO] Curiosity Model Loaded: Basis {basis.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to load subspace '{path}': {e}")
            print("[WARNING] Using dummy subspace (random init). Training will not work correctly.")
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
        z_common = h_centered @ self.basis.T
        
        # 共通成分を復元
        h_common = z_common @ self.basis
        
        # 3. 残差の計算 (Residual)
        h_residual = h_centered - h_common
        
        # 4. スコアリング (Log Norm)
        norms = torch.norm(h_residual, dim=-1) # [Seq]
        score = torch.log1p(norms).mean()      # Scalar
        
        return score.item()

# ============================================================
# 3. 報酬計算ヘルパー
# ============================================================
@torch.no_grad()
def compute_external_rm_rewards(
    rm_model, rm_tokenizer,
    problems, responses,
    device,
    max_length=512
):
    """外部 RM (DeBERTa) の報酬を計算"""
    inputs = []
    for p, r in zip(problems, responses):
        txt = f"User: {shorten(p)}\nAssistant: {r}"
        inputs.append(txt)
        
    enc = rm_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    out = rm_model(**enc)
    
    # 1次元出力ならそのまま、2次元なら差分など (モデル依存)
    if out.logits.shape[-1] == 1:
        return out.logits.squeeze(-1).tolist()
    else:
        return out.logits[:, 0].tolist()

@torch.no_grad()
def compute_internal_curiosity_rewards(
    policy_model, tokenizer,
    prompts, response_tensors,
    curiosity_model
):
    """
    バッチ内の各サンプルについて、生成部分のHiddenStatesを取得し、
    ResidualCuriosityModelで残差スコアを計算する。
    """
    rewards = []
    device = policy_model.pretrained_model.device
    
    # バッチデコードして再エンコード (正確なHidden取得のため)
    responses_text = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    
    for p, r_text in zip(prompts, responses_text):
        full_text = p + r_text
        inp = tokenizer(full_text, return_tensors="pt").to(device)
        
        # Forward Pass
        out = policy_model.pretrained_model(**inp, output_hidden_states=True)
        
        # 最終層: [1, Seq, Dim] -> [Seq, Dim]
        last_hidden = out.hidden_states[-1].squeeze(0)
        
        # 全トークンの平均残差を計算
        score = curiosity_model.get_reward(last_hidden)
        rewards.append(score)
        
    return rewards

# ============================================================
# 4. メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    # Policy
    parser.add_argument("--policy-model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    # 外部 RM
    parser.add_argument("--rm-model-name", type=str, default="OpenAssistant/reward-model-deberta-v3-large-v2")
    # サブスペース basis (Notebookで作ったファイル)
    parser.add_argument("--subspace-path", type=str, default="common_subspace.pt")

    # PPO Parameters
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mini-batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--init-kl-coef", type=float, default=0.05)

    # 報酬の重み & Gating
    parser.add_argument("--w-ext", type=float, default=1.0, help="外部報酬の重み")
    parser.add_argument("--w-int", type=float, default=0.5, help="内部報酬(好奇心)の重み")
    parser.add_argument("--gate-threshold", type=float, default=-1.0, help="外部報酬がこれ以下なら内部報酬を無効化")

    # W&B
    parser.add_argument("--wandb-project", type=str, default="rdrl-research", help="W&B Project Name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B Run Name")

    parser.add_argument("--debug-samples", action="store_true", help="Print debug samples")
    
    args = parser.parse_args()
    set_seed(args.seed)

    # --------------------------------------------------------
    # PPO Config
    # --------------------------------------------------------
    ppo_config = PPOConfig(
        model_name=args.policy_model_name,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        target_kl=0.1,
        init_kl_coef=args.init_kl_coef,
        ppo_epochs=1,
        log_with="wandb",
        tracker_project_name=args.wandb_project,
        tracker_kwargs={"wandb": {"name": args.wandb_run_name}}
    )

    print(f"[INFO] Policy Model: {args.policy_model_name}")
    print(f"[INFO] Reward Gating Threshold: {args.gate_threshold}")

    # --------------------------------------------------------
    # Models Load
    # --------------------------------------------------------
    # 1. Policy (Actor)
    tok_policy = AutoTokenizer.from_pretrained(args.policy_model_name, padding_side="left")
    if tok_policy.pad_token is None: tok_policy.pad_token = tok_policy.eos_token

    lora_config = LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM", bias="none")
    
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.policy_model_name,
        torch_dtype=torch.float16,
        peft_config=lora_config,
        device_map="auto",
    )
    
    trainer = PPOTrainer(config=ppo_config, model=policy_model, tokenizer=tok_policy)
    device = trainer.accelerator.device
    
    # 2. External Reward Model (Critic for Quality)
    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_model_name)
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_model_name, torch_dtype=torch.float16
    ).to(device)
    rm_model.eval()
    
    # 3. Internal Curiosity Model
    curiosity_model = ResidualCuriosityRewardModel(args.subspace_path, device)

    # --------------------------------------------------------
    # プロンプト生成 (Research Idea に特化)
    # --------------------------------------------------------
    base_inst = (
        "You are an expert LLM researcher. Propose a novel and concrete research idea "
        "about large language models.\n"
        "Output ONLY in the following format:\n\n"
        "Title: <concise LLM research title>\n"
        "Abstract: <150-220 word abstract with motivation, approach, and contribution>\n"
    )

    topics = [
        "Mixture of Experts (MoE)", "State Space Models (Mamba)", "Sparse Attention",
        "KV-Cache Optimization", "Continual Learning", "Federated Learning",
        "4-bit Quantization", "Knowledge Distillation", "Low-Rank Adaptation (LoRA)", 
        "RLHF", "Direct Preference Optimization (DPO)", "Constitutional AI", 
        "Chain of Thought (CoT)", "Multi-Agent collaboration", "Tool use",
        "RAG", "Synthetic data generation", "Multilingual instruction tuning"
    ]

    perspectives = [
        "efficiency", "interpretability", "robustness", "human-like reasoning", 
        "cost-effective training", "scalability", "ethical considerations"
    ]

    templates = [
        "Draft a research proposal about {topic}.\n" + base_inst,
        "Propose an experiment improving {topic} with a focus on {perspective}.\n" + base_inst,
        "Describe a novel method for {topic}, specifically targeting {perspective}.\n" + base_inst,
    ]

    problems = []
    for top in topics:
        for persp in perspectives:
            for t in templates:
                try:
                    txt = t.format(topic=top, perspective=persp)
                except KeyError:
                    txt = t.format(topic=top)
                problems.append(txt)
    
    problems = list(set(problems))
    random.shuffle(problems)
    print(f"[INFO] Generated {len(problems)} diverse prompts.")

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
    for step in range(args.num_steps):
        # --- 1. Rollout ---
        batch_prompts = random.sample(problems, k=ppo_config.batch_size)
        
        inputs = tok_policy(batch_prompts, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            response_tensors = trainer.generate(
                list(inputs.input_ids),
                return_prompt=False,
                max_new_tokens=args.max_new_tokens,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True
            )
            
        batch_responses = tok_policy.batch_decode(response_tensors, skip_special_tokens=True)

        # --- 2. Reward Calculation ---
        
        # External (Quality)
        ext_scores = compute_external_rm_rewards(
            rm_model, rm_tokenizer, batch_prompts, batch_responses, device
        )
        
        # Internal (Curiosity)
        int_scores = compute_internal_curiosity_rewards(
            policy_model, tok_policy, batch_prompts, response_tensors, curiosity_model
        )

        # --- 3. Hybrid Gating ---
        final_rewards = []
        gated_count = 0
        
        for ext, int_val in zip(ext_scores, int_scores):
            # 足切り: 品質の悪い回答には好奇心ボーナスを与えない
            if ext < args.gate_threshold:
                total = float(ext) # 罰のみ
                gated_count += 1
            else:
                # 品質クリアならボーナス付与
                total = float(ext) + (args.w_int * float(int_val))
            
            final_rewards.append(torch.tensor(total, device=device))

        # --- 4. PPO Step ---
        query_tensors = [t for t in inputs.input_ids]
        stats = trainer.step(query_tensors, response_tensors, final_rewards)

        # --- 5. Logging ---
        stats["env/reward_mean_total"] = sum([r.item() for r in final_rewards]) / len(final_rewards)
        stats["env/reward_mean_ext"] = np.mean(ext_scores)
        stats["env/reward_mean_int"] = np.mean(int_scores)
        stats["env/gated_ratio"] = gated_count / args.batch_size
        
        trainer.log_stats(stats, {"query": batch_prompts, "response": batch_responses}, final_rewards)
        
        if args.debug_samples:
            print(f"[Step {step}] Ext: {np.mean(ext_scores):.3f} | Int: {np.mean(int_scores):.3f} | Gated: {gated_count}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    save_dir = f"./saved_models/{args.wandb_run_name}" if args.wandb_run_name else "./saved_models/rdrl_final"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    print(f"\n[INFO] Saving model to {save_dir} ...")
    policy_model.save_pretrained(save_dir)
    tok_policy.save_pretrained(save_dir)
    
    print("\n=== DONE ===")
    wandb.finish()

if __name__ == "__main__":
    main()

    import time
    print("🔔 学習が終了しました！")
    for _ in range(5):
        print('\a')
        time.sleep(0.5)