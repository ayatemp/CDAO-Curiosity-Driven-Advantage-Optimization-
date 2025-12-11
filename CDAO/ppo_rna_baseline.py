#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ppo_rnd_baseline.py

【比較実験用ベースライン】Curiosity-Driven RL (RND)
 - 提案手法(RDRL)との比較用。
 - 一般的な「予測誤差に基づく好奇心」を実装。
 - Target Network (固定) と Predictor Network (学習) の誤差を報酬とする。
"""

import sys
import os

# ==============================================================================
# bitsandbytes / triton 完全無効化パッチ
# ==============================================================================
sys.modules["bitsandbytes"] = None
import peft.import_utils
peft.import_utils.is_bnb_available = lambda: False
peft.import_utils.is_bnb_4bit_available = lambda: False
# ==============================================================================

import argparse
import random
import textwrap
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
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
# 2. RND (Random Network Distillation) モジュール
# ============================================================
class RNDCuriosityModule(nn.Module):
    """
    一般的に使われる Curiosity-Driven RL のベースライン。
    Target(固定) と Predictor(学習) の出力誤差を報酬とする。
    """
    def __init__(self, input_dim, hidden_dim, output_dim, device, lr=1e-4):
        super().__init__()
        self.device = device
        
        # 1. Target Network (固定・ランダム)
        # 予測の正解となる「的」。重みは固定。
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)
        # 固定するため勾配を切る
        for p in self.target.parameters():
            p.requires_grad = False

        # 2. Predictor Network (学習対象)
        # Targetの出力を真似しようとする。
        # 未知の入力が来ると真似できず、誤差(報酬)が大きくなる。
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction='none')

    def get_reward(self, h_seq):
        """
        h_seq: [Seq, Input_Dim]
        戻り値: (reward_scalar, loss_mean)
        """
        h_seq = h_seq.to(self.device).float().detach() # Policyの勾配は流さない
        
        with torch.no_grad():
            target_out = self.target(h_seq)
        
        # Predictorで予測
        pred_out = self.predictor(h_seq)
        
        # 誤差計算 (これが好奇心報酬になる)
        # shape: [Seq, Out_Dim] -> [Seq]
        error = ((target_out - pred_out) ** 2).mean(dim=-1)
        
        # 値を安定させるため対数などを取ることもあるが、
        # ベースラインとしてはそのまま、あるいは正規化して使う。
        # ここでは提案手法と比較しやすくするため log1p を採用。
        reward = torch.log1p(error).mean().item()
        
        # 学習用ロス (mean)
        loss = error.mean()
        
        return reward, loss

    def update(self, loss):
        """Predictorを学習させて、既知のデータに対する報酬を下げる"""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ============================================================
# 3. 報酬計算ヘルパー
# ============================================================
@torch.no_grad()
def compute_external_rm_rewards(rm_model, rm_tokenizer, problems, responses, device):
    """外部 RM (DeBERTa) の報酬"""
    inputs = [f"User: {shorten(p)}\nAssistant: {r}" for p, r in zip(problems, responses)]
    enc = rm_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    out = rm_model(**enc)
    if out.logits.shape[-1] == 1:
        return out.logits.squeeze(-1).tolist()
    else:
        return out.logits[:, 0].tolist()

def compute_rnd_rewards_and_update(policy_model, tokenizer, prompts, response_tensors, rnd_model):
    """
    バッチごとにRND報酬を計算し、同時にRNDモデル自体を学習(Update)させる。
    """
    rewards = []
    device = policy_model.pretrained_model.device
    
    responses_text = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    
    total_rnd_loss = 0
    
    for p, r_text in zip(prompts, responses_text):
        full_text = p + r_text
        inp = tokenizer(full_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = policy_model.pretrained_model(**inp, output_hidden_states=True)
            last_hidden = out.hidden_states[-1].squeeze(0) # [Seq, Dim]

        # RND報酬計算 & ロス取得
        # Predictorの勾配が必要なので、ここでは no_grad しない部分があるが
        # get_reward 内で適切に処理する
        reward_scalar, loss_tensor = rnd_model.get_reward(last_hidden)
        
        rewards.append(reward_scalar)
        
        # ロスを蓄積 (バッチ学習するため)
        # backwardはここで行う
        loss_tensor.backward() 
        total_rnd_loss += loss_tensor.item()

    # バッチ全体の勾配が蓄積されたので更新
    rnd_model.optimizer.step()
    rnd_model.optimizer.zero_grad()
    
    return rewards, total_rnd_loss / len(prompts)

# ============================================================
# 4. メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--rm-model-name", type=str, default="OpenAssistant/reward-model-deberta-v3-large-v2")
    
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mini-batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--init-kl-coef", type=float, default=0.05)

    # 重み (RDRLと同じ条件にする)
    parser.add_argument("--w-ext", type=float, default=1.0)
    parser.add_argument("--w-int", type=float, default=0.2, help="RND報酬の重み")
    parser.add_argument("--gate-threshold", type=float, default=-1.0)

    # W&B
    parser.add_argument("--wandb-project", type=str, default="rdrl-research", help="W&B Project Name")
    parser.add_argument("--wandb-run-name", type=str, default="run-baseline-rnd-01")
    parser.add_argument("--debug-samples", action="store_true")
    
    args = parser.parse_args()
    set_seed(42)

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

    # Models
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
    
    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_model_name)
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_model_name, torch_dtype=torch.float16
    ).to(device)
    rm_model.eval()
    
    # ★ ベースライン: RND Curiosity Module
    # Qwen-7BのHidden=3584次元
    rnd_model = RNDCuriosityModule(
        input_dim=3584, 
        hidden_dim=1024, 
        output_dim=1024, 
        device=device,
        lr=1e-4
    )
    print("[INFO] Initialized RND Curiosity Module (Baseline)")

    # Prompts (同じものを使用)
    base_inst = (
        "You are an expert LLM researcher. Propose a novel and concrete research idea "
        "about large language models.\n"
        "Output ONLY in the following format:\n\n"
        "Title: <concise LLM research title>\n"
        "Abstract: <150-220 word abstract with motivation, approach, and contribution>\n"
    )
    topics = ["MoE", "Mamba", "Sparse Attention", "RLHF", "DPO", "CoT", "RAG"]
    templates = ["Draft a research proposal about {topic}.\n" + base_inst]
    problems = []
    for top in topics:
        for t in templates:
            problems.append(t.format(topic=top))
    problems = list(set(problems)) * 50 # 増幅
    random.shuffle(problems)

    # Training Loop
    print("=== Start RND Baseline Training ===")
    
    for step in range(args.num_steps):
        batch_prompts = random.sample(problems, k=ppo_config.batch_size)
        inputs = tok_policy(batch_prompts, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            response_tensors = trainer.generate(
                list(inputs.input_ids),
                return_prompt=False,
                max_new_tokens=args.max_new_tokens,
                temperature=0.9, top_p=0.95, do_sample=True
            )
        batch_responses = tok_policy.batch_decode(response_tensors, skip_special_tokens=True)

        # Rewards
        ext_scores = compute_external_rm_rewards(rm_model, rm_tokenizer, batch_prompts, batch_responses, device)
        
        # ★ RND報酬計算 & RNDモデル更新
        int_scores, rnd_loss = compute_rnd_rewards_and_update(
            policy_model, tok_policy, batch_prompts, response_tensors, rnd_model
        )

        final_rewards = []
        gated_count = 0
        for ext, int_val in zip(ext_scores, int_scores):
            # 同じGating条件で比較する
            if ext < args.gate_threshold:
                total = float(ext)
                gated_count += 1
            else:
                total = float(ext) + (args.w_int * float(int_val))
            final_rewards.append(torch.tensor(total, device=device))

        query_tensors = [t for t in inputs.input_ids]
        stats = trainer.step(query_tensors, response_tensors, final_rewards)

        # Logging
        stats["env/reward_mean_total"] = sum([r.item() for r in final_rewards]) / len(final_rewards)
        stats["env/reward_mean_ext"] = np.mean(ext_scores)
        stats["env/reward_mean_int"] = np.mean(int_scores) # RND Error
        stats["env/rnd_predictor_loss"] = rnd_loss # Predictorが賢くなっているか
        stats["env/gated_ratio"] = gated_count / args.batch_size
        
        trainer.log_stats(stats, {"query": batch_prompts, "response": batch_responses}, final_rewards)
        
        if args.debug_samples:
             print(f"[Step {step}] Ext: {np.mean(ext_scores):.3f} | Int(RND): {np.mean(int_scores):.3f} | Loss: {rnd_loss:.4f}")

    # Save
    save_dir = f"./saved_models/{args.wandb_run_name}"
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    policy_model.save_pretrained(save_dir)
    tok_policy.save_pretrained(save_dir)
    wandb.finish()

if __name__ == "__main__":
    main()