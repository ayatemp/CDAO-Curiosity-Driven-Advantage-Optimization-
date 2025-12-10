#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ppo_rdrl_lora_wandb.py

RDRL: Representation-Diversity RL (LoRA 版) + W&B Logging
 - Qwen2.5-7B-Instruct + LoRA + ValueHead を PPO で更新
 - 内部多様性サブスペース (div_basis.pt) による「内発的報酬」
 - 外部 RM (DeBERTa reward model) による「外発的報酬」
 - Weights & Biases (W&B) による実験管理を追加

python ppo_rdrl_lora_wandb.py \
    --num-steps 200 \
    --batch-size 4 \
    --mini-batch-size 2 \
    --wandb-project "my-rdrl-project" \
    --wandb-run-name "run-004-step100" \
    --w-ext 0.7 --w-int 0.3

python ppo_rdrl_lora_wandb.py \
    --num-steps 200 \
    --batch-size 4 \
    --mini-batch-size 2 \
    --wandb-project "my-rdrl-project" \
    --wandb-run-name "run-006-final-fix" \
    --w-ext 0.7 --w-int 0.3 \
    --lr 5e-6 \
    --init-kl-coef 0.2

python ppo_rdrl_lora_wandb.py \
    --num-steps 3 \
    --batch-size 4 \
    --mini-batch-size 2 \
    --wandb-project "my-rdrl-project" \
    --wandb-run-name "run-006-final-fix" \
    --w-ext 0.7 --w-int 0.3 \
    --lr 5e-6 \
    --init-kl-coef 0.2


python ppo_rdrl_lora_wandb.py \
    --num-steps 3 \
    --batch-size 4 \
    --mini-batch-size 2 \
    --wandb-project "my-rdrl-project" \
    --wandb-run-name "run-004-step100" \
    --w-ext 0.7 --w-int 0.3
"""

import argparse
import random
import textwrap
from typing import List, Tuple
import numpy as np

import torch
from torch import nn

# ★ W&B のインポートは TRL 内部で処理されますが、
# 明示的に wandb ログインチェック等を行いたい場合は import wandb を使います
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
# ユーティリティ
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
# 多様性サブスペースモデル
# ============================================================

class DiversitySubspaceModel(nn.Module):
    """
    事前に作成したサブスペース基底 (basis: [k, D]) を使って
    hidden_state h: [..., D] を射影して多様性スコアを計算する簡易なモデル。
    """
    def __init__(self, basis: torch.Tensor):
        super().__init__()
        with torch.no_grad():
            q, _ = torch.linalg.qr(basis.T)  # [D, k]
            ortho = q.T                      # [k, D]
        self.register_buffer("basis", ortho)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [..., D]
        return: [..., k]  (サブスペース上の座標)
        """
        D = self.basis.shape[1]
        assert h.shape[-1] == D, f"expected last dim {D}, got {h.shape[-1]}"

        # h の device / dtype に合わせて basis をキャスト
        basis = self.basis.to(device=h.device, dtype=h.dtype)

        # (..., D) x (D, k) = (..., k)
        return h @ basis.T

    def token_diversity(self, h: torch.Tensor) -> torch.Tensor:
        z = self.project(h)          # [..., k]
        score = torch.norm(z, dim=-1)
        return score


# ============================================================
# 外部 RM 用ヘルパー
# ============================================================

def _logits_to_scores(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        logits = logits.view(logits.size(0), -1)
    num_labels = logits.size(1)
    if num_labels == 1:
        return logits.squeeze(1)
    elif num_labels == 2:
        return logits[:, 1] - logits[:, 0]
    else:
        return logits.mean(dim=1)


@torch.no_grad()
def compute_external_rm_rewards(
    rm_model: nn.Module,
    rm_tokenizer,
    problems: List[str],
    responses: List[str],
    device: torch.device,
    max_length: int = 512,
) -> List[float]:
    """
    外部 RM の報酬 (sequence-level) を計算。
    """
    texts = []
    for p, r in zip(problems, responses):
        short_p = shorten(p, width=256)
        txt = f"User:\n{short_p}\n\nAssistant:\n{r}"
        texts.append(txt)

    inputs = rm_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    outputs = rm_model(**inputs)
    logits = outputs.logits
    scores = _logits_to_scores(logits)  # [B]
    return scores.detach().cpu().tolist()


# ============================================================
# 内部多様性報酬 (per-token)
# ============================================================

@torch.no_grad()
def compute_internal_diversity_rewards(
    policy_model: AutoModelForCausalLMWithValueHead,
    tokenizer,
    prompts: List[str],
    responses_ids: List[torch.Tensor],
    subspace_model: DiversitySubspaceModel,
    target_layer: int = -1,
    max_length: int = 512,
) -> List[torch.Tensor]:
    """
    各サンプルごとに内部状態をサブスペースへ射影し、L2ノルムを計算。
    """
    base_model = policy_model.pretrained_model
    device = next(policy_model.parameters()).device

    texts = []
    for p, resp_ids in zip(prompts, responses_ids):
        resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
        full = p + resp_text
        texts.append(full)

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    out = base_model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = out.hidden_states
    h_layer = hidden_states[target_layer]  # [B, T, D]

    rewards_per_sample: List[torch.Tensor] = []

    for i, (p, resp_ids) in enumerate(zip(prompts, responses_ids)):
        enc_prompt = tokenizer(
            p,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        prompt_len = enc_prompt["input_ids"].shape[1]
        full_len = (enc["attention_mask"][i] == 1).sum().item()

        start = prompt_len
        end = full_len
        if start >= end:
            start = max(0, full_len - 1)

        h_resp = h_layer[i, start:end, :]  # [T_resp, D]
        scores = subspace_model.token_diversity(h_resp)  # [T_resp]
        rewards_per_sample.append(scores.detach())

    return rewards_per_sample


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # Policy
    parser.add_argument("--policy-model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    # 外部 RM
    parser.add_argument("--rm-model-name", type=str, default="OpenAssistant/reward-model-deberta-v3-large-v2")
    # サブスペース basis
    parser.add_argument("--subspace-basis-path", type=str, default="div_basis.pt")

    # PPO Parameters
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--init-kl-coef", type=float, default=0.1)

    # 報酬の重み
    parser.add_argument("--w-ext", type=float, default=0.7, help="外部 RM 報酬の重み")
    parser.add_argument("--w-int", type=float, default=0.3, help="内部多様性報酬の重み")

    # W&B / Logging
    parser.add_argument("--wandb-project", type=str, default="rdrl-ppo-experiment", help="W&B Project Name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B Run Name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B Entity (User/Team)")
    parser.add_argument("--debug-samples", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    # --------------------------------------------------------
    # PPO Config with W&B
    # --------------------------------------------------------
    # log_with="wandb" を指定すると、PPOTrainer が自動で tracker を初期化します
    ppo_config = PPOConfig(
        model_name=args.policy_model_name,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        target_kl=0.1,
        init_kl_coef=args.init_kl_coef,
        ppo_epochs=1,
        log_with="wandb",  # ★ここでW&Bを指定
        tracker_project_name=args.wandb_project,
        tracker_kwargs={
            "wandb": {
                "name": args.wandb_run_name,
                "entity": args.wandb_entity,
            }
        },
    )

    print(f"[INFO] policy model : {args.policy_model_name}")
    print(f"[INFO] RM model     : {args.rm_model_name}")
    print(f"[INFO] subspace path: {args.subspace_basis_path}")
    print(f"[INFO] W&B Project  : {args.wandb_project}")

    # --------------------------------------------------------
    # Policy + LoRA
    # --------------------------------------------------------
    tok_policy = AutoTokenizer.from_pretrained(args.policy_model_name, use_fast=False)
    tok_policy.padding_side = "left"
    if tok_policy.pad_token is None:
        tok_policy.pad_token = tok_policy.eos_token

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.policy_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        peft_config=lora_config,
    )

    # PPOTrainer の初期化
    trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        tokenizer=tok_policy,
    )
    device_policy = trainer.accelerator.device
    print(f"[INFO] PPO accelerator device: {device_policy}")

    # --------------------------------------------------------
    # サブスペース basis & 外部RM
    # --------------------------------------------------------
    try:
        basis = torch.load(args.subspace_basis_path, map_location="cpu")
        basis = basis.to(device_policy)
        subspace_model = DiversitySubspaceModel(basis).to(device_policy)
        print(f"[INFO] subspace basis shape: {basis.shape}")
    except FileNotFoundError:
        print(f"[WARNING] Subspace basis not found at {args.subspace_basis_path}. Using random init for demo.")
        # ダミー basis (デモ用)
        dummy_basis = torch.randn(768, 4096, device=device_policy) # [k, D] ※実際の次元に合わせてください
        subspace_model = DiversitySubspaceModel(dummy_basis).to(device_policy)

    rm_device = torch.device("cpu")
    # もしGPUメモリに余裕があればRMもGPUへ
    if torch.cuda.device_count() > 1:
        rm_device = torch.device("cuda:1")
        print("[INFO] Moving RM to cuda:1")
    
    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_model_name, use_fast=True)
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_model_name,
        torch_dtype=torch.float32,
    ).to(rm_device)
    rm_model.eval()

    # --------------------------------------------------------
    # プロンプト群
    # --------------------------------------------------------
    base_inst = (
        "You are an expert LLM researcher. Propose a novel and concrete research idea "
        "about large language models.\n"
        "Output ONLY in the following format:\n\n"
        "Title: <concise LLM research title>\n"
        "Abstract: <150-220 word abstract with motivation, approach, and contribution>\n"
    )
    templates = [
        base_inst + "\nFocus on: {topic}.",
        "Draft a research proposal about {topic} in the context of LLMs.\n" + base_inst,
        "Describe a novel experiment regarding {topic} for large language models.\n" + base_inst,
    ]
    
    topics = [
        "alignment and safety", "multi-agent collaboration", "efficient training",
        "world models", "tool use and APIs", "long-context reasoning",
        "interpretability", "hallucination mitigation", "code generation",
        "mathematical reasoning", "medical applications", "legal reasoning",
        "multimodal understanding", "retrieval-augmented generation (RAG)",
        "quantization and compression", "reinforcement learning from human feedback (RLHF)"
    ]

    problems = []
    for t in templates:
        for top in topics:
            problems.append(t.format(topic=top))
    
    # さらにランダムに混ぜて、データ数を確保 (例: 3テンプレート x 16トピック x 5回 = 240個)
    problems = problems * 5
    random.shuffle(problems)
    
    print(f"[INFO] Generated {len(problems)} diverse prompts.")

    # --------------------------------------------------------
    # PPO Loop
    # --------------------------------------------------------
    for step in range(args.num_steps):
        # 1. バッチ作成
        batch_prompts = random.sample(problems, k=ppo_config.batch_size)
        
        enc = tok_policy(batch_prompts, padding=True, return_tensors="pt")
        enc = {k: v.to(device_policy) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        # 2. 生成
        with torch.no_grad():
            gen = policy_model.generate(
                **{k: v for k, v in enc.items() if k in ["input_ids", "attention_mask"]},
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                repetition_penalty=1.1,
                pad_token_id=tok_policy.pad_token_id,
            )

        query_tensors = [row for row in enc["input_ids"]]
        response_tensors = []
        response_texts = []

        for i in range(gen.size(0)):
            resp_ids = gen[i, prompt_len:]
            response_tensors.append(resp_ids)
            resp_text = tok_policy.decode(resp_ids, skip_special_tokens=True)
            response_texts.append(resp_text)

        # 3. 報酬計算
        # (A) 外部 RM
        ext_rewards = compute_external_rm_rewards(
            rm_model=rm_model, rm_tokenizer=rm_tokenizer,
            problems=batch_prompts, responses=response_texts,
            device=rm_device, max_length=512,
        )
        # (B) 内部多様性
        int_reward_tokens = compute_internal_diversity_rewards(
            policy_model=policy_model, tokenizer=tok_policy,
            prompts=batch_prompts, responses_ids=response_tensors,
            subspace_model=subspace_model, target_layer=-1, max_length=512,
        )

        combined_rewards = []
        agg_int_rewards = []

        # 正規化と結合
        for ext_r, r_tok in zip(ext_rewards, int_reward_tokens):
            if r_tok.numel() <= 1:
                norm_tok = torch.zeros_like(r_tok)
            else:
                mu = r_tok.mean()
                std = r_tok.std(unbiased=False) + 1e-6
                norm_tok = (r_tok - mu) / std
            
            int_scalar = norm_tok.mean().item() if norm_tok.numel() > 0 else 0.0
            agg_int_rewards.append(int_scalar)
            
            total_r = args.w_ext * float(ext_r) + args.w_int * float(int_scalar)
            combined_rewards.append(total_r)

        reward_tensors = [
            torch.tensor(r, device=device_policy, dtype=torch.float32)
            for r in combined_rewards
        ]

        # 4. PPO Step
        stats = trainer.step(query_tensors, response_tensors, reward_tensors)

        # 5. ログ情報の構築 (TRLの log_stats に渡すため)
        # batch 辞書を作って log_stats に渡すと、テキストなどもW&Bで見やすくなります
        batch_log = {
            "query": batch_prompts,
            "response": response_texts,
        }

        # カスタム指標を stats に追加 (W&Bのグラフ用)
        mean_ext = sum(ext_rewards) / max(len(ext_rewards), 1)
        mean_int = sum(agg_int_rewards) / max(len(agg_int_rewards), 1)
        mean_total = sum(combined_rewards) / max(len(combined_rewards), 1)

        stats["env/reward_mean"] = mean_total
        stats["env/reward_external_mean"] = mean_ext
        stats["env/reward_internal_mean"] = mean_int

        # ★ ここで W&B に送信 & コンソール出力
        trainer.log_stats(stats, batch_log, reward_tensors)

        if args.debug_samples:
             print(f"[STEP {step+1}] Ext: {mean_ext:.3f}, Int: {mean_int:.3f}, Tot: {mean_total:.3f}")

    print("\n=== DONE ===")
    # 最後に W&B セッションを終了
    wandb.finish()


if __name__ == "__main__":
    main()

    import time
    # 0.5秒間隔で20回鳴らす（10秒間鳴り響く）
    print("🔔 学習が終了しました！音が鳴ります！")
    for _ in range(10):
        print('\a')
        time.sleep(0.3)