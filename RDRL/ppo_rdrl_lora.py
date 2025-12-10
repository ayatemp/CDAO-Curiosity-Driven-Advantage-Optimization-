#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ppo_rdrl_lora.py

RDRL: Representation-Diversity RL (LoRA 版)
 - Qwen2.5-7B-Instruct + LoRA + ValueHead を PPO で更新
 - 内部多様性サブスペース (div_basis.pt) による「内発的報酬」
 - 外部 RM (DeBERTa reward model) による「外発的報酬」
 - 報酬はトークンごと (per-token) に設定
"""

import argparse
import random
import textwrap
from typing import List, Tuple

import torch
from torch import nn

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
        # ここでは dtype はそのまま保存（float32 のままで OK）
        self.register_buffer("basis", ortho)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [..., D]
        return: [..., k]  (サブスペース上の座標)
        """
        D = self.basis.shape[1]
        assert h.shape[-1] == D, f"expected last dim {D}, got {h.shape[-1]}"

        # h の device / dtype に合わせて basis をキャスト
        basis = self.basis.to(device=h.device, dtype=h.dtype)  # ★ここがポイント

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
    """
    任意の (batch, num_labels) logits から 1D スコアを作るヘルパー。

    - (B, 1) -> squeeze
    - (B, 2) -> logits[:,1] - logits[:,0]
    - それ以外 -> 平均
    """
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
    ここでは簡単に
        "User: <problem>\n\nAssistant: <response>"
    を RM に渡す。
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
    各サンプルごとに
      - full_text = prompt + response
      - policy_model(pretrained) に通して hidden_states を取得
      - target_layer の hidden_states をサブスペースに射影し、tokenごとのノルムをスコアにする
      - prompt 部分は捨てて、response 部分だけスコアを返す

    戻り値: List[Tensor], 各要素は shape (T_resp,) の 1D tensor
    """
    # ValueHead ラッパーからベースモデルを取る
    base_model = policy_model.pretrained_model  # TRL の標準実装

    device = next(policy_model.parameters()).device

    # まず full_text を作る
    texts = []
    for p, resp_ids in zip(prompts, responses_ids):
        # resp_ids を decode してフルテキストにする方法もあるが、
        # ここでは再tokenizeで整合をとるため、単純に decode する。
        resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
        full = p + resp_text
        texts.append(full)

    # 一括で tokenization
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    # モデル forward (hidden_states を取得)
    out = base_model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = out.hidden_states  # tuple[num_layers+1] of [B, T, D]
    h_layer = hidden_states[target_layer]  # [B, T, D]

    # 各サンプルごとに「prompt length」と「response length」を再計算するため、
    # もう一度 prompt+response を token 化して boundary を求める。
    rewards_per_sample: List[torch.Tensor] = []

    for i, (p, resp_ids) in enumerate(zip(prompts, responses_ids)):
        # prompt のみを token 化して長さを取得
        enc_prompt = tokenizer(
            p,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        prompt_len = enc_prompt["input_ids"].shape[1]

        # full での長さ
        full_len = (enc["attention_mask"][i] == 1).sum().item()

        # response 部分の token index 範囲
        start = prompt_len
        end = full_len  # [start, end) が response
        if start >= end:
            # 万一 prompt が長すぎて response が潰れていた場合は、
            # とりあえず最後の1トークンだけ使う
            start = max(0, full_len - 1)

        h_resp = h_layer[i, start:end, :]  # [T_resp, D]

        # サブスペース上の多様性スコア (L2 ノルム)
        scores = subspace_model.token_diversity(h_resp)  # [T_resp]
        rewards_per_sample.append(scores.detach())

    return rewards_per_sample


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # Policy
    parser.add_argument(
        "--policy-model-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
    )

    # 外部 RM
    parser.add_argument(
        "--rm-model-name",
        type=str,
        default="OpenAssistant/reward-model-deberta-v3-large-v2",
    )

    # サブスペース basis
    parser.add_argument(
        "--subspace-basis-path",
        type=str,
        default="div_basis.pt",  # Notebook で保存した basis
    )

    # PPO
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    # 報酬の重み
    parser.add_argument("--w-ext", type=float, default=0.7,
                        help="外部 RM 報酬の重み")
    parser.add_argument("--w-int", type=float, default=0.3,
                        help="内部多様性報酬の重み (サブスペース)")

    parser.add_argument("--debug-samples", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    print(f"[INFO] policy model : {args.policy_model_name}")
    print(f"[INFO] RM model     : {args.rm_model_name}")
    print(f"[INFO] subspace path: {args.subspace_basis_path}")
    print(f"[INFO] w_ext={args.w_ext}, w_int={args.w_int}")

    # --------------------------------------------------------
    # Policy + LoRA + ValueHead
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

    # trainable params 確認
    trainable, total = 0, 0
    for n, p in policy_model.named_parameters():
        n_params = p.numel()
        total += n_params
        if p.requires_grad:
            trainable += n_params
    print(
        f"[INFO] trainable params: {trainable} / {total} "
        f"({100.0 * trainable / total:.4f}%)"
    )

    # PPO 設定
    cfg_kwargs = dict(
        learning_rate=1e-5,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        target_kl=0.1,
        init_kl_coef=0.02,
    )
    try:
        ppo_config = PPOConfig(ppo_epochs=1, **cfg_kwargs)
    except TypeError:
        ppo_config = PPOConfig(num_ppo_epochs=1, **cfg_kwargs)

    trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        tokenizer=tok_policy,
    )
    device_policy = trainer.accelerator.device
    print(f"[INFO] PPO accelerator device: {device_policy}")

    # --------------------------------------------------------
    # サブスペース basis をロード
    # --------------------------------------------------------
    basis = torch.load(args.subspace_basis_path, map_location="cpu")
    basis = basis.to(device_policy)
    subspace_model = DiversitySubspaceModel(basis).to(device_policy)
    print(f"[INFO] subspace basis shape: {basis.shape}")

    # --------------------------------------------------------
    # 外部 RM を CPU でロード
    # --------------------------------------------------------
    rm_device = torch.device("cpu")
    print(f"[INFO] RM device: {rm_device}")

    rm_tokenizer = AutoTokenizer.from_pretrained(args.rm_model_name, use_fast=True)
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_model_name,
        torch_dtype=torch.float32,
    ).to(rm_device)
    rm_model.eval()

    # --------------------------------------------------------
    # プロンプト群（研究お題）
    # --------------------------------------------------------
    base_inst = (
        "You are an expert LLM researcher. Propose a novel and concrete research idea "
        "about large language models.\n"
        "Output ONLY in the following format:\n\n"
        "Title: <concise LLM research title>\n"
        "Abstract: <150-220 word abstract with motivation, approach, and contribution>\n"
    )

    problems = [
        base_inst + "\nFocus on: alignment and safety.",
        base_inst + "\nFocus on: multi-agent LLM collaboration.",
        base_inst + "\nFocus on: efficient training / inference.",
        base_inst + "\nFocus on: world models / model-based RL in LLMs.",
        base_inst + "\nFocus on: tool use and API-calling LLMs.",
        base_inst + "\nFocus on: long-context reasoning and memory.",
    ] * 4

    # --------------------------------------------------------
    # PPO ループ
    # --------------------------------------------------------
    for step in range(args.num_steps):
        print("\n" + "=" * 80)
        print(f"[PPO STEP {step+1}/{args.num_steps}]")

        batch_prompts = random.sample(problems, k=ppo_config.batch_size)

        # ---- 1) query_tensors (prompts) ----
        enc = tok_policy(
            batch_prompts,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device_policy) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        # ---- 2) 生成 (policy_model.generate) ----
        with torch.no_grad():
            gen = policy_model.generate(
                **{
                    k: v
                    for k, v in enc.items()
                    if k in ["input_ids", "attention_mask"]
                },
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=tok_policy.pad_token_id,
            )

        # query / response に分解
        query_tensors = [row for row in enc["input_ids"]]  # List[Tensor(seq_len_prompt)]
        response_tensors = []
        response_texts = []

        for i in range(gen.size(0)):
            resp_ids = gen[i, prompt_len:]
            response_tensors.append(resp_ids)
            resp_text = tok_policy.decode(
                resp_ids,
                skip_special_tokens=True,
            )
            response_texts.append(resp_text)

        # ---- 3) 外部 RM 報酬（スカラー） ----
        ext_rewards = compute_external_rm_rewards(
            rm_model=rm_model,
            rm_tokenizer=rm_tokenizer,
            problems=batch_prompts,
            responses=response_texts,
            device=rm_device,
            max_length=512,
        )  # List[float], len=B

        # ---- 4) 内部多様性報酬（トークンごと） ----
        int_reward_tokens = compute_internal_diversity_rewards(
            policy_model=policy_model,
            tokenizer=tok_policy,
            prompts=batch_prompts,
            responses_ids=response_tensors,
            subspace_model=subspace_model,
            target_layer=-1,  # 最終層
            max_length=512,
        )  # List[Tensor(T_resp)]

        # 正規化（各サンプルごとに平均0, 分散1くらいに）
        norm_int_rewards: List[torch.Tensor] = []
        for r_tok in int_reward_tokens:
            if r_tok.numel() <= 1:
                norm_int_rewards.append(torch.zeros_like(r_tok))
            else:
                mu = r_tok.mean()
                std = r_tok.std(unbiased=False) + 1e-6
                norm_int_rewards.append((r_tok - mu) / std)

        # ---- 5) 内部報酬をスカラーに畳み込んでから reward_tensors を作る ----
        combined_rewards = []  # List[float]（バッチごと1スカラー）
        agg_int_rewards = []   # デバッグ用に内部報酬スカラーも保存

        for ext_r, int_r_tok in zip(ext_rewards, norm_int_rewards):
            # ★ここが「トークン→スカラー」の縮約ポイント
            # 例: 平均で潰す（ここは設計ポイントなので変えてもOK）
            int_scalar = int_r_tok.mean().item()
            agg_int_rewards.append(int_scalar)

            total_r = args.w_ext * float(ext_r) + args.w_int * float(int_scalar)
            combined_rewards.append(total_r)

        # PPOTrainer が期待している形式: List[0-dim Tensor]
        reward_tensors: List[torch.Tensor] = [
            torch.tensor(r, device=device_policy, dtype=torch.float32)
            for r in combined_rewards
        ]

        # ---- 6) debug 出力 ----
        mean_ext = sum(ext_rewards) / max(len(ext_rewards), 1)
        mean_int = sum(agg_int_rewards) / max(len(agg_int_rewards), 1)
        print(f"[STEP {step+1}] mean external RM reward: {mean_ext:.4f}")
        print(f"[STEP {step+1}] mean internal div reward (agg): {mean_int:.4f}")

        if args.debug_samples:
            print("\n--- Sample[0] prompt & response ---")
            print("PROMPT:")
            print(batch_prompts[0])
            print("\nRESPONSE:")
            print(response_texts[0])
            print(f"\nExternal reward: {ext_rewards[0]:.4f}")
            print(f"Internal reward (first 10 tokens, norm): {norm_int_rewards[0][:10].cpu().tolist()}")
            print(f"Internal reward (aggregated scalar)   : {agg_int_rewards[0]:.4f}")
            print(f"Total reward (ext+int)                : {combined_rewards[0]:.4f}")

        # ---- 7) PPO 更新 ----
        stats = trainer.step(
            query_tensors,
            response_tensors,
            reward_tensors,
        )

        kl = stats.get("objective/kl", stats.get("kl", "N/A"))
        loss = stats.get("loss/total", stats.get("loss/policy", "N/A"))
        print(f"[STEP {step+1}] KL   : {kl}")
        print(f"[STEP {step+1}] LOSS : {loss}")

    print("\n=== DONE ===")
    print(
        f"OK: {args.num_steps} PPO updates finished "
        f"with RDRL (LoRA, external RM + internal diversity subspace)."
    )


if __name__ == "__main__":
    main()