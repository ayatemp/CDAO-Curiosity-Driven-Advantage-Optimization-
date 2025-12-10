"""
ppo_creative_subspace_train_forEachToken.py

Qwen + PPO + 外部RM + サブスペース内部報酬 で学習する叩き台コード。

ポイント：
- policy: Qwen/Qwen2.5-7B-Instruct（ValueHead付きラッパ）
- RM: OpenAssistant/reward-model-deberta-v3-large-v2（DeBERTaベース）
- 内部報酬: (prompt + response) 全トークンの最後層 hidden をサブスペースに射影し、
            そのノルムの平均を 1シーケンスの報酬として使用（※トークンごと情報は含む）
- PPO: trl.PPOTrainer を使用
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ============================================================
# 1. 設定
# ============================================================

@dataclass
class TrainConfig:
    # モデル名
    policy_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    reward_model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2"
    subspace_basis_path: str = "./div_basis.pt"

    # PPO 設定
    learning_rate: float = 1e-6
    ppo_batch_size: int = 4          # VRAM に応じて調整
    ppo_mini_batch_size: int = 2
    ppo_epochs: int = 2
    target_kl: float = 0.1
    gamma: float = 1.0
    lam: float = 0.95

    # 報酬の重み
    alpha_internal: float = 1.0   # サブスペース報酬
    beta_external: float = 1.0    # 外部RM報酬

    # 生成
    max_new_tokens: int = 128

    # 学習ループ
    train_steps: int = 300
    log_sample_every: int = 50
    save_every: int = 100

    save_dir: str = "./ppo_creative_subspace_ckpt_forEachToken"


cfg = TrainConfig()


# ============================================================
# 2. サブスペースモデル
# ============================================================

class DiversitySubspaceModel:
    """
    Representation Diversity Subspace:
    - basis: [k, D]
    - project: h -> z
    - norm: サブスペース上のノルム
    """
    def __init__(self, basis: torch.Tensor):
        self.basis = basis.to(torch.float32).to(DEVICE)  # [k, D]
        self.k, self.D = self.basis.shape

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [..., D]
        return: [..., k]
        """
        if h.dtype != self.basis.dtype:
            h = h.to(self.basis.dtype)
        assert h.shape[-1] == self.D, f"dim mismatch: {h.shape[-1]} vs {self.D}"

        orig_shape = h.shape[:-1]
        h_flat = h.reshape(-1, self.D)           # [N, D]
        z_flat = h_flat @ self.basis.t()         # [N, k]
        z = z_flat.reshape(*orig_shape, self.k)  # [..., k]
        return z

    def norm(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [..., D]
        return: [...], サブスペース上の L2 ノルム
        """
        z = self.project(h)          # [..., k]
        return torch.linalg.norm(z, dim=-1)  # [...]


# ============================================================
# 3. モデル・トークナイザ・サブスペース読込
# ============================================================

def load_models_and_subspace(cfg: TrainConfig):
    # policy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # policy model (ValueHead付き)
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.policy_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map={"": 0},
    )
    policy_model.eval()

    # RM用 tokenizer
    rm_tokenizer = AutoTokenizer.from_pretrained(cfg.reward_model_name)

    # reward model (SequenceClassification 前提)
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.reward_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map={"": 1},
    )
    rm_model.eval()

    # subspace basis (div_basis.pt)
    if cfg.subspace_basis_path.endswith(".pt"):
        basis = torch.load(cfg.subspace_basis_path, map_location="cpu")
    elif cfg.subspace_basis_path.endswith(".npy"):
        basis_np = np.load(cfg.subspace_basis_path)
        basis = torch.from_numpy(basis_np)
    else:
        raise ValueError(f"Unsupported basis file format: {cfg.subspace_basis_path}")

    if basis.ndim != 2:
        raise ValueError(f"basis must be 2D [k, D], got {basis.shape}")

    subspace_model = DiversitySubspaceModel(basis)

    print("Loaded policy, RM, and subspace basis.")
    return tokenizer, policy_model, rm_model, subspace_model, rm_tokenizer


tokenizer, policy_model, rm_model, subspace_model, rm_tokenizer = load_models_and_subspace(cfg)


# ============================================================
# 4. PPOTrainer 設定
# ============================================================

ppo_config = PPOConfig(
    model_name=cfg.policy_model_name,
    learning_rate=cfg.learning_rate,
    batch_size=cfg.ppo_batch_size,
    mini_batch_size=cfg.ppo_mini_batch_size,
    ppo_epochs=cfg.ppo_epochs,
    target_kl=cfg.target_kl,
    gamma=cfg.gamma,
    lam=cfg.lam,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=None,  # 必要なら別途 ref_model を用意
    tokenizer=tokenizer,
)

print("PPOTrainer initialized.")


# ============================================================
# 5. プロンプト生成（ML研究アイデア用）
# ============================================================

BASE_INSTRUCTION = """\
You are a machine learning researcher.
Given the following topic, propose ONE concise research idea (2-4 sentences).
Avoid very generic ideas; make it slightly specific but still simple.
Write in English.
"""

ml_topics = [
    "data augmentation for image classification",
    "robustness against adversarial examples",
    "self-supervised learning for time series",
    "efficient fine-tuning methods for large language models",
    "reinforcement learning for recommendation systems",
    "uncertainty estimation in deep neural networks",
    "multi-modal learning with text and images",
    "domain adaptation for medical imaging",
    "continual learning without catastrophic forgetting",
    "explainability methods for black-box models",
]


def build_prompt(topic: str) -> str:
    return BASE_INSTRUCTION + f"\n\nTopic: {topic}\n\nResearch idea:"


def sample_prompts(batch_size: int) -> List[str]:
    idx = np.random.choice(len(ml_topics), size=batch_size, replace=True)
    return [build_prompt(ml_topics[i]) for i in idx]


# ============================================================
# 6. 外部RM報酬の計算
# ============================================================

@torch.no_grad()
def compute_external_rewards(
    rm_model: nn.Module,
    rm_tokenizer,
    prompts: List[str],
    responses: List[str],
) -> torch.Tensor:
    """
    RM に (prompt + response) を入れてスコアを1つ出す。
    DeBERTa系 RM にはその tokenizer を使うことが重要。
    """
    texts = []
    for p, r in zip(prompts, responses):
        texts.append(p + "\n\n" + r)

    inputs = rm_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    out = rm_model(**inputs)
    logits = out.logits  # [B, num_labels] or [B, 1]

    if logits.shape[-1] == 1:
        rewards = logits.squeeze(-1)
    else:
        rewards = logits[:, 0]

    return rewards.detach()  # [B]


# ============================================================
# 7. 内部サブスペース報酬の計算
# ============================================================

@torch.no_grad()
def compute_internal_rewards(
    policy_model: nn.Module,
    policy_tokenizer,
    prompts: List[str],
    responses: List[str],
    subspace_model: DiversitySubspaceModel,
) -> torch.Tensor:
    """
    policy_model の「中身のベースLM」を直接呼んで hidden_states を取る。
    (AutoModelForCausalLMWithValueHead の wrapper をバイパスする)
    """

    # 1) テキストを結合
    texts = []
    for p, r in zip(prompts, responses):
        texts.append(p + r)

    inputs = policy_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    # 2) value-head ラッパーの中にある元モデルを取り出す
    base_model = getattr(policy_model, "pretrained_model", None)
    if base_model is None:
        base_model = getattr(policy_model, "model", None)
    if base_model is None:
        base_model = getattr(policy_model, "transformer", None)
    if base_model is None:
        # 最悪そのまま（wrapper）を使う
        base_model = policy_model

    # 3) 元モデルで hidden_states を取得
    out = base_model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = out.hidden_states          # Tuple[num_layers+1, B, T, D]
    last_hidden = hidden_states[-1]            # [B, T, D]

    # 4) サブスペースノルムの平均を報酬にする
    norms = subspace_model.norm(last_hidden)   # [B, T]
    r_internal = norms.mean(dim=-1)            # [B]

    return r_internal.detach()


# ============================================================
# 8. 評価用サンプル出力
# ============================================================

@torch.no_grad()
def generate_sample_text(
    step: int,
    policy_model: nn.Module,
    policy_tokenizer,
    topic: str = "efficient fine-tuning methods for large language models",
) -> None:
    prompt = build_prompt(topic)
    inputs = policy_tokenizer(prompt, return_tensors="pt").to(DEVICE)

    gen_kwargs = {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 1.0,
        "pad_token_id": policy_tokenizer.eos_token_id,
    }

    output_ids = policy_model.generate(**inputs, **gen_kwargs)[0]
    full_text = policy_tokenizer.decode(output_ids, skip_special_tokens=True)

    print("\n" + "=" * 80)
    print(f"[Sample output at step {step}]")
    print("Topic:", topic)
    print("Prompt:\n", prompt)
    print("\nModel output:\n")
    print(full_text)
    print("=" * 80 + "\n")


# ============================================================
# 9. PPO 学習ループ
# ============================================================

def train(cfg: TrainConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)

    for step in range(cfg.train_steps):
        # ---- 1. プロンプトのサンプリング ----
        prompts = sample_prompts(cfg.ppo_batch_size)

        # ---- 2. クエリ（プロンプト）のエンコード ----
        batch = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        # PPOTrainer.generate は「List[Tensor(seq_len)]」を期待
        query_tensors = [q for q in batch["input_ids"]]  # List[Tensor(T,)]

        # ---- 3. 応答生成 ----
        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        # response_tensors: List[Tensor(T_resp,)]

        # ---- 4. テキスト化（prompt 部分を切り落とす）----
        responses: List[str] = []
        for rt, prompt in zip(response_tensors, prompts):
            full_text = tokenizer.decode(rt, skip_special_tokens=True)
            # ざっくり文字列長で prompt 部分を切り落とす（厳密にやりたければ token ベースでやる）
            resp = full_text[len(prompt):]
            responses.append(resp)

        # ---- 5. 外部RM報酬 ----
        ext_rewards = compute_external_rewards(
            rm_model,
            rm_tokenizer,
            prompts,
            responses,
        )  # [B]

        # ---- 6. 内部サブスペース報酬 ----
        int_rewards = compute_internal_rewards(
            policy_model,
            tokenizer,
            prompts,
            responses,
            subspace_model,
        )  # [B]

        # ---- 7. 総合報酬 ----
        rewards = cfg.alpha_internal * int_rewards + cfg.beta_external * ext_rewards
        rewards = rewards.to(DEVICE)  # [B]

        reward_tensors = [r.unsqueeze(0) for r in rewards]  # List[Tensor([1]), ...]

        # ---- 8. PPO update ----
        stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
        ppo_trainer.log_stats(stats, {"prompts": prompts, "responses": responses}, rewards)

        # ---- 9. 進捗ログ ----
        mean_ext = ext_rewards.mean().item()
        mean_int = int_rewards.mean().item()
        mean_tot = rewards.mean().item()
        print(
            f"[step {step+1:04d}/{cfg.train_steps}] "
            f"ext_reward={mean_ext:.4f}  int_reward={mean_int:.4f}  total={mean_tot:.4f}"
        )

        # ---- 10. サンプル出力 ----
        if (step + 1) % cfg.log_sample_every == 0:
            generate_sample_text(step + 1, policy_model, tokenizer)

        # ---- 11. 定期的な保存 ----
        if (step + 1) % cfg.save_every == 0:
            save_path = os.path.join(cfg.save_dir, f"step_{step+1}")
            os.makedirs(save_path, exist_ok=True)
            ppo_trainer.model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"[step {step+1}] Saved checkpoint to {save_path}")


if __name__ == "__main__":
    train(cfg)