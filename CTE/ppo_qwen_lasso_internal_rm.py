#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PPO + (Lasso-based Internal Creativity Reward) + (外部RLHF RM) with WritingPrompts

- Policy: Qwen/Qwen2.5-7B-Instruct (bf16, 非量子化)
- Internal RM: Lasso で学習した hidden → creativity スコア
- External RM: 例えば OpenAssistant/reward-model-deberta-v3-large-v2
- 報酬: r_total = internal_coef * r_int + rm_coef * r_rm

generate スタイルは「セル2」と同じ:
- WritingPrompts から prompt をサンプリング
- policy.pretrained_model.generate(..., output_hidden_states=True, return_dict_in_generate=True)
- hidden_states[-1] から各レイヤの「生成部分平均ベクトル」を集める
"""

import argparse
import os
import random
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)
from peft import LoraConfig, get_peft_model


# ============================================================
# ユーティリティ
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


# ============================================================
# モデル構築 (非量子化, bf16)
# ============================================================
def build_policy(model_id: str, device_index: int = 0, bf16: bool = True):
    """
    HF の自動 quantization (bnb 4bit) を完全に無効化して、
    普通の CausalLM (bf16/fp16) + ValueHead モデルとしてロードする。
    """
    dev = f"cuda:{device_index}"

    # 1) Config をロードして quantization_config を潰す
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        print("[INFO] removing quantization_config from config to avoid bitsandbytes")
        config.quantization_config = None

    # 2) ValueHead 付きモデルとしてそのままロード
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_id,
        config=config,
        dtype=torch.bfloat16 if bf16 else torch.float16,  # torch_dtype は deprecated
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map={"": dev},
    )

    # 省メモリ設定
    if hasattr(policy.pretrained_model, "gradient_checkpointing_enable"):
        policy.pretrained_model.gradient_checkpointing_enable()
    if hasattr(policy.pretrained_model, "config"):
        policy.pretrained_model.config.use_cache = False
        policy.pretrained_model.config.output_hidden_states = True

    return policy


def apply_lora_to_policy(policy: AutoModelForCausalLMWithValueHead, peft_cfg: LoraConfig):
    base = policy.pretrained_model
    peft_base = get_peft_model(base, peft_cfg)
    policy.pretrained_model = peft_base
    policy.is_gradient_checkpointing = True
    peft_base.is_gradient_checkpointing = True
    return policy


def build_policy_and_ref_dual_gpu(
    model_id: str,
    policy_gpu: int = 0,
    ref_gpu: int = 1,
    bf16: bool = True,
):
    # policy: GPU(policy_gpu)
    policy = build_policy(model_id, device_index=policy_gpu, bf16=bf16)

    # LoRA 事前適用
    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj",
        ],
    )
    policy = apply_lora_to_policy(policy, peft_cfg)

    # reference = policy のコピー → GPU(ref_gpu)
    ref = create_reference_model(policy)
    ref = ref.to(f"cuda:{ref_gpu}")

    return policy, ref


# ============================================================
# Lasso RM のロード & 内部報酬計算
# ============================================================
def load_lasso_rm_state(path: str, device: torch.device):
    # PyTorch 2.6 以降は weights_only=True がデフォルトなので、
    # 自前で作った state を読むときは明示的に False にする
    state = torch.load(path, map_location="cpu", weights_only=False)

    required_keys = [
        "num_layers",
        "hidden_dim",
        "scaler_X_mean",
        "scaler_X_scale",
        "scaler_y_mean",
        "scaler_y_scale",
        "coef",
        "intercept",
    ]
    for k in required_keys:
        if k not in state:
            raise RuntimeError(f"lasso_rm_state missing key: {k}")

    num_layers = int(state["num_layers"])
    hidden_dim = int(state["hidden_dim"])

    scaler_X_mean = torch.tensor(state["scaler_X_mean"], dtype=torch.float32, device=device)
    scaler_X_scale = torch.tensor(state["scaler_X_scale"], dtype=torch.float32, device=device)
    coef = torch.tensor(state["coef"], dtype=torch.float32, device=device)
    intercept = torch.tensor(state["intercept"], dtype=torch.float32, device=device)

    scaler_y_mean = float(state["scaler_y_mean"])
    scaler_y_scale = float(state["scaler_y_scale"])

    if scaler_X_mean.shape[0] != num_layers * hidden_dim:
        raise RuntimeError("scaler_X_mean の次元と num_layers/hidden_dim が一致しません")
    if coef.shape[0] != num_layers * hidden_dim:
        raise RuntimeError("coef の次元と num_layers/hidden_dim が一致しません")

    return {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "scaler_X_mean": scaler_X_mean,
        "scaler_X_scale": scaler_X_scale,
        "coef": coef,
        "intercept": intercept,
        "scaler_y_mean": scaler_y_mean,
        "scaler_y_scale": scaler_y_scale,
    }


def compute_lasso_internal_reward_from_hidden_single_exact(
    output_hidden_states: tuple,
    input_ids: torch.Tensor,
    lasso_state: Dict[str, Any],
) -> torch.Tensor:
    """
    ★ 学習時 (all_gen_layers) と完全に同じ実装 ★

    - output_hidden_states: generate(..., output_hidden_states=True,
        return_dict_in_generate=True) の .hidden_states
        → 長さ = gen_len の tuple
        → 各要素は「その生成ステップにおける (num_layers, [B, seq, D])」の tuple
    - 我々は「最後のステップ」の hidden_states[-1] を取り、
      各レイヤごとの「生成部分だけの平均ベクトル」を取る

    手順:
    1. last_step_hiddens = output_hidden_states[-1]
       → tuple(num_layers), それぞれ [1, seq_len_total, hidden_dim]
    2. 入力長 input_len を使って生成長 gen_len = seq_len_total - input_len
    3. 各レイヤについて:
         gen_hidden = layer_hidden[0, -gen_len:, :]
         gen_mean   = gen_hidden.mean(dim=0)   # [hidden_dim]
       を集める → [num_layers, hidden_dim]
    4. flatten → [num_layers * hidden_dim]
    5. StandardScaler (mean/scale) で標準化 → Lasso の coef + intercept
       → y_pred_std を「内部報酬」として利用
    """
    # 1ステップごとの hidden_states のうち、最後のステップ
    last_step_hiddens = output_hidden_states[-1]  # tuple(num_layers)
    num_layers_model = len(last_step_hiddens)

    batch_size, seq_len_total, hidden_dim_model = last_step_hiddens[0].shape
    assert batch_size == 1, "この関数は batch=1 前提です"

    ls_num_layers = lasso_state["num_layers"]
    ls_hidden_dim = lasso_state["hidden_dim"]
    if ls_num_layers != num_layers_model:
        raise RuntimeError(f"num_layers mismatch: lasso={ls_num_layers}, model={num_layers_model}")
    if ls_hidden_dim != hidden_dim_model:
        raise RuntimeError(f"hidden_dim mismatch: lasso={ls_hidden_dim}, model={hidden_dim_model}")

    scaler_X_mean = lasso_state["scaler_X_mean"]
    scaler_X_scale = lasso_state["scaler_X_scale"]
    coef = lasso_state["coef"]
    intercept = lasso_state["intercept"]

    # 入力長と生成長
    input_len = input_ids.shape[1]  # [1, seq_in]
    gen_len = max(1, seq_len_total - input_len)

    feats = []
    for layer_hidden in last_step_hiddens:
        # layer_hidden: [1, seq_len_total, hidden_dim]
        h = layer_hidden[0, :, :]              # [seq_len_total, hidden_dim]
        gen_hidden = h[-gen_len:, :]           # [gen_len, hidden_dim]
        gen_mean = gen_hidden.mean(dim=0)      # [hidden_dim]
        feats.append(gen_mean)

    # (num_layers, hidden_dim) → flatten
    X = torch.stack(feats, dim=0).reshape(-1)   # [num_layers*hidden_dim]

    # 標準化: (X - mean) / scale
    X_std = (X - scaler_X_mean) / (scaler_X_scale + 1e-8)

    # Lasso 線形予測（標準化空間の y_pred_std を reward として使用）
    y_pred_std = torch.dot(X_std, coef) + intercept  # scalar
    y_pred_std = torch.clamp(y_pred_std, -3.0, 3.0)

    return y_pred_std.view(1)  # [1]


# ============================================================
# メイン
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--lasso_state_path", required=True, help="torch.save した lasso_rm_state のパス")
    ap.add_argument("--total_updates", type=int, default=50, help="PPO のステップ数")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--kl_coef", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--policy_gpu", type=int, default=0)
    ap.add_argument("--ref_gpu", type=int, default=1)

    # 内部 & 外部 RM の重み
    ap.add_argument("--internal_coef", type=float, default=1.0,
                    help="内部報酬のスケール: r_total に対する係数 (w_int)")
    ap.add_argument("--rm_coef", type=float, default=1.0,
                    help="外部RM報酬のスケール: r_total に対する係数 (w_rm)")

    # 外部 RM の設定
    ap.add_argument("--rm_model_name", type=str,
                    default="OpenAssistant/reward-model-deberta-v3-large-v2",
                    help="外部 RLHF RM のモデル名")
    ap.add_argument("--rm_gpu", type=int, default=None,
                    help="外部 RM を載せる GPU。None の場合 ref_gpu と同じ")

    ap.add_argument("--num_prompts", type=int, default=200,
                    help="WritingPrompts からサンプリングする prompt 数")
    ap.add_argument("--display_every", type=int, default=10,
                    help="何ステップごとに prompt / generated / reward / KL を print するか")
    ap.add_argument("--debug_stats_every", type=int, default=20,
                    help="何ステップごとに r_int / r_rm の統計と相関を表示するか")
    ap.add_argument("--save_dir", type=str, default="ppo_internal_rm_ckpt")
    args = ap.parse_args()

    set_seed(args.seed)

    # ========================================================
    # WritingPrompts データセットから prompt をサンプリング
    # ========================================================
    print("[INFO] loading WritingPrompts dataset...")
    wp_ds = load_dataset("euclaise/writingprompts")
    train_ds = wp_ds["train"]
    print("[INFO] train size:", len(train_ds))

    if args.num_prompts > len(train_ds):
        args.num_prompts = len(train_ds)

    indices = random.sample(range(len(train_ds)), args.num_prompts)
    prompts: List[str] = [train_ds[i]["prompt"] for i in indices]

    print(f"[INFO] sampled {len(prompts)} prompts from WritingPrompts")
    print("----- sample prompts -----")
    for i, p in enumerate(prompts[:3]):
        print(f"--- prompt {i} ---")
        print(p)
        print("-------------------------")

    # ========================================================
    # tokenizer / Policy & Ref 構築
    # ========================================================
    tok = build_tokenizer(args.model_id)
    policy, ref_model = build_policy_and_ref_dual_gpu(
        args.model_id,
        policy_gpu=args.policy_gpu,
        ref_gpu=args.ref_gpu,
        bf16=True,
    )
    policy_device = next(policy.parameters()).device
    print("[DEVICES] policy:", policy_device, "ref:", next(ref_model.parameters()).device)

    # Lasso RM state のロード（内部 RM）
    lasso_state = load_lasso_rm_state(args.lasso_state_path, device=policy_device)
    print("[INFO] Loaded Lasso RM from", args.lasso_state_path)
    print("[INFO] Lasso num_layers =", lasso_state["num_layers"],
          "hidden_dim =", lasso_state["hidden_dim"])

    # ========================================================
    # 外部 RM のロード
    # ========================================================
    rm_gpu = args.rm_gpu if args.rm_gpu is not None else args.ref_gpu
    rm_device = torch.device(f"cuda:{rm_gpu}")
    print(f"[INFO] loading external RM: {args.rm_model_name} on {rm_device}")

    rm_tok = AutoTokenizer.from_pretrained(args.rm_model_name)
    if rm_tok.pad_token_id is None and rm_tok.eos_token_id is not None:
        rm_tok.pad_token = rm_tok.eos_token
    rm_tok.padding_side = "right"

    rm_model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_model_name
    ).to(rm_device)
    rm_model.eval()
    for p in rm_model.parameters():
        p.requires_grad = False

    # ========================================================
    # PPO 設定 & Trainer
    # ========================================================
    cfg_kwargs = dict(
        learning_rate=args.lr,
        batch_size=1,          # 1 サンプルずつ
        mini_batch_size=1,
        target_kl=0.1,
        init_kl_coef=args.kl_coef,
    )
    try:
        # 新しめの TRL だと ppo_epochs が有効
        ppo_cfg = PPOConfig(ppo_epochs=1, **cfg_kwargs)
    except TypeError:
        # 少し古い TRL だと num_ppo_epochs しかない
        ppo_cfg = PPOConfig(num_ppo_epochs=1, **cfg_kwargs)

    trainer = PPOTrainer(
        config=ppo_cfg,
        model=policy,
        tokenizer=tok,
    )

    print("=== start PPO training (Lasso internal RM + external RM + WritingPrompts) ===")

    # ========================================================
    # デバッグ用ヒストリー
    # ========================================================
    hist_r_int: List[float] = []
    hist_r_rm: List[float] = []
    hist_r_total: List[float] = []
    hist_kl: List[float] = []

    def print_debug_stats(tag: str):
        """r_int / r_rm / r_total の統計と相関を表示"""
        if len(hist_r_int) < 5:
            print(f"\n[DEBUG STATS {tag}] samples < 5, skip stats\n")
            return

        r_int_arr = np.array(hist_r_int, dtype=np.float64)
        r_rm_arr = np.array(hist_r_rm, dtype=np.float64)
        r_tot_arr = np.array(hist_r_total, dtype=np.float64)

        def fmt_stats(name, arr):
            return (
                f"{name}: mean={arr.mean():+.4f}, std={arr.std():.4f}, "
                f"min={arr.min():+.4f}, max={arr.max():+.4f}"
            )

        def safe_corr(a, b):
            try:
                return float(np.corrcoef(a, b)[0, 1])
            except Exception:
                return float("nan")

        print("\n==================== DEBUG STATS", tag, "====================")
        print(fmt_stats("r_int", r_int_arr))
        print(fmt_stats("r_rm ", r_rm_arr))
        print(fmt_stats("r_tot", r_tot_arr))
        print(f"corr(r_int, r_rm)  = {safe_corr(r_int_arr, r_rm_arr):+.4f}")
        print(f"corr(r_int, r_tot) = {safe_corr(r_int_arr, r_tot_arr):+.4f}")
        print(f"corr(r_rm,  r_tot) = {safe_corr(r_rm_arr,  r_tot_arr):+.4f}")
        print("=========================================================\n")

    # ========================================================
    # PPO ループ（tqdm＋ステップごとのログ）
    # ========================================================
    pbar = tqdm(range(args.total_updates), desc="PPO steps", dynamic_ncols=True)
    num_prompts = len(prompts)

    for step in pbar:
        # 1サンプル: インデックスを決めて prompt を選ぶ
        idx = step % num_prompts
        prompt = prompts[idx]

        # --- generate (セル2スタイル) ---
        inputs = tok(prompt, return_tensors="pt").to(policy_device)

        with torch.no_grad():
            output = policy.pretrained_model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # =========================
        # 内部報酬 (Lasso RM)
        # =========================
        r_int = compute_lasso_internal_reward_from_hidden_single_exact(
            output_hidden_states=output.hidden_states,
            input_ids=inputs["input_ids"],
            lasso_state=lasso_state,
        )  # [1], policy_device 上想定

        # =========================
        # 外部 RM 報酬
        # =========================
        # response_only 部分をテキスト化
        full_seq = output.sequences[0]          # [seq_total]
        query_tensor = inputs["input_ids"][0]   # [seq_in]
        seq_in = query_tensor.size(0)
        response_only = full_seq[seq_in:]       # [gen_len]
        if response_only.numel() == 0:
            response_only = full_seq[-1:].clone()

        response_text = tok.decode(response_only, skip_special_tokens=True)

        rm_input_text = f"Human: {prompt}\nAssistant: {response_text}"

        with torch.no_grad():
            rm_enc = rm_tok(
                rm_input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(rm_device)

            rm_out = rm_model(**rm_enc)
            if hasattr(rm_out, "logits"):
                rm_logits = rm_out.logits
            else:
                rm_logits = rm_out[0]
            r_rm = rm_logits.squeeze()  # スカラー想定（B=1）
            if r_rm.ndim > 0:
                r_rm = r_rm.mean()
        # policy_device 側に合わせる
        r_rm = r_rm.to(policy_device, dtype=torch.float32).view(1)   # [1]

        # =========================
        # 合成報酬
        # =========================
        r_int = r_int.to(policy_device, dtype=torch.float32)   # [1]
        r_total = args.internal_coef * r_int + args.rm_coef * r_rm  # [1]

        # TRL 期待フォーマットに合わせる
        query_list = [query_tensor]          # List[Tensor]
        response_list = [response_only]      # List[Tensor]
        reward_list = [r_total[0]]           # List[Tensor] (各要素はスカラー)

        # ===== PPO update =====
        stats = trainer.step(query_list, response_list, reward_list)
        trainer.log_stats(
            stats,
            batch={"query": query_list, "response": response_list},
            rewards=reward_list,
        )

        # KL を stats から拾う
        kl_val = None
        for k in ["kl", "ppo/kl", "objective/kl", "global/kl"]:
            if k in stats:
                v = stats[k]
                if isinstance(v, torch.Tensor):
                    kl_val = float(v.mean().item())
                elif isinstance(v, (list, tuple)) and len(v) > 0:
                    x = v[0]
                    if isinstance(x, torch.Tensor):
                        kl_val = float(x.mean().item())
                    else:
                        try:
                            kl_val = float(x)
                        except Exception:
                            kl_val = None
                else:
                    try:
                        kl_val = float(v)
                    except Exception:
                        kl_val = None
                break

        r_int_val = float(r_int.item())
        r_rm_val = float(r_rm.item())
        r_tot_val = float(r_total.item())

        # ヒストリーに保存
        hist_r_int.append(r_int_val)
        hist_r_rm.append(r_rm_val)
        hist_r_total.append(r_tot_val)
        if kl_val is not None:
            hist_kl.append(kl_val)

        # tqdm に今の reward / KL を表示
        pbar.set_postfix({
            "r_int": r_int_val,
            "r_rm": r_rm_val,
            "r_total": r_tot_val,
            "kl": kl_val if kl_val is not None else float("nan"),
        })

        # ===============================
        # display_every ステップごとに詳細ログを表示
        # ===============================
        if step % args.display_every == 0:
            gen_text_full = tok.decode(full_seq, skip_special_tokens=True)

            print("\n==============================")
            print(f"[step {step}]")
            print("=== Prompt (WritingPrompts) ===")
            print(prompt)
            print("\n=== Generated (policy output, full) ===")
            print(gen_text_full)
            print("\n=== Response-only (for RM) ===")
            print(response_text)
            print("\n=== Internal Reward (Lasso RM, raw y_pred_std) ===")
            print(f"r_int = {r_int_val:.4f}")
            print("\n=== External RM Reward (raw) ===")
            print(f"r_rm  = {r_rm_val:.4f}")
            print("\n=== Total Reward (PPO に渡した値) ===")
            print(
                f"r_total = internal_coef * r_int + rm_coef * r_rm "
                f"= {args.internal_coef} * {r_int_val:.4f} + {args.rm_coef} * {r_rm_val:.4f} "
                f"= {r_tot_val:.4f}"
            )
            print("\n=== KL (from PPO stats, if available) ===")
            print("kl =", "N/A" if kl_val is None else f"{kl_val:.6f}")
            print("==============================\n")

        # ===============================
        # debug_stats_every ごとに統計＆相関を表示
        # ===============================
        if (step + 1) % args.debug_stats_every == 0:
            print_debug_stats(tag=f"step {step+1}")

    # ループ終了後にも最終統計を表示
    print_debug_stats(tag="final")

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    policy.pretrained_model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    print(f"[SAVE] policy + tokenizer saved to {save_dir}")


if __name__ == "__main__":
    main()