#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PPO + LoRA (no bitsandbytes) for Qwen2.5-7B with Reward Model (DeBERTa).
Final-stable version for: grad-checkpoint + LoRA issues + generation warnings.

Fixes included:
- tokenizer.padding_side = "left" (decoder-only correct padding)
- generate() runs under no_grad, so disable gradient checkpointing ONLY during generate
  to avoid: "None of the inputs have requires_grad=True. Gradients will be None"
- robust GC+LoRA grad enabling for PPO step
- generation path: policy.pretrained_model.generate() (NOT trainer.generate)
"""

import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead


# ----------------------------
# Prompt pool
# ----------------------------
STRUCTURED_PROMPTS = [
    {
        "category": "Synthesis",
        "topic_a": "Origami folding patterns",
        "topic_b": "Deployable space telescope mirrors",
        "instruction": "Propose a new mechanism for folding large telescope mirrors using origami techniques. Provide a specific name for this mechanism and explain its logical steps."
    },
    {
        "category": "Constraint",
        "topic_a": "No electricity environment",
        "topic_b": "High-precision medical cooling",
        "instruction": "Design a high-precision cooling system for vaccines that operates entirely without electricity. Explain the thermodynamic principles used in your design."
    },
    {
        "category": "Metaphor",
        "topic_a": "Fluid Dynamics",
        "topic_b": "Economic Hyperinflation",
        "instruction": "Model the phenomenon of economic hyperinflation using the equations or principles of fluid dynamics. Create a unique conceptual bridge between the two."
    },
]


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def freeze_(m: torch.nn.Module):
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()


def format_prompt(ex: Dict[str, Any]) -> str:
    return (
        "You are a creative but rigorous researcher.\n"
        f"[Category] {ex['category']}\n"
        f"[Topic A] {ex['topic_a']}\n"
        f"[Topic B] {ex['topic_b']}\n"
        f"[Task] {ex['instruction']}\n"
        "Answer with:\n"
        "1) A short, unique name\n"
        "2) Step-by-step mechanism\n"
        "3) Why it is logically consistent\n"
    )


@dataclass
class Batch:
    prompts: List[str]


def sample_batch(pool: List[Dict[str, Any]], batch_size: int) -> Batch:
    return Batch(prompts=[format_prompt(random.choice(pool)) for _ in range(batch_size)])


@torch.no_grad()
def rm_score_batch(
    rm_tok,
    rm_model,
    prompts: List[str],
    responses: List[str],
    device: torch.device,
    max_length: int = 1024,
) -> torch.Tensor:
    enc = rm_tok(
        prompts,
        responses,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(device)
    out = rm_model(**enc)
    return out.logits.squeeze(-1)


def save_lora_only(policy: AutoModelForCausalLMWithValueHead, tok, out_dir: str):
    ensure_dir(out_dir)
    policy.pretrained_model.save_pretrained(out_dir)  # PEFT adapter only
    tok.save_pretrained(out_dir)


# ----------------------------
# Robust fix for grad-checkpoint + LoRA
# ----------------------------
def force_input_grads_for_gc(pretrained_model):
    """Force embedding outputs to require_grad via a forward hook."""
    try:
        emb = pretrained_model.get_input_embeddings()
    except Exception:
        emb = None

    if emb is None:
        print("[warn] cannot get input embeddings; skip force_input_grads_for_gc")
        return

    def _hook(_module, _inputs, output):
        if isinstance(output, torch.Tensor) and (not output.requires_grad):
            return output.requires_grad_(True)
        return output

    emb.register_forward_hook(_hook)
    print("[fix] registered forward hook: embedding outputs will require_grad=True")


def robust_enable_grads_for_gc(policy: AutoModelForCausalLMWithValueHead, use_grad_checkpoint: bool):
    """
    Belt-and-suspenders fix:
    - enable_input_require_grads on peft and wrapper if available
    - force embedding weight requires_grad=True (optimizer still only updates LoRA params)
    - forward hook on embedding outputs
    - enable gradient checkpointing & disable cache
    """
    if hasattr(policy.pretrained_model, "enable_input_require_grads"):
        policy.pretrained_model.enable_input_require_grads()

    if hasattr(policy, "enable_input_require_grads"):
        policy.enable_input_require_grads()

    try:
        emb = policy.pretrained_model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            emb.weight.requires_grad_(True)
            print("[fix] forced input embedding weight requires_grad=True")
    except Exception as e:
        print(f"[warn] failed to force embedding weight grads: {e}")

    force_input_grads_for_gc(policy.pretrained_model)

    if use_grad_checkpoint:
        policy.pretrained_model.gradient_checkpointing_enable()
        policy.pretrained_model.config.use_cache = False


# ----------------------------
# Generation (avoid TRL trainer.generate) + disable GC only during generate
# ----------------------------
@torch.no_grad()
def generate_batch(
    tok,
    model,  # policy.pretrained_model (PEFT-wrapped)
    prompts: List[str],
    device: torch.device,
    max_prompt_len: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    use_cache: bool,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """
    Returns:
      query_tensors: list of 1D tensors (prompt ids, no pad)
      response_tensors: list of 1D tensors (response only)
      responses_text: list of decoded responses

    IMPORTANT:
      generation is under no_grad. If gradient checkpointing is enabled,
      PyTorch may warn: "None of the inputs have requires_grad=True" during generate.
      To avoid it, disable GC ONLY around model.generate().
    """
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
    ).to(device)

    # --- disable gradient checkpointing only during generate (no_grad) ---
    was_gc = getattr(model, "is_gradient_checkpointing", False)
    if was_gc:
        model.gradient_checkpointing_disable()

    full = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=use_cache,
    )

    if was_gc:
        model.gradient_checkpointing_enable()

    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]

    query_tensors: List[torch.Tensor] = []
    response_tensors: List[torch.Tensor] = []
    responses_text: List[str] = []

    for i in range(full.size(0)):
        q_len = int(attn[i].sum().item())  # number of non-pad tokens
        q_ids = input_ids[i, :q_len].detach()
        resp_ids = full[i, q_len:].detach()

        query_tensors.append(q_ids)
        response_tensors.append(resp_ids)
        responses_text.append(tok.decode(resp_ids, skip_special_tokens=True))

    return query_tensors, response_tensors, responses_text


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--rm-model", type=str, default="OpenAssistant/reward-model-deberta-v3-large-v2")

    ap.add_argument("--output-dir", type=str, default="./ckpts_ppo_rm_qwen_lora")
    ap.add_argument("--log-dir", type=str, default="./logs_ppo_rm_qwen_lora")
    ap.add_argument("--save-every", type=int, default=50)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=200)

    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--mini-batch-size", type=int, default=2)
    ap.add_argument("--ppo-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--cliprange", type=float, default=0.2)

    ap.add_argument("--max-prompt-len", type=int, default=1536)
    ap.add_argument("--gen-max-new-tokens", type=int, default=160)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--grad-checkpoint", action="store_true")

    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--print-samples", type=int, default=1)

    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target", type=str, default="q_proj,k_proj,v_proj,o_proj")

    ap.add_argument("--rm-max-length", type=int, default=1024)

    args = ap.parse_args()

    # smoke-test: keep small but NEVER batch_size=1
    if args.smoke_test:
        args.steps = min(args.steps, 5)
        args.gen_max_new_tokens = min(args.gen_max_new_tokens, 96)
        args.batch_size = max(args.batch_size, 2)
        args.mini_batch_size = min(args.mini_batch_size, args.batch_size)

    if args.batch_size < 2:
        raise ValueError("batch_size must be >= 2 (PPO stats/whitening stability).")

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(args.log_dir)
    log_path = os.path.join(args.log_dir, "train_log.jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # ----------------------------
    # Tokenizers
    # ----------------------------
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # IMPORTANT for decoder-only models: left padding
    tok.padding_side = "left"

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rm_tok = AutoTokenizer.from_pretrained(args.rm_model)

    # ----------------------------
    # Policy model + LoRA
    # ----------------------------
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)

    target_modules = [s.strip() for s in args.lora_target.split(",") if s.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    policy.pretrained_model = get_peft_model(policy.pretrained_model, lora_cfg)
    policy.pretrained_model.print_trainable_parameters()

    # Robust grad-checkpoint compatibility setup (training forward)
    robust_enable_grads_for_gc(policy, use_grad_checkpoint=args.grad_checkpoint)

    # ----------------------------
    # Ref model (for KL)
    # ----------------------------
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    freeze_(ref_model)

    # ----------------------------
    # Reward model (frozen)
    # ----------------------------
    rm = AutoModelForSequenceClassification.from_pretrained(args.rm_model).to(device)
    freeze_(rm)

    # ----------------------------
    # PPO config/trainer
    # ----------------------------
    ppo_cfg = PPOConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        cliprange=args.cliprange,
        log_with=None,
    )

    trainer = PPOTrainer(
        config=ppo_cfg,
        model=policy,
        ref_model=ref_model,
        tokenizer=tok,
        dataset=None,
        data_collator=None,
    )

    has_ref = hasattr(trainer, "ref_model") and trainer.ref_model is not None
    print(f"[sanity] trainer.ref_model is set: {has_ref}")

    # ----------------------------
    # Debug (step1 only): confirm grads
    # ----------------------------
    def debug_grads_once():
        try:
            emb = policy.pretrained_model.get_input_embeddings()
            emb_flag = bool(emb.weight.requires_grad)
        except Exception:
            emb_flag = False

        any_req = any(p.requires_grad for p in policy.pretrained_model.parameters())

        lora_on = 0
        for n, p in policy.pretrained_model.named_parameters():
            if "lora" in n.lower():
                lora_on += int(p.requires_grad)

        print(f"[debug] any parameter requires_grad: {any_req}")
        print(f"[debug] emb.weight.requires_grad: {emb_flag}")
        print(f"[debug] #trainable LoRA params: {lora_on}")

    # ----------------------------
    # Train loop
    # ----------------------------
    for step in range(1, args.steps + 1):
        batch = sample_batch(STRUCTURED_PROMPTS, args.batch_size)
        prompts = batch.prompts

        # generate (avoid TRL trainer.generate)
        query_tensors, response_tensors, responses_text = generate_batch(
            tok=tok,
            model=policy.pretrained_model,
            prompts=prompts,
            device=device,
            max_prompt_len=args.max_prompt_len,
            max_new_tokens=args.gen_max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            use_cache=(not args.grad_checkpoint),  # gc時はcache切るのが安全
        )

        if step == 1:
            debug_grads_once()

        # RM rewards
        rm_scores = rm_score_batch(
            rm_tok=rm_tok,
            rm_model=rm,
            prompts=prompts,
            responses=responses_text,
            device=device,
            max_length=args.rm_max_length,
        )
        rewards = [rm_scores[i].detach() for i in range(rm_scores.shape[0])]

        # PPO step
        trainer.step(query_tensors, response_tensors, rewards)

        # log
        if step % args.log_every == 0:
            mean_r = float(rm_scores.mean().item())
            row = {
                "step": step,
                "mean_rm_reward": mean_r,
                "rm_min": float(rm_scores.min().item()),
                "rm_max": float(rm_scores.max().item()),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(f"[step {step}] mean_rm_reward={mean_r:.4f} (min={row['rm_min']:.4f}, max={row['rm_max']:.4f})")

            n_show = min(args.print_samples, len(prompts))
            for k in range(n_show):
                print("\n" + "=" * 90)
                print(f"[sample {k}] rm_reward={float(rm_scores[k].item()):.4f}")
                print("- PROMPT ------------------------------------------------------------")
                print(prompts[k])
                print("- RESPONSE ----------------------------------------------------------")
                print(responses_text[k])
                print("=" * 90)

        # save
        if step % args.save_every == 0 or step == args.steps:
            save_dir = os.path.join(args.output_dir, f"step_{step}")
            save_lora_only(policy, tok, save_dir)
            print(f"[save] {save_dir}")


if __name__ == "__main__":
    main()