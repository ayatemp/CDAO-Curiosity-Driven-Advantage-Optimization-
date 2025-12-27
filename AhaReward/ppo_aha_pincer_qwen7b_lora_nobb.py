#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
from typing import List, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

# ----------------------------
# Prompt pool (あなたの例)
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

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def rm_score_batch(
    rm_tokenizer,
    rm_model,
    prompts: List[str],
    responses: List[str],
    device: torch.device,
    max_length: int = 1024,
) -> torch.Tensor:
    """
    RM input is typically (prompt, response).
    For OpenAssistant DeBERTa RM: use pair encoding.
    Output: scores [B]
    """
    enc = rm_tokenizer(
        prompts,
        responses,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(device)

    out = rm_model(**enc)
    # reward models usually output a single scalar logit per example
    logits = out.logits.squeeze(-1)  # [B]
    return logits

def main():
    parser = argparse.ArgumentParser()

    # generator model (policy/sft)
    parser.add_argument("--gen-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--rm-model", type=str, default="OpenAssistant/reward-model-deberta-v3-large-v2")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--out-jsonl", type=str, default="./rm_scores.jsonl")
    parser.add_argument("--print-samples", type=int, default=8)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.bf16:
        gen_dtype = torch.bfloat16
    elif args.fp16:
        gen_dtype = torch.float16
    else:
        gen_dtype = torch.float32

    # --- Generator ---
    gen_tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token

    gen_model = AutoModelForCausalLM.from_pretrained(
        args.gen_model,
        trust_remote_code=True,
        torch_dtype=gen_dtype,
        device_map=None,
    ).to(device)
    gen_model.eval()

    # --- Reward Model (DeBERTa sequence classification) ---
    rm_tok = AutoTokenizer.from_pretrained(args.rm_model)
    rm_model = AutoModelForSequenceClassification.from_pretrained(args.rm_model).to(device)
    rm_model.eval()

    # sample prompts
    prompts = [format_prompt(random.choice(STRUCTURED_PROMPTS)) for _ in range(args.num_samples)]

    # generate responses
    responses = []
    for p in prompts:
        inputs = gen_tok(p, return_tensors="pt", truncation=True, max_length=1536).to(device)
        gen_ids = gen_model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=gen_tok.pad_token_id,
            eos_token_id=gen_tok.eos_token_id,
        )
        # decode only the newly generated tokens
        new_tokens = gen_ids[0, inputs["input_ids"].shape[1]:]
        resp = gen_tok.decode(new_tokens, skip_special_tokens=True)
        responses.append(resp)

    # score with RM
    scores = rm_score_batch(rm_tok, rm_model, prompts, responses, device=device)

    # save + print
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    rows = []
    for i, (p, r, s) in enumerate(zip(prompts, responses, scores.tolist())):
        row = {"i": i, "rm_score": float(s), "prompt": p, "response": r}
        rows.append(row)

    with open(args.out_jsonl, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # sort by score for readability
    rows_sorted = sorted(rows, key=lambda x: x["rm_score"], reverse=True)
    n_show = min(args.print_samples, len(rows_sorted))

    for k in range(n_show):
        row = rows_sorted[k]
        print("\n" + "=" * 90)
        print(f"[rank {k+1}/{n_show}] rm_score={row['rm_score']:.4f}")
        print("- PROMPT ------------------------------------------------------------")
        print(row["prompt"])
        print("- RESPONSE ----------------------------------------------------------")
        print(row["response"])
        print("=" * 90)

    print(f"\n[done] wrote: {args.out_jsonl}")

if __name__ == "__main__":
    main()