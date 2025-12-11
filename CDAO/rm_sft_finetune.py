#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rm_sft_finetune.py

【RM安定化ステップ】Qwen出力に特化したExternal Reward Model (DeBERTa) のSFT
 - External RM の評価を安定させることを目的とする。


 python rm_sft_finetune.py \
    --epochs 3 \
    --batch-size 4 \
    --output-dir ./rm_finetuned
"""

import argparse
import random
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

# ============================================================
# 1. ユーティリティ
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

def create_mock_data(tokenizer):
    """
    Qwenの研究プロンプトの形式に合わせた模擬データを作成
    実際には、人間の評価済みデータまたはGPT-4評価済みデータを使用
    """
    data = []
    
    # 良い例 (label=1) - 複雑で創造的
    good_prompts = [
        "Draft a research proposal about RLHF.",
        "Propose an experiment for RAG focusing on efficiency."
    ]
    good_responses = [
        "Title: Adaptive Preference Modeling...\nAbstract: We propose a dynamic model...",
        "Title: Query-Adaptive Indexing for RAG...\nAbstract: Current RAG systems suffer..."
    ]
    
    # 悪い例 (label=0) - 単純、繰り返し、または文法ミス
    bad_prompts = [
        "Draft a research proposal about RLHF.",
        "Propose an experiment for RAG focusing on efficiency."
    ]
    bad_responses = [
        "RLHF is good. RLHF is very good. We will study RLHF. This is the end.",
        "Title: Simple RAG Experiment\nAbstract: The rag system is retrieval. Retrieval is good. Retrieval is important."
    ]
    
    # データセット構築
    for p, r in zip(good_prompts, good_responses):
        data.append({"text": f"User: {p}\nAssistant: {r}", "label": 1.0})
    for p, r in zip(bad_prompts, bad_responses):
        data.append({"text": f"User: {p}\nAssistant: {r}", "label": 0.0})
        
    return Dataset.from_list(data)

# ============================================================
# 2. メイン学習関数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm-model-name", type=str, default="OpenAssistant/reward-model-deberta-v3-large-v2")
    parser.add_argument("--output-dir", type=str, default="./rm_finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    set_seed(42)

    # 1. Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.rm_model_name)
    # 報酬モデルは通常 float32 で学習させる
    model = AutoModelForSequenceClassification.from_pretrained(args.rm_model_name, num_labels=1) 
    
    # 2. Data Preparation
    raw_dataset = create_mock_data(tokenizer)
    
    def tokenize_function(examples):
        # トークナイズ処理: RMが評価する入力形式に合わせる
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")
    
    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=1e-5,
        logging_dir=f'{args.output_dir}/logs',
        save_strategy="epoch",
        overwrite_output_dir=True,
        fp16=True, # GPU使用を前提
    )

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # 5. Training
    print("\n=== Starting RM Fine-tuning ===")
    trainer.train()
    
    # 6. Save
    print(f"\n[INFO] Saving fine-tuned RM to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()