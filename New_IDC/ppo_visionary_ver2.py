import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import datasets
import random

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, TaskType
from torch.utils.data import Dataset

# ==========================================
# ⚙️ Configuration
# ==========================================
CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "vector_path": "probe_visionary_vector.pt",
    "target_layer": 16,
    "hidden_idx": 17,
    "learning_rate": 1.41e-5,
    "batch_size": 16,
    "mini_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "ppo_epochs": 4,
    "target_kl": 0.1,
    "init_kl_coef": 0.2,
    "steps": 2000,          # 成功実績のある200ステップ
    "reward_scale": 20.0,
    "wandb_project": "Visionary-PPO-Final-Fix", # プロジェクト名を変更
    "run_name": "run-robust-fix",               # 実行名
    "output_dir": "Qwen-Visionary-Robust-LoRA", # 保存先
    
    # Robustness Settings
    "min_new_tokens": 64,
    "max_new_tokens": 128,
    "reward_baseline": 0.0,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 🧠 Reward Engine
# ==========================================
class LatentSteeringReward:
    def __init__(self, vector_path, target_hidden_idx, scale, device):
        print(f"Loading Steering Vector from {vector_path}...")
        self.vector = torch.load(vector_path).to(device).float()
        self.vector = F.normalize(self.vector, dim=0)
        self.target_hidden_idx = target_hidden_idx
        self.scale = scale
        self.device = device

    def compute_reward(self, model, input_ids, attention_mask, response_start_idx):
        base_model = model.pretrained_model
        with torch.no_grad():
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        target_h = outputs.hidden_states[self.target_hidden_idx].float()
        target_h_norm = F.normalize(target_h, dim=-1)
        similarity = torch.matmul(target_h_norm, self.vector)
        
        rewards = []
        batch_size = input_ids.shape[0]
        
        for i in range(batch_size):
            start = response_start_idx[i]
            resp_sim = similarity[i, start:] 
            if len(resp_sim) > 0:
                score = resp_sim.mean().item()
            else:
                score = 0.0
            rewards.append(score * self.scale)
            
        return rewards

# ==========================================
# 📚 Dataset
# ==========================================
def create_diverse_dataset(tokenizer, num_samples):
    topics = [
        "Quantum Consciousness", "Time Travel Paradoxes", "Artificial General Intelligence", 
        "Interstellar Propulsion", "The Nature of Reality", "Dream Recording", 
        "Biological Immortality", "Non-Euclidean Geometry", "The Origin of Language",
        "Cybernetic Enhancements", "Dark Matter Civilizations", "The End of the Universe"
    ]
    tasks = [
        "Propose a radical theory about", "Write a myth concerning", 
        "Explain to a 5-year-old", "Design a machine for", 
        "Write a philosophical dialogue about", "Imagine a color related to"
    ]
    inputs = []
    print("Building diverse dataset...")
    for _ in range(num_samples):
        t = random.choice(topics)
        task = random.choice(tasks)
        prompt = f"{task} {t}."
        msgs = [{"role": "user", "content": prompt}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs.append(txt)
    return datasets.Dataset.from_dict({"query": inputs})

def collator(data):
    return dict(query=[d["query"] for d in data])

# ==========================================
# 🏃‍♂️ Training Loop
# ==========================================
def train():
    # 手動wandb.initは削除し、PPOConfigに任せる
    
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    tokenizer.pad_token = tokenizer.eos_token 

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    print("Loading Model with LoRA...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        CONFIG["model_name"],
        peft_config=lora_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    ppo_config = PPOConfig(
        model_name=CONFIG["model_name"],
        learning_rate=CONFIG["learning_rate"],
        batch_size=CONFIG["batch_size"],
        mini_batch_size=CONFIG["mini_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        ppo_epochs=CONFIG["ppo_epochs"],
        target_kl=CONFIG["target_kl"],
        init_kl_coef=CONFIG["init_kl_coef"],
        remove_unused_columns=False,
        log_with="wandb", # ここで自動設定
        tracker_project_name=CONFIG["wandb_project"],
        tracker_kwargs={"wandb": {"name": CONFIG["run_name"]}}
    )
    
    dataset = create_diverse_dataset(tokenizer, CONFIG["steps"] * CONFIG["batch_size"])
    
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None, 
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator
    )
    
    reward_engine = LatentSteeringReward(
        CONFIG["vector_path"], 
        CONFIG["hidden_idx"], 
        CONFIG["reward_scale"], 
        DEVICE
    )
    
    print("Starting Robust PPO Training...")
    
    for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if step >= CONFIG["steps"]: break
        
        queries = batch["query"]
        query_tensors = [tokenizer(q, return_tensors="pt").input_ids.squeeze().to(DEVICE) for q in queries]
        
        # 1. Generate
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": CONFIG["max_new_tokens"],
            "temperature": 0.9,
        }
        
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # 2. Reward
        full_input_ids = []
        masks = []
        response_start_indices = []
        
        for q_t, r_t in zip(query_tensors, response_tensors):
            full = torch.cat((q_t, r_t))
            full_input_ids.append(full)
            masks.append(torch.ones_like(full))
            response_start_indices.append(len(q_t))
            
        full_input_ids_tensor = torch.nn.utils.rnn.pad_sequence(
            full_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(DEVICE)
        
        attention_mask_tensor = torch.nn.utils.rnn.pad_sequence(
            masks, batch_first=True, padding_value=0
        ).to(DEVICE)
        
        raw_rewards = reward_engine.compute_reward(
            ppo_trainer.model, 
            full_input_ids_tensor, 
            attention_mask_tensor,
            response_start_indices
        )
        
        # 3. Normalization & Step
        rewards_tensor = torch.tensor(raw_rewards, device=DEVICE)
        rewards_tensor = rewards_tensor - CONFIG["reward_baseline"]
        
        mean = rewards_tensor.mean()
        std = rewards_tensor.std() + 1e-8
        norm_rewards_tensor = (rewards_tensor - mean) / std
        norm_rewards_tensor = torch.clamp(norm_rewards_tensor, -4.0, 4.0)
        
        final_rewards = [r for r in norm_rewards_tensor]
        
        stats = ppo_trainer.step(query_tensors, response_tensors, final_rewards)
        
        # 4. Logging (Fix: use accelerator.log)
        # ログデータを作成
        log_metrics = {
            "reward/raw_mean": mean.item(),
            "reward/norm_mean": norm_rewards_tensor.mean().item(),
            "env/kl_mean": stats["objective/kl"],
        }
        
        # TRLの標準ログに追加する形ではなく、acceleratorで直接投げる
        ppo_trainer.accelerator.log(log_metrics, step=step)
        
        # サンプルテキストの表示 (コンソール出力のみにするか、wandbテーブルにするか)
        # TRLと競合しないよう、テーブルは頻度を落としてコンソールに出すだけでも良いが、
        # ここではテーブル保存を試みる（エラーが出たらtry-exceptで逃げる）
        if step % 20 == 0:
            try:
                # accelerator経由だとTableは投げにくいので、メインプロセスのみwandb.logを呼ぶ裏技
                if ppo_trainer.accelerator.is_main_process:
                    import wandb as wb
                    if wb.run is not None:
                        table = wb.Table(columns=["Query", "Response", "Raw Reward"])
                        for q, r, raw in zip(queries[:2], batch["response"][:2], raw_rewards[:2]):
                            table.add_data(q, r, raw)
                        wb.log({"generated_samples": table}, step=step)
            except Exception:
                pass # ログで止まるのは避ける

    print("Training Complete. Saving Adapter...")
    
    # ★修正ポイント: フォルダを作成してから保存
    save_path = CONFIG["output_dir"]
    os.makedirs(save_path, exist_ok=True)
    
    ppo_trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ Successfully saved to {save_path}")

if __name__ == "__main__":
    train()