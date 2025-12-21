import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import os
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, TaskType
from torch.utils.data import Dataset

# ==========================================
# ⚙️ Configuration
# ==========================================
CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "vector_path": "probe_visionary_vector.pt", # あなたが抽出したベクトル
    "target_layer": 16,     # 介入する層
    "hidden_idx": 17,       # hidden_statesのインデックス (Emb=0なので+1)
    "learning_rate": 1.41e-5,
    "batch_size": 16,       # PPO全体のバッチサイズ
    "mini_batch_size": 4,   # GPUに乗るサイズ (勾配蓄積で調整)
    "gradient_accumulation_steps": 1,
    "ppo_epochs": 4,
    "target_kl": 0.1,       # KL制御の目標値
    "init_kl_coef": 0.2,
    "steps": 100,           # デモ用ステップ数 (適宜増やしてください)
    "reward_scale": 15.0,   # 報酬の倍率
    "wandb_project": "Visionary-PPO-Steering-LoRA",
    
    # --- LoRA Settings ---
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 🧠 Reward Engine (Fixed: Float32 Cast)
# ==========================================
class LatentSteeringReward:
    def __init__(self, vector_path, target_hidden_idx, scale, device):
        print(f"Loading Steering Vector from {vector_path}...")
        # ベクトルは float32 でロード
        self.vector = torch.load(vector_path).to(device).float()
        self.vector = F.normalize(self.vector, dim=0)
        self.target_hidden_idx = target_hidden_idx
        self.scale = scale
        self.device = device

    def compute_reward(self, model, input_ids, attention_mask, response_start_idx):
        # ValueHead付きモデルからBaseモデル(LoRA適用済み)を取り出す
        base_model = model.pretrained_model
        
        # 推論モード (勾配不要)
        with torch.no_grad():
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # 指定層のHidden State取得
        target_h = outputs.hidden_states[self.target_hidden_idx]
        
        # ★修正: 強制的にFloat32にキャストして計算 (Halfエラー回避)
        target_h = target_h.float()
        
        # 正規化
        target_h_norm = F.normalize(target_h, dim=-1)
        
        # コサイン類似度
        similarity = torch.matmul(target_h_norm, self.vector)
        
        rewards = []
        batch_size = input_ids.shape[0]
        
        for i in range(batch_size):
            # レスポンス部分のみ抽出
            resp_sim = similarity[i, response_start_idx[i]:] 
            
            if len(resp_sim) > 0:
                score = resp_sim.mean().item()
            else:
                score = 0.0
            
            rewards.append(score * self.scale)
            
        return rewards

# ==========================================
# 📚 Creative Dataset
# ==========================================
class VisionaryDataset(Dataset):
    def __init__(self, tokenizer, num_samples=500):
        self.tokenizer = tokenizer
        # ベクトルが反応しやすい創造的なお題
        base_prompts = [
            "Propose a radical new technology to manipulate gravity.",
            "Theorize a biological mechanism for immortality.",
            "Explain how consciousness arises from quantum effects.",
            "Invent a device to record dreams.",
            "What exists outside of time?",
            "Is the universe a simulation? Argue for yes.",
            "Define 'justice' for an alien civilization.",
            "Describe a color that implies sadness.",
            "Write a myth about the death of the sun.",
            "Describe the sound of silence in a crowded room.",
            "A poem about a clock that counts backwards.",
            "The diary entry of the last human on Earth.",
            "Describe a city built entirely of glass.",
            "Explain the concept of infinity to a child using metaphors.",
            "Design a new organ for humans to survive on Mars."
        ]
        
        self.inputs = []
        print("Building dataset...")
        # データ増幅
        for _ in range(num_samples // len(base_prompts) + 1):
            for p in base_prompts:
                # シンプルなユーザープロンプト
                msgs = [{"role": "user", "content": p}]
                txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                self.inputs.append(txt)
                
        self.inputs = self.inputs[:num_samples]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

def collator(data):
    return dict(query=data)

# ==========================================
# 🏃‍♂️ Training Loop with LoRA
# ==========================================
def train():
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    
    # 1. Tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    tokenizer.pad_token = tokenizer.eos_token 

    # 2. LoRA Config
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=CONFIG["lora_target_modules"]
    )

    # 3. Load Model with LoRA
    print("Loading Model with Adapter...")
    # TRLのAutoModelForCausalLMWithValueHeadは、peft_configを渡すと自動的にLoRAモデルを作ってくれます
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        CONFIG["model_name"],
        peft_config=lora_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 4. PPO Config
    # ref_model=None にすると、TRLは自動的に「アダプターを無効化した状態」をRefとして扱います(メモリ節約)
    ppo_config = PPOConfig(
        model_name=CONFIG["model_name"],
        learning_rate=CONFIG["learning_rate"],
        batch_size=CONFIG["batch_size"],
        mini_batch_size=CONFIG["mini_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        ppo_epochs=CONFIG["ppo_epochs"],
        target_kl=CONFIG["target_kl"],
        init_kl_coef=CONFIG["init_kl_coef"],
        remove_unused_columns=False
    )
    
    # 5. Prepare Dataset
    dataset = VisionaryDataset(tokenizer, num_samples=CONFIG["steps"] * CONFIG["batch_size"])
    
    # 6. Initialize Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None, # LoRAの場合はNone推奨
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator
    )
    
    # 7. Reward Engine
    reward_engine = LatentSteeringReward(
        CONFIG["vector_path"], 
        CONFIG["hidden_idx"], 
        CONFIG["reward_scale"], 
        DEVICE
    )
    
    print("Starting LoRA-PPO Training...")
    
    # --- Loop ---
    for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if step >= CONFIG["steps"]: break
        
        queries = batch["query"]
        query_tensors = [tokenizer(q, return_tensors="pt").input_ids.squeeze().to(DEVICE) for q in queries]
        
        # A. Rollout (Generate)
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            max_new_tokens=64,
            temperature=0.9,
            top_p=0.95,
            do_sample=True
        )
        
        batch["response"] = tokenizer.batch_decode(response_tensors)
        
        # B. Reward Calculation
        rewards = []
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
        
        # List of tensors for TRL
        rewards = [torch.tensor(r).to(DEVICE) for r in raw_rewards]
        
        # C. PPO Step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # D. Logging
        mean_reward = np.mean(raw_rewards)
        
        log_data = {
            "step": step,
            "visionary_reward": mean_reward,
            "ppo/learning_rate": stats["ppo/learning_rate"],
            "env/kl_mean": stats["objective/kl"],
        }
        
        if step % 5 == 0:
            table = wandb.Table(columns=["Query", "Response", "Reward"])
            # 先頭2つだけログ
            for q, r, rew in zip(queries[:2], batch["response"][:2], raw_rewards[:2]):
                table.add_data(q, r, rew)
            log_data["generated_samples"] = table
            
        wandb.log(log_data)
        
    print("Training Complete!")
    
    # Save LoRA Adapter
    print("Saving LoRA adapters...")
    ppo_trainer.model.save_pretrained("Qwen-Visionary-LoRA")
    tokenizer.save_pretrained("Qwen-Visionary-LoRA")

if __name__ == "__main__":
    train()