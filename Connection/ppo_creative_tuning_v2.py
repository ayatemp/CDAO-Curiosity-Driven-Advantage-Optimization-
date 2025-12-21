import os
import json
import torch
import numpy as np
import random
from tqdm import tqdm
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset

# ==========================================
# 1. 定数・環境設定
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PROBE_DIR = "creative_probe_final_v1"
SAVE_DIR = "creative_fusion_ppo_model"
EXT_RM_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. プローブモデルの定義 (Layer 18専用)
# ==========================================
class StructuralProbe(torch.nn.Module):
    def __init__(self, input_dim=3584, d_model=256, seq_len=8):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, d_model)
        enc = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(enc, num_layers=2)
        self.embed_head = torch.nn.Linear(d_model * seq_len, 128)
        self.seq_len = seq_len

    def forward(self, x):
        x = self.transformer(self.input_proj(x)).reshape(x.size(0), -1)
        return torch.nn.functional.normalize(self.embed_head(x), p=2, dim=1)

def load_essentials():
    with open(os.path.join(PROBE_DIR, "config.json"), "r") as f:
        cfg = json.load(f)
    probe = StructuralProbe(input_dim=cfg["input_dim"], seq_len=cfg["seq_len"]).to(DEVICE)
    probe.load_state_dict(torch.load(os.path.join(PROBE_DIR, "probe_model.pt")))
    probe.eval()
    prototype = torch.load(os.path.join(PROBE_DIR, "creative_prototype.pt")).to(DEVICE)
    return probe, prototype, cfg

# ==========================================
# 3. 融合研究用プロンプトの生成
# ==========================================
domains = [
    "Quantum Physics", "Jazz Improvisation", "Microbiology", "Ancient Architecture",
    "Blockchain Technology", "Neuroscience", "Fungal Ecology", "Cybersecurity",
    "Music Theory", "Botany", "Aerospace Engineering", "Gastronomy", "Textile Design"
]

def generate_fusion_prompts(num=100):
    prompts = []
    for _ in range(num):
        d1, d2 = random.sample(domains, 2)
        p = f"Create a groundbreaking research proposal that synthesizes elements from {d1} and {d2}. Focus on an unconventional 'what-if' scenario."
        prompts.append(p)
    return prompts

# ==========================================
# 4. モデル・PPO初期化
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MODEL_NAME, peft_config=lora_config, device_map="auto", torch_dtype=torch.float16
)

ext_tokenizer = AutoTokenizer.from_pretrained(EXT_RM_NAME)
ext_model = AutoModelForSequenceClassification.from_pretrained(EXT_RM_NAME).to(DEVICE)

# データセット準備
raw_prompts = generate_fusion_prompts(200)
dataset = Dataset.from_list([{"query": p} for p in raw_prompts])
dataset = dataset.map(lambda x: {"input_ids": tokenizer.encode(x["query"], add_special_tokens=True)}, batched=False)
dataset.set_format(type="torch")

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

config = PPOConfig(
    model_name=MODEL_NAME, learning_rate=5e-6, batch_size=8, mini_batch_size=1,
    gradient_accumulation_steps=8, target_kl=0.1, init_kl_coef=0.2, adap_kl_ctrl=True, log_with="wandb"
)
ppo_trainer = PPOTrainer(config, model, None, tokenizer, dataset=dataset, data_collator=collator)

# ==========================================
# 5. 報酬エンジン (Layer 18末尾8トークン)
# ==========================================
probe, prototype, probe_cfg = load_essentials()

def get_hybrid_reward(query, response):
    full_text = query + response
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.pretrained_model(inputs.input_ids, output_hidden_states=True, return_dict=True)
        # 末尾8トークンの特徴量抽出
        h = outputs.hidden_states[probe_cfg["target_layer"]][0, -probe_cfg["seq_len"]:, :].float().unsqueeze(0)
        intrinsic_sim = torch.nn.functional.cosine_similarity(probe(h), prototype).item()
        
        # 妥当性ゲート
        ext_inputs = ext_tokenizer(response, return_tensors="pt").to(DEVICE)
        ext_score = ext_model(**ext_inputs).logits[0].item()
        
    ext_norm = (ext_score + 2.0) / 4.0
    gate = 1 / (1 + np.exp(-(ext_norm - 0.3) / (1.0/15.0)))
    return torch.tensor(intrinsic_sim * gate)

# ==========================================
# 6. 学習ループ
# ==========================================
generation_kwargs = {
    "min_length": -1, "top_k": 50, "top_p": 0.9, "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id, "max_new_tokens": 160, # 長めのアイデア出しを許可
    "temperature": 0.85, "repetition_penalty": 1.15
}

print("\n🚀 Starting Cross-Domain Fusion PPO...")
for epoch in tqdm(range(5), desc="Epochs"):
    for batch in ppo_trainer.dataloader:
        query_tensors = [q.clone().detach() for q in batch["input_ids"]]
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
        rewards = [get_hybrid_reward(q, r) for q, r in zip(batch["query"], batch["response"])]
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        print(f"\n[Fusion Reward] {np.mean([r.item() for r in rewards]):.4f} | [KL] {stats['objective/kl']:.4f}")
        print(f"[Proposal] {batch['response'][0][:150]}...")

# ==========================================
# 7. 保存
# ==========================================
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ Training Complete. Model saved to: {SAVE_DIR}")