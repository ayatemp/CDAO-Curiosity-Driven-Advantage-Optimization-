import os
import json
import torch
import numpy as np
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from datasets import Dataset

# ==========================================
# 1. 定数・ディレクトリ設定
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PROBE_DIR = "creative_probe_final_v1"
SAVE_DIR = "creative_ppo_model_v1"
EXT_RM_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 保存先がない場合に作成
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. プローブ & 報酬エンジン定義
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
        # x: [batch, seq_len, input_dim]
        x = self.transformer(self.input_proj(x)).reshape(x.size(0), -1)
        return torch.nn.functional.normalize(self.embed_head(x), p=2, dim=1)

def load_reward_components():
    # 設定の読み込み
    with open(os.path.join(PROBE_DIR, "config.json"), "r") as f:
        cfg = json.load(f)
    
    # プローブの復元
    probe = StructuralProbe(input_dim=cfg["input_dim"], seq_len=cfg["seq_len"]).to(DEVICE)
    probe.load_state_dict(torch.load(os.path.join(PROBE_DIR, "probe_model.pt")))
    probe.eval()
    
    # プロトタイプの復元
    prototype = torch.load(os.path.join(PROBE_DIR, "creative_prototype.pt")).to(DEVICE)
    
    return probe, prototype, cfg

# ==========================================
# 3. LoRA設定 & モデル初期化
# ==========================================
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

print(f"Loading Policy Model with LoRA: {MODEL_NAME}")
# PPO用のValueHead付きモデル
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MODEL_NAME,
    peft_config=lora_config,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# External RM (妥当性ゲート用)
ext_tokenizer = AutoTokenizer.from_pretrained(EXT_RM_NAME)
ext_model = AutoModelForSequenceClassification.from_pretrained(EXT_RM_NAME).to(DEVICE)

# ==========================================
# 4. PPO学習の設定
# ==========================================
config = PPOConfig(
    model_name=MODEL_NAME,
    learning_rate=1.41e-5,
    batch_size=8,
    mini_batch_size=1,
    gradient_accumulation_steps=8,
    optimize_cuda_cache=True,
    early_stopping=True,
    target_kl=0.1, # 崩壊を防ぐ
)

# 学習用プロンプト（研究の成果が出るよう、創造的ドメインに特化）
# 学習用プロンプト
ppo_prompts = [
    "Propose a new form of architecture that interacts with the wind.",
    "Describe a futuristic sport played in low gravity using magnets.",
    "Explain a method for humans to communicate with plants using light.",
    "Imagine a digital economy based on the sharing of dreams.",
    "Design a transportation system for a city built inside a giant tree."
] * 20 

# リストを作成
raw_data = [{"query": p} for p in ppo_prompts]

# 1. Hugging Face Dataset オブジェクトに変換 (ここが修正ポイント)
dataset = Dataset.from_list(raw_data)

# 2. トークナイズ処理の定義
def tokenize_fn(example):
    example["input_ids"] = tokenizer.encode(example["query"], add_special_tokens=False)
    return example

# データセットに適用
dataset = dataset.map(tokenize_fn, batched=False)
dataset.set_format(type="torch")

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

ppo_trainer = PPOTrainer(config, model, None, tokenizer, dataset=dataset, data_collator=collator)

# ==========================================
# 5. 報酬計算関数 (Gated Internal Reward)
# ==========================================
probe, prototype, probe_cfg = load_reward_components()

def calculate_hybrid_reward(query, response):
    # 1. 内発的報酬 (Probe)
    full_text = query + response
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        # Policyモデル（ValueHead付き）からHidden Statesを抽出
        outputs = model.pretrained_model(
            inputs.input_ids, 
            output_hidden_states=True, 
            return_dict=True
        )
        # 指定レイヤーの末尾SEQ_LENトークンを取得
        h = outputs.hidden_states[probe_cfg["target_layer"]][0, -probe_cfg["seq_len"]:, :].float().unsqueeze(0)
        intrinsic_sim = torch.nn.functional.cosine_similarity(probe(h), prototype).item()
    
    # 2. 外部的報酬 (Normalcy Gate)
    ext_inputs = ext_tokenizer(response, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ext_score = ext_model(**ext_inputs).logits[0].item()
    
    # ゲート処理 (前回検証した最適パラメータ)
    ext_norm = (ext_score + 2.0) / 4.0 # 簡易正規化
    gate = 1 / (1 + np.exp(-(ext_norm - 0.3) / (1.0/15.0)))
    
    return torch.tensor(intrinsic_sim * gate)

# ==========================================
# 6. 学習ループ
# ==========================================
print("Starting PPO Training...")
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 48,
}

for epoch in tqdm(range(10), desc="Epochs"):
    for batch in ppo_trainer.dataloader:
        query_tensors = [torch.tensor(ids).to(DEVICE) for ids in batch["input_ids"]]
        
        # モデルによる応答生成
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
        # 報酬の計算
        rewards = [calculate_hybrid_reward(q, r) for q, r in zip(batch["query"], batch["response"])]
        
        # PPOステップの実行
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

# ==========================================
# 7. モデル保存
# ==========================================
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ PPO Training Complete. Model saved to: {SAVE_DIR}")