import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model

# ---------------------------------------------------------
# 1. ノートブックで定義したHybridProbeクラスの再現
# ---------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# ノートブックの設定と合わせる必要があります
SEQ_LEN = 8 

class HybridProbe(nn.Module):
    def __init__(self, input_dim=3584, d_model=256):
        super().__init__()
        # 1. 入力層 (Unexpected key: input_proj と一致)
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 2. Transformer層 (num_layers=2 にすることで layers.1 が作成されます)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        
        # 3. 出力ヘッド (Unexpected key: embed_head と一致)
        # ノートブックでは全トークンをフラットにしてLinearに入力していました
        self.embed_head = nn.Linear(d_model * SEQ_LEN, 128)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.input_proj(x)
        x = self.transformer(x)
        # フラット化してヘッドへ
        x = x.reshape(x.size(0), -1)
        return F.normalize(self.embed_head(x), p=2, dim=1)

# ---------------------------------------------------------
# 2. PPO設定と資産のロード
# ---------------------------------------------------------
SAVE_DIR = "creative_probe_final_v1"
with open(os.path.join(SAVE_DIR, "config.json"), "r") as f:
    cfg = json.load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PPOの設定
config = PPOConfig(
    model_name=cfg["model_name"],
    learning_rate=1e-5,
    batch_size=8,
    mini_batch_size=2,
    gradient_accumulation_steps=4,
    target_kl=0.06, # 内部特徴量の急激な変化を防ぐ
    optimize_cuda_cache=True,
)

# モデルとトークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    cfg["model_name"], peft_config=lora_config, device_map="auto", torch_dtype=torch.bfloat16
)
ref_model = create_reference_model(model)

# 資産のロード
probe = HybridProbe(input_dim=cfg["input_dim"], d_model=cfg["d_model"]).to(DEVICE)
probe.load_state_dict(torch.load(os.path.join(SAVE_DIR, "probe_model.pt")))
probe.eval()
creative_prototype = torch.load(os.path.join(SAVE_DIR, "creative_prototype.pt")).to(DEVICE)

# ---------------------------------------------------------
# 3. 報酬計算関数（Gated Reward Engineの統合）
# ---------------------------------------------------------
def get_external_reward(texts):
    """
    ここにDeBERTaなどの論理性判定モデルを連携させます。
    現在は簡易的な長さ・形式チェックをプレースホルダーとしています。
    """
    # 実際の実装例: return reward_model(texts)
    return [1.0 if len(t) > 30 else -1.0 for t in texts]

def compute_rewards(queries, responses, model, tokenizer):
    rewards = []
    # ゲート用パラメータ（ノートブックの設定値を反映）
    tau = cfg.get("threshold_tau", 0.3)
    k = 1.0 / cfg.get("steepness_k", 15.0)

    for q, r in zip(queries, responses):
        # 内部特徴量の抽出（生成された文全体ではなく、モデルの応答部分を重視）
        inputs = tokenizer(q + r, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.pretrained_model(**inputs, output_hidden_states=True)
            # 指定レイヤー（18層目など）の隠れ状態を取得
            h = outputs.hidden_states[cfg["target_layer"]][0, -1:, :].float()
            # 内発的報酬：プローブによるプロトタイプとの類似度
            intrinsic = torch.nn.functional.cosine_similarity(probe(h), creative_prototype).item()

        # 外部的報酬：論理性
        ext_score = get_external_reward([r])[0]
        ext_norm = np.clip((ext_score + 2.0) / 4.0, 0, 1)

        # シグモイドゲートの適用
        gate = 1 / (1 + np.exp(-(ext_norm - tau) / k))
        
        # 最終報酬
        total_reward = torch.tensor(intrinsic * gate, dtype=torch.float32)
        rewards.append(total_reward)
    
    return rewards

# ---------------------------------------------------------
# 4. 学習ループ
# ---------------------------------------------------------
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

print("🚀 PPO学習開始: 創造的サブスペースへの最適化...")

for epoch in range(100):
    # クエリの生成（実際にはデータセットからサンプリング）
    query_txt = "新しいバイオコンピュータの概念を提案してください。"
    query_tensor = tokenizer.encode(query_txt, return_tensors="pt")[0]
    
    # ロールアウト（生成）
    response_tensors = ppo_trainer.generate(
        [query_tensor], max_new_tokens=64, do_sample=True, top_p=0.9
    )
    response_txt = [tokenizer.decode(r) for r in response_tensors]

    # 報酬計算
    rewards = compute_rewards([query_txt], response_txt, model, tokenizer)

    # PPOステップ
    stats = ppo_trainer.step([query_tensor], response_tensors, rewards)
    
    # ログ出力
    ppo_trainer.log_stats(stats, {"query": query_txt, "response": response_txt[0]}, rewards[0])
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Reward: {rewards[0].item():.4f}")

# 学習済みLoRAアダプターの保存
model.save_pretrained("qwen2.5_creative_ppo_final")