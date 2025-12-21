import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig, create_reference_model

# ==========================================
# 1. 保存・ディレクトリ管理用ユーティリティ
# ==========================================
def safe_save_model(model, tokenizer, path):
    """ディレクトリが存在しない場合は作成し、モデルとトークナイザーを保存"""
    directory = os.path.dirname(path) if os.path.dirname(path) != "" else path
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"📁 ディレクトリを作成しました: {directory}")
    
    # LoRAアダプターのみを保存（メモリ効率のため）
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"✅ モデルを保存しました: {path}")

# ==========================================
# 2. PPO設定とモデルロード
# ==========================================
model_name = "Qwen/Qwen2.5-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
    batch_size=4,
    mini_batch_size=1,
    gradient_accumulation_steps=4,
    optimize_cuda_cache=True,
    early_stopping=True,
    target_kl=0.1,
)

# トークナイザーのロード (生成用に左パディングを設定)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# モデルのロード (LoRA適用)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    output_hidden_states=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
ref_model = create_reference_model(model) # 学習の基準点となる参照モデル

# ==========================================
# 3. 報酬計算用プローブのセットアップ
# ==========================================
# 先ほど学習した PooledCreativityProbe クラスは定義済みと仮定
# probe = load_creativity_probe("creativity_pooled_probe_v1.pth")
probe.eval()

def get_reward_from_probe(query_tensors, response_tensors):
    """
    生成された応答から内部特徴量を抽出し、プローブで報酬スコアリングを行う。
    内部特徴量の差分を利用して創造性を評価する。
    """
    rewards = []
    # 各サンプルについて推論を回す
    for q, r in zip(query_tensors, response_tensors):
        full_input_ids = torch.cat([q, r], dim=-1).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model.get_base_model()(**full_input_ids, output_hidden_states=True)
            
        # 生成されたトークン位置の特定
        gen_len = r.shape[0]
        pooled_layers = {}
        for l_idx in range(20, 27): # 研究で特定したスイートスポット
            # 生成されたトークン全体の時間軸方向での平均 (Mean Pooling)
            step_vectors = [outputs.hidden_states[l_idx+1][:, -(gen_len - s), :] for s in range(gen_len)]
            pooled_layers[l_idx] = torch.stack(step_vectors).mean(dim=0).to(torch.float32)
            
        # プローブによるスコアリング (Logitをそのまま報酬として利用)
        reward_logit = probe(pooled_layers)
        rewards.append(reward_logit.squeeze())
        
    return rewards

# ==========================================
# 4. トレーニングループ
# ==========================================
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

# 学習用クエリ（地獄の検証等で使用したような多様なトピック）
dataset = ["次世代のエネルギーについて、極めて独創的な案を出して。"] * 100 

print("🚀 PPOトレーニングを開始します...")
for epoch, batch in enumerate(tqdm(dataset)):
    query_tensors = [tokenizer.encode(q, return_tensors="pt").squeeze().to(device) for q in [batch]]
    
    # モデルによる生成
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 32,
    }
    response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
    
    # 報酬の計算
    rewards = get_reward_from_probe(query_tensors, response_tensors)
    
    # PPOステップの実行
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    
    # 10ステップごとにチェックポイント保存
    if (epoch + 1) % 10 == 0:
        save_path = f"./checkpoints/ppo_creativity_model_step_{epoch+1}"
        safe_save_model(model, tokenizer, save_path)

# 最終保存
safe_save_model(model, tokenizer, "./final_creative_model_lora")