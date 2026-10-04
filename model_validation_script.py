# ============================================================
# validate_model.py
# 使用驗證集測試訓練好的模型
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from typing import List
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# 配置
# ----------------------
VALIDATION_DATA_PATH = "C:/Users/USER/Desktop/課程/專題/驗證資料/validation_data.xlsx"
OUTPUT_DIR = "C:/Users/USER/Desktop/課程/專題/驗證資料/validation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8

# ----------------------
# 載入模型架構（與訓練時相同）
# ----------------------
class PostsEncoder(nn.Module):
    def __init__(self, embed_dim, nhead=4, nhid=256, nlayers=2, dropout=0.3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=nhid,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        out = self.transformer(x)
        out = self.layer_norm(out)
        pooled = out.mean(dim=1)
        return pooled

class UserLogicClassifier(nn.Module):
    def __init__(self, embed_dim, num_num_feats, num_logic_feats, hidden=128, dropout=0.4):
        super().__init__()
        self.posts_encoder = PostsEncoder(embed_dim, dropout=dropout)
        self.bio_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        total_dim = embed_dim * 2 + num_num_feats + num_logic_feats
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, posts, bio, num_feats, logic_feats):
        posts_vec = self.posts_encoder(posts)
        bio_vec = self.bio_proj(bio)
        x = torch.cat([posts_vec, bio_vec, num_feats, logic_feats], dim=1)
        logits = self.mlp(x).squeeze(-1)
        return logits

# ----------------------
# 特徵提取函數
# ----------------------
def compute_user_numeric_features(post_texts: List[str], bio: str, post_dates: List[pd.Timestamp]):
    post_count = len(post_texts)
    avg_post_len = np.mean([len(str(t)) for t in post_texts]) if post_count > 0 else 0.0
    bio_len = len(str(bio))
    night_count = sum([1 for d in post_dates if 0 <= pd.to_datetime(d).hour <= 5])
    night_ratio = night_count / post_count if post_count > 0 else 0.0
    max_post_len = max([len(str(t)) for t in post_texts]) if post_count > 0 else 0.0
    min_post_len = min([len(str(t)) for t in post_texts]) if post_count > 0 else 0.0
    return [post_count, avg_post_len, bio_len, night_ratio, max_post_len, min_post_len]

def extract_logic_features(post_list: List[str], bio: str) -> List[float]:
    neg_words = ["痛苦", "沒意義", "撐不住", "累", "空虛", "需要陪伴", "消失", 
                 "不想活", "孤獨", "絕望"]
    pos_words = ["開心", "期待", "感謝", "喜歡", "平靜", "快樂", "滿足"]
    
    neg_count = sum([sum([w in p for w in neg_words]) for p in post_list])
    pos_count = sum([sum([w in p for w in pos_words]) for p in post_list])
    logic_score = neg_count - pos_count
    bio_flag = 1 if any(w in bio for w in ["抑鬱", "憂鬱", "低潮", "焦慮"]) else 0
    total_count = neg_count + pos_count
    neg_ratio = neg_count / total_count if total_count > 0 else 0
    return [neg_count, pos_count, logic_score, bio_flag, neg_ratio]

# ----------------------
# 視覺化函數
# ----------------------
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['正常', '憂鬱傾向'],
                yticklabels=['正常', '憂鬱傾向'])
    plt.title('混淆矩陣 (Confusion Matrix)', fontsize=14, fontweight='bold')
    plt.ylabel('實際標籤', fontsize=12)
    plt.xlabel('預測標籤', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 混淆矩陣已儲存至: {save_path}")

def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC曲線已儲存至: {save_path}")

def plot_prediction_distribution(y_probs, y_true, save_path):
    plt.figure(figsize=(10, 6))
    
    # 分別繪製正常和憂鬱用戶的預測機率分佈
    probs_normal = y_probs[y_true == 0]
    probs_depression = y_probs[y_true == 1]
    
    plt.hist(probs_normal, bins=30, alpha=0.6, label='正常用戶', color='green')
    plt.hist(probs_depression, bins=30, alpha=0.6, label='憂鬱傾向用戶', color='red')
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='決策邊界 (0.5)')
    
    plt.xlabel('預測機率 (憂鬱傾向)', fontsize=12)
    plt.ylabel('用戶數量', fontsize=12)
    plt.title('預測機率分佈', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 預測機率分佈圖已儲存至: {save_path}")

# ----------------------
# 主驗證函數
# ----------------------
def validate_model():
    print("="*60)
    print("模型驗證程式")
    print("="*60)
    
    # 1. 載入 metadata
    print("\n[1/7] 載入模型配置...")
    with open(os.path.join('C:/Users/USER/Desktop/課程/專題/模型訓練/saved_user_model/', "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    EMBED_MODEL_NAME = meta["embed_model"]
    EMBED_DIM = meta["embed_dim"]
    MAX_POSTS = meta["max_posts"]
    
    print(f"  - 嵌入模型: {EMBED_MODEL_NAME}")
    print(f"  - 嵌入維度: {EMBED_DIM}")
    print(f"  - 最大貼文數: {MAX_POSTS}")
    
    # 2. 載入驗證資料
    print("\n[2/7] 載入驗證資料...")
    df = pd.read_excel(VALIDATION_DATA_PATH)
    df["bio"] = df["bio"].fillna("")
    df["post_content"] = df["post_content"].fillna("")
    df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")
    df["depression_or_not"] = df["depression_or_not"].astype(int)
    
    grouped = df.sort_values(["username", "post_date"]).groupby("username").agg({
        "display_name": "first",
        "bio": "first",
        "post_content": lambda x: list(x),
        "post_date": lambda x: list(x),
        "depression_or_not": "max"
    }).reset_index()
    
    print(f"  - 總用戶數: {len(grouped)}")
    print(f"  - 憂鬱用戶: {grouped['depression_or_not'].sum()}")
    print(f"  - 正常用戶: {(1 - grouped['depression_or_not']).sum()}")
    
    # 3. 提取特徵
    print("\n[3/7] 提取數值特徵...")
    num_feats = np.array([
        compute_user_numeric_features(r["post_content"], r["bio"], r["post_date"])
        for _, r in grouped.iterrows()
    ])
    num_scaler = joblib.load(os.path.join('C:/Users/USER/Desktop/課程/專題/模型訓練/saved_user_model/', "numerical_scaler.pkl"))
    num_feats_scaled = num_scaler.transform(num_feats)
    
    print("\n[4/7] 提取邏輯特徵...")
    logic_feats = np.array([
        extract_logic_features(r["post_content"], r["bio"])
        for _, r in grouped.iterrows()
    ], dtype=np.float32)
    logic_scaler = joblib.load(os.path.join('C:/Users/USER/Desktop/課程/專題/模型訓練/saved_user_model/', "logic_scaler.pkl"))
    logic_feats_scaled = logic_scaler.transform(logic_feats)
    
    # 4. 生成文本嵌入
    print("\n[5/7] 生成文本嵌入...")
    st_model = SentenceTransformer(EMBED_MODEL_NAME)
    
    bio_embs = np.array(st_model.encode(grouped["bio"].tolist(), show_progress_bar=True), 
                        dtype=np.float32)
    
    all_post_embs = []
    for post_list in grouped["post_content"].tolist():
        texts = [str(t) for t in post_list][:MAX_POSTS]
        if len(texts) == 0:
            all_post_embs.append(np.zeros((MAX_POSTS, EMBED_DIM), dtype=np.float32))
            continue
        emb = st_model.encode(texts, show_progress_bar=False)
        if emb.shape[0] < MAX_POSTS:
            pad = np.zeros((MAX_POSTS - emb.shape[0], EMBED_DIM), dtype=np.float32)
            emb = np.vstack([emb, pad])
        else:
            emb = emb[:MAX_POSTS]
        all_post_embs.append(emb.astype(np.float32))
    all_post_embs = np.stack(all_post_embs)
    
    labels = grouped["depression_or_not"].values
    
    # 5. 載入模型並預測
    print("\n[6/7] 載入模型進行預測...")
    
    # 使用所有 fold 的模型進行集成預測
    all_probs = []
    n_folds = meta["n_folds"]
    
    for fold in range(1, n_folds + 1):
        model_path = os.path.join('C:/Users/USER/Desktop/課程/專題/模型訓練/saved_user_model/', f"fold_{fold}_best.pt")
        if not os.path.exists(model_path):
            print(f"  ⚠ 警告: 找不到 fold {fold} 的模型")
            continue
        
        model = UserLogicClassifier(
            EMBED_DIM, 
            num_feats_scaled.shape[1], 
            logic_feats_scaled.shape[1]
        ).to(DEVICE)
        
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        fold_probs = []
        with torch.no_grad():
            for i in range(0, len(grouped), BATCH_SIZE):
                batch_posts = torch.tensor(all_post_embs[i:i+BATCH_SIZE]).to(DEVICE)
                batch_bio = torch.tensor(bio_embs[i:i+BATCH_SIZE]).to(DEVICE)
                batch_num = torch.tensor(num_feats_scaled[i:i+BATCH_SIZE], 
                                        dtype=torch.float32).to(DEVICE)
                batch_logic = torch.tensor(logic_feats_scaled[i:i+BATCH_SIZE], 
                                          dtype=torch.float32).to(DEVICE)
                
                logits = model(batch_posts, batch_bio, batch_num, batch_logic)
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_probs.extend(probs)
        
        all_probs.append(fold_probs)
        print(f"  ✓ Fold {fold} 預測完成")
    
    # 集成預測（平均）
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    
    # 6. 計算評估指標
    print("\n[7/7] 計算評估指標...")
    
    accuracy = accuracy_score(labels, ensemble_preds)
    precision = precision_score(labels, ensemble_preds, zero_division=0)
    recall = recall_score(labels, ensemble_preds, zero_division=0)
    f1 = f1_score(labels, ensemble_preds, zero_division=0)
    auc = roc_auc_score(labels, ensemble_probs)
    
    # 7. 輸出結果
    print("\n" + "="*60)
    print("驗證結果")
    print("="*60)
    print(f"準確率 (Accuracy):  {accuracy:.4f}")
    print(f"精確率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1分數 (F1-Score):  {f1:.4f}")
    print(f"AUC-ROC:            {auc:.4f}")
    
    print("\n" + "-"*60)
    print("分類報告")
    print("-"*60)
    print(classification_report(labels, ensemble_preds, 
                                target_names=['正常', '憂鬱傾向']))
    
    # 8. 儲存結果
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_roc": float(auc),
        "total_users": len(grouped),
        "depression_cases": int(labels.sum()),
        "normal_cases": int((1-labels).sum())
    }
    
    with open(os.path.join(OUTPUT_DIR, "validation_results.json"), "w", 
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 9. 生成視覺化
    print("\n生成視覺化圖表...")
    plot_confusion_matrix(labels, ensemble_preds, 
                         os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plot_roc_curve(labels, ensemble_probs, 
                  os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plot_prediction_distribution(ensemble_probs, labels, 
                                os.path.join(OUTPUT_DIR, "prediction_distribution.png"))
    
    # 10. 儲存詳細預測結果
    detailed_results = grouped[["username", "display_name"]].copy()
    detailed_results["actual_label"] = labels
    detailed_results["predicted_label"] = ensemble_preds
    detailed_results["prediction_probability"] = ensemble_probs
    detailed_results["correct"] = (labels == ensemble_preds)
    
    detailed_results.to_csv(os.path.join(OUTPUT_DIR, "detailed_predictions.csv"), 
                           index=False, encoding="utf-8-sig")
    
    print(f"\n✓ 詳細預測結果已儲存至: {os.path.join(OUTPUT_DIR, 'detailed_predictions.csv')}")
    print(f"✓ 所有結果已儲存至資料夾: {OUTPUT_DIR}")
    
    print("\n" + "="*60)
    print("驗證完成！")
    print("="*60)

# ----------------------
# 執行驗證
# ----------------------
if __name__ == "__main__":
    validate_model()
