import os
import math
import json
import random
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

# -----------------------
# Config
# -----------------------
DATA_PATH = "C:/Users/USER/Desktop/課程/專題/threads_data_14days.xlsx"
OUTPUT_DIR = "./saved_user_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384
MAX_POSTS = 32
BATCH_SIZE = 4
EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42
N_FOLDS = 5  # K-Fold 交叉驗證的折數
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# ========== Model Definition ==============
# ============================================================

class PostsEncoder(nn.Module):
    def __init__(self, embed_dim, nhead=4, nhid=256, nlayers=1, dropout=0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def forward(self, x):
        """
        x: (batch, seq_len, embed_dim)
        """
        out = self.transformer(x)
        pooled = out.mean(dim=1)
        return pooled


class GeneralLLMClassifier(nn.Module):
    """
    一般型 LLM 模型：多模態特徵融合 (posts + bio + numeric)
    """
    def __init__(self, embed_dim, num_num_feats, hidden=128, dropout=0.3):
        super().__init__()
        self.posts_encoder = PostsEncoder(embed_dim)
        self.bio_proj = nn.Linear(embed_dim, embed_dim)

        total_dim = embed_dim * 2 + num_num_feats

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, posts, bio, num_feats):
        posts_vec = self.posts_encoder(posts)
        bio_vec = torch.relu(self.bio_proj(bio))
        x = torch.cat([posts_vec, bio_vec, num_feats], dim=1)
        logits = self.mlp(x).squeeze(-1)
        return logits


# ============================================================
# ========== Data Preparation ==========
# ============================================================

# -----------------------
# Load data
# -----------------------
df = pd.read_excel(DATA_PATH)
expected_cols = ["username", "display_name", "bio", "post_content", "post_date", "depression_or_not"]
for c in expected_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

df["bio"] = df["bio"].fillna("")
df["post_content"] = df["post_content"].fillna("")
df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")
df["depression_or_not"] = pd.to_numeric(df["depression_or_not"], errors="coerce").fillna(0).astype(int)

grouped = df.sort_values(["username", "post_date"]).groupby("username").agg({
    "display_name": "first",
    "bio": "first",
    "post_content": lambda x: list(x),
    "post_date": lambda x: list(x),
    "depression_or_not": "max"
}).reset_index()

# -----------------------
# Numeric features
# -----------------------
def compute_user_numeric_features(post_texts: List[str], bio: str, post_dates: List[pd.Timestamp]):
    post_count = len(post_texts)
    avg_post_len = np.mean([len(str(t)) for t in post_texts]) if post_count > 0 else 0.0
    bio_len = len(str(bio))
    night_count = sum([1 for d in post_dates if 0 <= pd.to_datetime(d).hour <= 5])
    night_ratio = night_count / post_count if post_count > 0 else 0.0
    return [post_count, avg_post_len, bio_len, night_ratio]

num_feats = np.array([compute_user_numeric_features(r["post_content"], r["bio"], r["post_date"])
                      for _, r in grouped.iterrows()])

# -----------------------
# Text embeddings
# -----------------------
st_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Encoding bio texts...")
bio_embs = np.array(st_model.encode(grouped["bio"].tolist(), show_progress_bar=True), dtype=np.float32)

print("Encoding posts...")
all_post_embs = []
for post_list in tqdm(grouped["post_content"].tolist(), desc="Encoding posts"):
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
labels = grouped["depression_or_not"].astype(int).values

# -----------------------
# Dataset
# -----------------------
class UserPostsDataset(Dataset):
    def __init__(self, post_embs, bio_embs, numeric_feats, labels):
        self.post_embs = torch.tensor(post_embs)
        self.bio_embs = torch.tensor(bio_embs)
        self.num_feats = torch.tensor(numeric_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return self.post_embs.shape[0]

    def __getitem__(self, idx):
        return {
            "posts": self.post_embs[idx],
            "bio": self.bio_embs[idx],
            "num": self.num_feats[idx],
            "label": self.labels[idx]
        }

# ============================================================
# ========== Training & Evaluation Functions ==========
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        posts = batch["posts"].to(device)
        bio = batch["bio"].to(device)
        num = batch["num"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(posts, bio, num)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * posts.size(0)
    
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    
    with torch.no_grad():
        for batch in loader:
            posts = batch["posts"].to(device)
            bio = batch["bio"].to(device)
            num = batch["num"].to(device)
            label = batch["label"].to(device)

            logits = model(posts, bio, num)
            probs = torch.sigmoid(logits)

            preds.extend((probs.cpu().numpy() >= 0.5).astype(int).tolist())
            trues.extend(label.cpu().numpy().astype(int).tolist())

    acc = accuracy_score(trues, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(trues, preds, average='binary', zero_division=0)
    
    return acc, precision, recall, f1, preds, trues


# ============================================================
# ========== K-Fold Cross Validation ==========
# ============================================================

print(f"\n{'='*60}")
print(f"開始 {N_FOLDS}-Fold 交叉驗證")
print(f"{'='*60}\n")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# 儲存每個 fold 的結果
cv_results = {
    'fold': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}

best_overall_acc = 0.0
best_fold = -1

for fold, (train_idx, val_idx) in enumerate(skf.split(all_post_embs, labels), 1):
    print(f"\n{'─'*60}")
    print(f"Fold {fold}/{N_FOLDS}")
    print(f"{'─'*60}")
    
    # 為每個 fold 創建新的 scaler
    scaler = StandardScaler()
    num_feats_train = scaler.fit_transform(num_feats[train_idx])
    num_feats_val = scaler.transform(num_feats[val_idx])
    
    # 創建 datasets
    train_dataset = UserPostsDataset(
        all_post_embs[train_idx], 
        bio_embs[train_idx], 
        num_feats_train, 
        labels[train_idx]
    )
    val_dataset = UserPostsDataset(
        all_post_embs[val_idx], 
        bio_embs[val_idx], 
        num_feats_val, 
        labels[val_idx]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化模型
    model = GeneralLLMClassifier(EMBED_DIM, num_feats_train.shape[1]).to(DEVICE)
    
    # Loss with class weights
    train_labels = labels[train_idx]
    pos_weight = torch.tensor([max(1.0, (len(train_labels) - train_labels.sum()) / train_labels.sum())]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # 訓練
    best_fold_acc = 0.0
    best_fold_model_state = None
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(model, val_loader, DEVICE)
        
        print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | P: {val_precision:.4f} | R: {val_recall:.4f} | F1: {val_f1:.4f}")
        
        if val_acc > best_fold_acc:
            best_fold_acc = val_acc
            best_fold_model_state = model.state_dict().copy()
    
    # 使用最佳模型評估
    model.load_state_dict(best_fold_model_state)
    final_acc, final_precision, final_recall, final_f1, preds, trues = evaluate(model, val_loader, DEVICE)
    
    print(f"\n📊 Fold {fold} 最佳結果:")
    print(f"   Accuracy:  {final_acc:.4f}")
    print(f"   Precision: {final_precision:.4f}")
    print(f"   Recall:    {final_recall:.4f}")
    print(f"   F1-Score:  {final_f1:.4f}")
    
    # 儲存結果
    cv_results['fold'].append(fold)
    cv_results['accuracy'].append(final_acc)
    cv_results['precision'].append(final_precision)
    cv_results['recall'].append(final_recall)
    cv_results['f1'].append(final_f1)
    
    # 追蹤最佳 fold
    if final_acc > best_overall_acc:
        best_overall_acc = final_acc
        best_fold = fold
        
        # 保存最佳模型
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "user_model.pt"))
        import joblib
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, "numerical_scaler.pkl"))
        print(f"   ✓ 保存為目前最佳模型")

# ============================================================
# ========== Cross Validation Summary ==========
# ============================================================

print(f"\n{'='*60}")
print(f"交叉驗證總結")
print(f"{'='*60}\n")

cv_df = pd.DataFrame(cv_results)

print("各 Fold 詳細結果:")
print(cv_df.to_string(index=False))

print(f"\n{'─'*60}")
print("平均指標 (Mean ± Std):")
print(f"{'─'*60}")
print(f"Accuracy:  {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")
print(f"Precision: {cv_df['precision'].mean():.4f} ± {cv_df['precision'].std():.4f}")
print(f"Recall:    {cv_df['recall'].mean():.4f} ± {cv_df['recall'].std():.4f}")
print(f"F1-Score:  {cv_df['f1'].mean():.4f} ± {cv_df['f1'].std():.4f}")

print(f"\n🏆 最佳模型來自 Fold {best_fold} (Accuracy: {best_overall_acc:.4f})")

# 保存交叉驗證結果
cv_df.to_csv(os.path.join(OUTPUT_DIR, "cv_results.csv"), index=False)

# 保存 metadata
meta = {
    "embed_model": EMBED_MODEL_NAME, 
    "embed_dim": EMBED_DIM, 
    "max_posts": MAX_POSTS,
    "n_folds": N_FOLDS,
    "best_fold": int(best_fold),
    "cv_mean_accuracy": float(cv_df['accuracy'].mean()),
    "cv_std_accuracy": float(cv_df['accuracy'].std()),
    "cv_mean_f1": float(cv_df['f1'].mean()),
    "cv_std_f1": float(cv_df['f1'].std())
}

with open(os.path.join(OUTPUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n✓ 所有檔案已保存至: {OUTPUT_DIR}")
print(f"  - user_model.pt (最佳模型)")
print(f"  - numerical_scaler.pkl (特徵標準化器)")
print(f"  - cv_results.csv (交叉驗證詳細結果)")
print(f"  - meta.json (模型元資訊)")