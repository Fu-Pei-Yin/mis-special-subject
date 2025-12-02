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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------
# Config
# -----------------------
DATA_PATH = "C:/Users/USER/Desktop/課程/專題/user.xlsx"
OUTPUT_DIR = "./saved_user_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384
MAX_POSTS = 32
BATCH_SIZE = 4  # 小資料集降低 batch size
EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------
# Data preparation
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
    avg_post_len = np.mean([len(str(t)) for t in post_texts]) if post_count>0 else 0.0
    bio_len = len(str(bio))
    night_count = sum([1 for d in post_dates if 0 <= pd.to_datetime(d).hour <=5])
    night_ratio = night_count / post_count if post_count>0 else 0.0
    return [post_count, avg_post_len, bio_len, night_ratio]

num_feats = np.array([compute_user_numeric_features(r["post_content"], r["bio"], r["post_date"])
                      for _, r in grouped.iterrows()])

scaler = StandardScaler()
num_feats_scaled = scaler.fit_transform(num_feats)
import joblib
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "numerical_scaler.pkl"))

# -----------------------
# Text embeddings
# -----------------------
st_model = SentenceTransformer(EMBED_MODEL_NAME)
bio_embs = np.array(st_model.encode(grouped["bio"].tolist(), show_progress_bar=True), dtype=np.float32)

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

train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=SEED, stratify=labels)
train_dataset = UserPostsDataset(all_post_embs[train_idx], bio_embs[train_idx], num_feats_scaled[train_idx], labels[train_idx])
test_dataset  = UserPostsDataset(all_post_embs[test_idx], bio_embs[test_idx], num_feats_scaled[test_idx], labels[test_idx])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------
# Model
# -----------------------
class PostsEncoder(nn.Module):
    def __init__(self, embed_dim, nhead=4, nhid=256, nlayers=1, dropout=0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                                                   dim_feedforward=nhid, dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
    def forward(self, x):
        out = self.transformer(x)
        pooled = out.mean(dim=1)
        return pooled

class UserClassifier(nn.Module):
    def __init__(self, embed_dim, num_num_feats, hidden=128, dropout=0.3):
        super().__init__()
        self.posts_encoder = PostsEncoder(embed_dim)
        self.bio_proj = nn.Linear(embed_dim, embed_dim)
        total_dim = embed_dim*2 + num_num_feats
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, posts, bio, num_feats):
        posts_vec = self.posts_encoder(posts)
        bio_vec = torch.relu(self.bio_proj(bio))
        x = torch.cat([posts_vec, bio_vec, num_feats], dim=1)
        logits = self.mlp(x).squeeze(-1)
        return logits

model = UserClassifier(EMBED_DIM, num_feats_scaled.shape[1]).to(DEVICE)

# -----------------------
# Loss with class weights
# -----------------------
pos_weight = torch.tensor([max(1.0, (len(labels)-labels.sum())/labels.sum())]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# -----------------------
# Training
# -----------------------
def evaluate(loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            posts = batch["posts"].to(DEVICE)
            bio = batch["bio"].to(DEVICE)
            num = batch["num"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            logits = model(posts, bio, num)
            probs = torch.sigmoid(logits)
            preds.extend((probs.cpu().numpy() >= 0.5).astype(int).tolist())
            trues.extend(label.cpu().numpy().astype(int).tolist())
    acc = accuracy_score(trues, preds)
    return acc, preds, trues

best_acc = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        posts = batch["posts"].to(DEVICE)
        bio = batch["bio"].to(DEVICE)
        num = batch["num"].to(DEVICE)
        label = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        logits = model(posts, bio, num)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * posts.size(0)
    avg_loss = total_loss / len(train_dataset)
    val_acc, _, _ = evaluate(test_loader)
    print(f"Epoch {epoch} | Train Loss {avg_loss:.4f} | Val Acc {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "user_model.pt"))
        print("Saved best model.")

val_acc, preds, trues = evaluate(test_loader)
print("Final val acc:", val_acc)
print(classification_report(trues, preds))

# save metadata
meta = {"embed_model": EMBED_MODEL_NAME, "embed_dim": EMBED_DIM, "max_posts": MAX_POSTS}
with open(os.path.join(OUTPUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("All artifacts saved to:", OUTPUT_DIR)
