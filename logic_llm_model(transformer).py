# ============================================================
# model.py (Enhanced Version with Cross-Validation & Generalization)
# ============================================================
import os
import math
import json
import random
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import joblib

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
EPOCHS = 30  # Increased for early stopping
LR = 1e-3
WEIGHT_DECAY = 1e-4  # Increased regularization
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cross-validation settings
N_FOLDS = 5
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 0.001

# Data augmentation
AUGMENT_PROB = 0.3
NOISE_STD = 0.05

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# 1. Load Data
# ============================================================
print("Loading data...")
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

print(f"Total users: {len(grouped)}")
print(f"Depression cases: {grouped['depression_or_not'].sum()}")
print(f"Non-depression cases: {(1 - grouped['depression_or_not']).sum()}")

# ============================================================
# 2. Numeric Features
# ============================================================
def compute_user_numeric_features(post_texts: List[str], bio: str, post_dates: List[pd.Timestamp]):
    post_count = len(post_texts)
    avg_post_len = np.mean([len(str(t)) for t in post_texts]) if post_count > 0 else 0.0
    bio_len = len(str(bio))
    night_count = sum([1 for d in post_dates if 0 <= pd.to_datetime(d).hour <= 5])
    night_ratio = night_count / post_count if post_count > 0 else 0.0
    
    # Additional features for better generalization
    max_post_len = max([len(str(t)) for t in post_texts]) if post_count > 0 else 0.0
    min_post_len = min([len(str(t)) for t in post_texts]) if post_count > 0 else 0.0
    
    return [post_count, avg_post_len, bio_len, night_ratio, max_post_len, min_post_len]

print("Computing numeric features...")
num_feats = np.array([
    compute_user_numeric_features(r["post_content"], r["bio"], r["post_date"]) 
    for _, r in grouped.iterrows()
])

scaler = StandardScaler()
num_feats_scaled = scaler.fit_transform(num_feats)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "numerical_scaler.pkl"))

# ============================================================
# 3. Text Embeddings (Sentence-BERT)
# ============================================================
print("Loading sentence transformer...")
st_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Encoding bios...")
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

# ============================================================
# 4. Logic-based Features
# ============================================================
def extract_logic_features(post_list: List[str], bio: str) -> List[float]:
    neg_words = ["痛苦", "沒意義", "撐不住", "累", "空虛", "需要陪伴", "消失", "不想活", "孤獨", "絕望"]
    pos_words = ["開心", "期待", "感謝", "喜歡", "平靜", "快樂", "滿足"]
    
    neg_count = sum([sum([w in p for w in neg_words]) for p in post_list])
    pos_count = sum([sum([w in p for w in pos_words]) for p in post_list])
    logic_score = neg_count - pos_count
    bio_flag = 1 if any(w in bio for w in ["抑鬱", "憂鬱", "低潮", "焦慮"]) else 0
    
    # Ratio features for better generalization
    total_count = neg_count + pos_count
    neg_ratio = neg_count / total_count if total_count > 0 else 0
    
    return [neg_count, pos_count, logic_score, bio_flag, neg_ratio]

print("Extracting logic features...")
logic_feats = np.array([
    extract_logic_features(r["post_content"], r["bio"]) 
    for _, r in grouped.iterrows()
], dtype=np.float32)

logic_scaler = StandardScaler()
logic_feats_scaled = logic_scaler.fit_transform(logic_feats)
joblib.dump(logic_scaler, os.path.join(OUTPUT_DIR, "logic_scaler.pkl"))

# ============================================================
# 5. Dataset with Augmentation
# ============================================================
class UserPostsDataset(Dataset):
    def __init__(self, post_embs, bio_embs, numeric_feats, logic_feats, labels, augment=False):
        self.post_embs = torch.tensor(post_embs)
        self.bio_embs = torch.tensor(bio_embs)
        self.num_feats = torch.tensor(numeric_feats, dtype=torch.float32)
        self.logic_feats = torch.tensor(logic_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.augment = augment
    
    def __len__(self):
        return self.post_embs.shape[0]
    
    def __getitem__(self, idx):
        posts = self.post_embs[idx]
        bio = self.bio_embs[idx]
        
        # Data augmentation: add Gaussian noise
        if self.augment and random.random() < AUGMENT_PROB:
            posts = posts + torch.randn_like(posts) * NOISE_STD
            bio = bio + torch.randn_like(bio) * NOISE_STD
        
        return {
            "posts": posts,
            "bio": bio,
            "num": self.num_feats[idx],
            "logic": self.logic_feats[idx],
            "label": self.labels[idx]
        }

# ============================================================
# 6. Enhanced Model
# ============================================================
class PostsEncoder(nn.Module):
    def __init__(self, embed_dim, nhead=4, nhid=256, nlayers=2, dropout=0.3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True
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

# ============================================================
# 7. Early Stopping
# ============================================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop

# ============================================================
# 8. Training and Evaluation Functions
# ============================================================
def evaluate(model, loader, criterion):
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            posts = batch["posts"].to(DEVICE)
            bio = batch["bio"].to(DEVICE)
            num = batch["num"].to(DEVICE)
            logic = batch["logic"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            
            logits = model(posts, bio, num, logic)
            loss = criterion(logits, label)
            total_loss += loss.item() * posts.size(0)
            
            probs = torch.sigmoid(logits)
            preds.extend((probs.cpu().numpy() >= 0.5).astype(int).tolist())
            trues.extend(label.cpu().numpy().astype(int).tolist())
    
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, zero_division=0)
    precision = precision_score(trues, preds, zero_division=0)
    recall = recall_score(trues, preds, zero_division=0)
    avg_loss = total_loss / len(loader.dataset)
    
    return acc, f1, precision, recall, avg_loss, preds, trues

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        posts = batch["posts"].to(DEVICE)
        bio = batch["bio"].to(DEVICE)
        num = batch["num"].to(DEVICE)
        logic = batch["logic"].to(DEVICE)
        label = batch["label"].to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(posts, bio, num, logic)
        loss = criterion(logits, label)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item() * posts.size(0)
    
    return total_loss / len(loader.dataset)

# ============================================================
# 9. Cross-Validation Training
# ============================================================
print(f"\n{'='*60}")
print(f"Starting {N_FOLDS}-Fold Cross-Validation")
print(f"{'='*60}\n")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(all_post_embs, labels), 1):
    print(f"\n{'='*60}")
    print(f"FOLD {fold}/{N_FOLDS}")
    print(f"{'='*60}")
    
    # Create datasets
    train_dataset = UserPostsDataset(
        all_post_embs[train_idx], 
        bio_embs[train_idx],
        num_feats_scaled[train_idx], 
        logic_feats_scaled[train_idx],
        labels[train_idx],
        augment=True
    )
    
    val_dataset = UserPostsDataset(
        all_post_embs[val_idx], 
        bio_embs[val_idx],
        num_feats_scaled[val_idx], 
        logic_feats_scaled[val_idx],
        labels[val_idx],
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = UserLogicClassifier(
        EMBED_DIM, 
        num_feats_scaled.shape[1],
        logic_feats_scaled.shape[1]
    ).to(DEVICE)
    
    # Loss with class weights
    pos_weight = torch.tensor([max(1.0, (len(train_idx) - labels[train_idx].sum()) / labels[train_idx].sum())]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=MIN_DELTA)
    
    best_val_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc, val_f1, val_prec, val_rec, val_loss, _, _ = evaluate(model, val_loader, criterion)
        
        scheduler.step(val_f1)
        
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f} | Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"fold_{fold}_best.pt"))
            print(f"✓ Saved best model for fold {fold}")
        
        if early_stopping(val_f1):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"fold_{fold}_best.pt")))
    val_acc, val_f1, val_prec, val_rec, val_loss, preds, trues = evaluate(model, val_loader, criterion)
    
    fold_results.append({
        'fold': fold,
        'best_epoch': best_epoch,
        'val_acc': val_acc,
        'val_f1': val_f1,
        'val_precision': val_prec,
        'val_recall': val_rec,
        'val_loss': val_loss
    })
    
    print(f"\n{'-'*60}")
    print(f"Fold {fold} Best Results (Epoch {best_epoch}):")
    print(f"Accuracy: {val_acc:.4f} | F1: {val_f1:.4f} | "
          f"Precision: {val_prec:.4f} | Recall: {val_rec:.4f}")
    print(f"{'-'*60}\n")
    print("Classification Report:")
    print(classification_report(trues, preds, target_names=['Non-Depression', 'Depression']))

# ============================================================
# 10. Summary Results
# ============================================================
print(f"\n{'='*60}")
print("CROSS-VALIDATION SUMMARY")
print(f"{'='*60}\n")

results_df = pd.DataFrame(fold_results)
print(results_df.to_string(index=False))

print(f"\n{'-'*60}")
print("Average Performance Across Folds:")
print(f"{'-'*60}")
for metric in ['val_acc', 'val_f1', 'val_precision', 'val_recall', 'val_loss']:
    mean_val = results_df[metric].mean()
    std_val = results_df[metric].std()
    print(f"{metric.replace('val_', '').capitalize():12s}: {mean_val:.4f} ± {std_val:.4f}")

# Save results
results_df.to_csv(os.path.join(OUTPUT_DIR, "cv_results.csv"), index=False)

# Save metadata
meta = {
    "embed_model": EMBED_MODEL_NAME,
    "embed_dim": EMBED_DIM,
    "max_posts": MAX_POSTS,
    "n_folds": N_FOLDS,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "augmentation_prob": AUGMENT_PROB,
    "logic_features": ["neg_count", "pos_count", "logic_score", "bio_flag", "neg_ratio"],
    "numeric_features": ["post_count", "avg_post_len", "bio_len", "night_ratio", "max_post_len", "min_post_len"],
    "average_performance": {
        "accuracy": float(results_df['val_acc'].mean()),
        "f1_score": float(results_df['val_f1'].mean()),
        "precision": float(results_df['val_precision'].mean()),
        "recall": float(results_df['val_recall'].mean())
    }
}

with open(os.path.join(OUTPUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"All artifacts saved to: {OUTPUT_DIR}")
print(f"{'='*60}\n")
print("Saved files:")
print("  - fold_1_best.pt to fold_5_best.pt (model weights)")
print("  - cv_results.csv (cross-validation results)")
print("  - meta.json (model metadata)")
print("  - numerical_scaler.pkl (numeric feature scaler)")
print("  - logic_scaler.pkl (logic feature scaler)")