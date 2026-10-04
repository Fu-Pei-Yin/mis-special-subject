# ============================================================
# 模型 D：General BiLSTM（BiLSTM，無邏輯特徵）
# 修改重點：evaluate() 補回傳 probs，並儲存 fold_X_predictions.csv
# ============================================================
import os, json, random
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
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
import joblib

# -----------------------
# Config  ← 請修改 DATA_PATH 與 OUTPUT_DIR
# -----------------------
DATA_PATH  = "C:/Users/USER/Desktop/課程/專題/模型訓練/data/new_user_data.xlsx"
OUTPUT_DIR = "C:/Users/USER/Desktop/課程/專題/模型訓練/results_D_general_bilstm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME   = "model_D_general_bilstm"
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM    = 384
MAX_POSTS    = 32
BATCH_SIZE   = 4
EPOCHS       = 30
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SEED         = 42
N_FOLDS      = 5
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA    = 0.001
AUGMENT_PROB = 0.3
NOISE_STD    = 0.05
LSTM_HIDDEN  = 256
LSTM_LAYERS  = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ============================================================
# 1. Load Data
# ============================================================
print("Loading data...")
df = pd.read_excel(DATA_PATH)
for c in ["username","display_name","bio","post_content","post_date","depression_or_not"]:
    if c not in df.columns: raise ValueError(f"Missing column: {c}")

df["bio"]              = df["bio"].fillna("")
df["post_content"]     = df["post_content"].fillna("")
df["post_date"]        = pd.to_datetime(df["post_date"], errors="coerce")
df["depression_or_not"]= pd.to_numeric(df["depression_or_not"], errors="coerce").fillna(0).astype(int)

grouped = df.sort_values(["username","post_date"]).groupby("username").agg({
    "display_name": "first", "bio": "first",
    "post_content": list, "post_date": list,
    "depression_or_not": "max"
}).reset_index()

print(f"Total users: {len(grouped)}")
print(f"Depression: {(grouped['depression_or_not']==1).sum()} | Non-depression: {(grouped['depression_or_not']==0).sum()}")

# ============================================================
# 2. Numeric Features（行為特徵，無邏輯特徵）
# ============================================================
def compute_numeric(post_texts, bio, post_dates):
    n = len(post_texts)
    avg_len = np.mean([len(str(t)) for t in post_texts]) if n > 0 else 0.0
    max_len = max([len(str(t)) for t in post_texts]) if n > 0 else 0.0
    min_len = min([len(str(t)) for t in post_texts]) if n > 0 else 0.0
    night   = sum(1 for d in post_dates if pd.notna(d) and 0 <= pd.to_datetime(d).hour <= 5)
    return [n, avg_len, len(str(bio)), night/n if n>0 else 0.0, max_len, min_len]

num_feats = np.array([compute_numeric(r["post_content"], r["bio"], r["post_date"])
                      for _, r in grouped.iterrows()])

# ============================================================
# 3. Text Embeddings
# ============================================================
st_model = SentenceTransformer(EMBED_MODEL_NAME)
print("Encoding bios..."); bio_embs = np.array(st_model.encode(grouped["bio"].tolist(), show_progress_bar=True), dtype=np.float32)

print("Encoding posts...")
all_post_embs = []
for post_list in tqdm(grouped["post_content"].tolist()):
    texts = [str(t) for t in post_list][:MAX_POSTS]
    if not texts:
        all_post_embs.append(np.zeros((MAX_POSTS, EMBED_DIM), dtype=np.float32)); continue
    emb = st_model.encode(texts, show_progress_bar=False).astype(np.float32)
    if emb.shape[0] < MAX_POSTS:
        emb = np.vstack([emb, np.zeros((MAX_POSTS - emb.shape[0], EMBED_DIM), dtype=np.float32)])
    all_post_embs.append(emb[:MAX_POSTS])
all_post_embs = np.stack(all_post_embs)
labels = grouped["depression_or_not"].astype(int).values

# ============================================================
# 4. Dataset（無 logic_feats）
# ============================================================
class UserDataset(Dataset):
    def __init__(self, post_embs, bio_embs, num_feats, labels, augment=False):
        self.posts = torch.tensor(post_embs)
        self.bio   = torch.tensor(bio_embs)
        self.num   = torch.tensor(num_feats, dtype=torch.float32)
        self.labels= torch.tensor(labels, dtype=torch.float32)
        self.augment = augment
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        p, b = self.posts[i], self.bio[i]
        if self.augment and random.random() < AUGMENT_PROB:
            p = p + torch.randn_like(p) * NOISE_STD
            b = b + torch.randn_like(b) * NOISE_STD
        return {"posts": p, "bio": b, "num": self.num[i], "label": self.labels[i]}

# ============================================================
# 5. BiLSTM Model（無 logic_feats）
# ============================================================
class PostsLSTMEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True, bidirectional=True)
        self.output_dim = hidden_dim * 2
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

class GeneralBiLSTMClassifier(nn.Module):
    def __init__(self, embed_dim, num_num, lstm_hidden=256, lstm_layers=2, hidden=128, dropout=0.4):
        super().__init__()
        self.posts_encoder = PostsLSTMEncoder(embed_dim, lstm_hidden, lstm_layers, dropout)
        self.bio_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        total_dim = self.posts_encoder.output_dim + embed_dim + num_num
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1))
    def forward(self, posts, bio, num_feats):
        x = torch.cat([self.posts_encoder(posts), self.bio_proj(bio), num_feats], dim=1)
        return self.mlp(x).squeeze(-1)

# ============================================================
# 6. Early Stopping
# ============================================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience; self.min_delta = min_delta
        self.counter = 0; self.best_score = None; self.early_stop = False
    def __call__(self, val_score):
        if self.best_score is None: self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else: self.best_score = val_score; self.counter = 0
        return self.early_stop

# ============================================================
# 7. Train / Evaluate
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train(); total_loss = 0.0
    for batch in loader:
        posts = batch["posts"].to(DEVICE); bio = batch["bio"].to(DEVICE)
        num   = batch["num"].to(DEVICE);   label = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(posts, bio, num), label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * posts.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion):
    """回傳 acc, f1, precision, recall, avg_loss, preds, trues, probs"""
    model.eval(); preds, trues, probs_all = [], [], []; total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            posts = batch["posts"].to(DEVICE); bio = batch["bio"].to(DEVICE)
            num   = batch["num"].to(DEVICE);   label = batch["label"].to(DEVICE)
            logits = model(posts, bio, num)
            total_loss += criterion(logits, label).item() * posts.size(0)
            prob = torch.sigmoid(logits).cpu().numpy()
            probs_all.extend(prob.tolist())
            preds.extend((prob >= 0.5).astype(int).tolist())
            trues.extend(label.cpu().numpy().astype(int).tolist())
    acc  = accuracy_score(trues, preds)
    f1   = f1_score(trues, preds, zero_division=0)
    prec = precision_score(trues, preds, zero_division=0)
    rec  = recall_score(trues, preds, zero_division=0)
    return acc, f1, prec, rec, total_loss/len(loader.dataset), preds, trues, probs_all

# ============================================================
# 8. Cross-Validation
# ============================================================
print(f"\n{'='*60}\nStarting {N_FOLDS}-Fold CV — {MODEL_NAME}\n{'='*60}\n")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(all_post_embs, labels), 1):
    print(f"\n{'='*60}\nFOLD {fold}/{N_FOLDS}\n{'='*60}")

    num_scaler = StandardScaler().fit(num_feats[train_idx])

    train_ds = UserDataset(all_post_embs[train_idx], bio_embs[train_idx],
                           num_scaler.transform(num_feats[train_idx]),
                           labels[train_idx], augment=True)
    val_ds   = UserDataset(all_post_embs[val_idx], bio_embs[val_idx],
                           num_scaler.transform(num_feats[val_idx]),
                           labels[val_idx], augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model = GeneralBiLSTMClassifier(EMBED_DIM, num_feats.shape[1],
                                    LSTM_HIDDEN, LSTM_LAYERS).to(DEVICE)

    pos_w = torch.tensor([max(1.0, (len(train_idx)-labels[train_idx].sum())/labels[train_idx].sum())]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    es = EarlyStopping(EARLY_STOPPING_PATIENCE, MIN_DELTA)

    best_f1, best_epoch = 0.0, 0
    for epoch in range(1, EPOCHS+1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        acc, f1, prec, rec, val_loss, _, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(f1)
        print(f"Epoch {epoch:2d} | TrainLoss: {tr_loss:.4f} | ValLoss: {val_loss:.4f} | "
              f"Acc: {acc:.4f} | F1: {f1:.4f} | P: {prec:.4f} | R: {rec:.4f}")
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"fold_{fold}_best.pt"))
        if es(f1): print(f"Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"fold_{fold}_best.pt")))
    acc, f1, prec, rec, val_loss, preds, trues, probs = evaluate(model, val_loader, criterion)

    print(f"\nFold {fold} Best (Epoch {best_epoch}): Acc={acc:.4f} F1={f1:.4f} P={prec:.4f} R={rec:.4f}")
    print(classification_report(trues, preds, target_names=['Non-Depression','Depression']))

    pd.DataFrame({
        "y_true": trues, "y_pred": preds, "probability": probs, "fold": fold
    }).to_csv(os.path.join(OUTPUT_DIR, f"fold_{fold}_predictions.csv"), index=False)

    fold_results.append({
        "fold": fold, "best_epoch": best_epoch,
        "val_acc": acc, "val_f1": f1, "val_precision": prec, "val_recall": rec, "val_loss": val_loss
    })
    joblib.dump(num_scaler, os.path.join(OUTPUT_DIR, f"fold_{fold}_num_scaler.pkl"))

# ============================================================
# 9. Summary
# ============================================================
results_df = pd.DataFrame(fold_results)
print(f"\n{'='*60}\nCROSS-VALIDATION SUMMARY — {MODEL_NAME}\n{'='*60}")
print(results_df.to_string(index=False))
print(f"\nMean ± Std:")
for col in ["val_acc","val_f1","val_precision","val_recall"]:
    print(f"  {col.replace('val_',''):12s}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")

results_df.to_csv(os.path.join(OUTPUT_DIR, "cv_results.csv"), index=False)

meta = {
    "model_name": MODEL_NAME,
    "embed_model": EMBED_MODEL_NAME, "embed_dim": EMBED_DIM, "max_posts": MAX_POSTS,
    "architecture": "BiLSTM", "lstm_hidden": LSTM_HIDDEN, "lstm_layers": LSTM_LAYERS,
    "bidirectional": True, "logic_features": False,
    "n_folds": N_FOLDS, "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY,
    "batch_size": BATCH_SIZE, "seed": SEED, "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "num_features": ["post_count","avg_post_len","bio_len","night_ratio","max_post_len","min_post_len"],
    "avg_acc":  float(results_df["val_acc"].mean()),  "std_acc":  float(results_df["val_acc"].std()),
    "avg_f1":   float(results_df["val_f1"].mean()),   "std_f1":   float(results_df["val_f1"].std()),
    "avg_prec": float(results_df["val_precision"].mean()),
    "avg_rec":  float(results_df["val_recall"].mean()),
}
with open(os.path.join(OUTPUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n✓ 所有結果儲存至: {OUTPUT_DIR}")
print("  fold_X_best.pt | fold_X_predictions.csv | cv_results.csv | meta.json")