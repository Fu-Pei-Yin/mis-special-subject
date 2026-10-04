# ============================================================
# 消融實驗：General BiLSTM — 四組特徵組合對照
#
# 組合說明：
#   Text Only              → 只用貼文語義（posts encoder）
#   Text + Bio             → 貼文語義 + bio 向量（完整文本）
#   Text + Behavior        → 貼文語義 + 行為數值特徵（num）
#   Text + Bio + Behavior  → 完整輸入（與原始 General BiLSTM 相同）
#
# 修改重點：
#   - GeneralBiLSTMClassifier 新增 use_bio / use_behavior 旗標
#   - 關閉時以零向量替代，MLP 輸入維度保持不變（公平比較）
#   - 外層跑 ablation_configs 迴圈，每組各跑完整 5-Fold CV
#   - 最終輸出 ablation_summary.csv 與 ablation_summary.json
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
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             roc_auc_score, classification_report)
import joblib

# -----------------------
# Config  ← 請修改 DATA_PATH
# -----------------------
DATA_PATH  = "C:/Users/USER/Desktop/課程/專題/模型訓練/data/new_user_data.xlsx"
OUTPUT_DIR = "C:/Users/USER/Desktop/課程/專題/模型訓練/results_ablation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# 四組消融配置
ABLATION_CONFIGS = [
    {"name": "Text Only",             "use_bio": False, "use_behavior": False},
    {"name": "Text + Bio",            "use_bio": True,  "use_behavior": False},
    {"name": "Text + Behavior",       "use_bio": False, "use_behavior": True},
    {"name": "Text + Bio + Behavior", "use_bio": True,  "use_behavior": True},  # 原始完整模型
]

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ============================================================
# 1. Load Data（與原始程式相同）
# ============================================================
print("Loading data...")
df = pd.read_excel(DATA_PATH)
for c in ["username","display_name","bio","post_content","post_date","depression_or_not"]:
    if c not in df.columns: raise ValueError(f"Missing column: {c}")

df["bio"]               = df["bio"].fillna("")
df["post_content"]      = df["post_content"].fillna("")
df["post_date"]         = pd.to_datetime(df["post_date"], errors="coerce")
df["depression_or_not"] = pd.to_numeric(df["depression_or_not"], errors="coerce").fillna(0).astype(int)

grouped = df.sort_values(["username","post_date"]).groupby("username").agg({
    "display_name": "first", "bio": "first",
    "post_content": list, "post_date": list,
    "depression_or_not": "max"
}).reset_index()

print(f"Total users: {len(grouped)}")
print(f"Depression: {(grouped['depression_or_not']==1).sum()} | Non-depression: {(grouped['depression_or_not']==0).sum()}")

# ============================================================
# 2. 行為特徵（與原始程式相同）
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
NUM_FEATS_DIM = num_feats.shape[1]  # = 6

# ============================================================
# 3. Text Embeddings（與原始程式相同，只跑一次）
# ============================================================
st_model = SentenceTransformer(EMBED_MODEL_NAME)
print("Encoding bios...")
bio_embs = np.array(st_model.encode(grouped["bio"].tolist(), show_progress_bar=True), dtype=np.float32)

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
# 4. Dataset（與原始程式相同）
# ============================================================
class UserDataset(Dataset):
    def __init__(self, post_embs, bio_embs, num_feats, labels, augment=False):
        self.posts  = torch.tensor(post_embs)
        self.bio    = torch.tensor(bio_embs)
        self.num    = torch.tensor(num_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.augment = augment
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        p, b = self.posts[i], self.bio[i]
        if self.augment and random.random() < AUGMENT_PROB:
            p = p + torch.randn_like(p) * NOISE_STD
            b = b + torch.randn_like(b) * NOISE_STD
        return {"posts": p, "bio": b, "num": self.num[i], "label": self.labels[i]}

# ============================================================
# 5. 消融版模型
#    use_bio      = False → bio 向量以零向量替代
#    use_behavior = False → num 特徵以零向量替代
#    MLP 輸入維度固定不變，確保公平比較
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

class AblationBiLSTMClassifier(nn.Module):
    def __init__(self, embed_dim, num_num,
                 use_bio=True, use_behavior=True,
                 lstm_hidden=256, lstm_layers=2,
                 hidden=128, dropout=0.4):
        super().__init__()
        self.use_bio      = use_bio
        self.use_behavior = use_behavior

        self.posts_encoder = PostsLSTMEncoder(embed_dim, lstm_hidden, lstm_layers, dropout)
        self.bio_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))

        # MLP 輸入維度固定（不論旗標），才能公平比較
        total_dim = self.posts_encoder.output_dim + embed_dim + num_num
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden),    nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    def forward(self, posts, bio, num_feats):
        post_enc = self.posts_encoder(posts)

        # bio：關閉時以零向量替代
        bio_enc = self.bio_proj(bio) if self.use_bio \
                  else torch.zeros(bio.size(0), bio.size(1), device=bio.device)

        # 行為特徵：關閉時以零向量替代
        num_in = num_feats if self.use_behavior \
                 else torch.zeros_like(num_feats)

        x = torch.cat([post_enc, bio_enc, num_in], dim=1)
        return self.mlp(x).squeeze(-1)

# ============================================================
# 6. Early Stopping（與原始程式相同）
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
# 7. Train / Evaluate（與原始程式相同）
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
    try:    auc = roc_auc_score(trues, probs_all)
    except: auc = float("nan")
    return acc, f1, prec, rec, auc, total_loss/len(loader.dataset), preds, trues, probs_all

# ============================================================
# 8. 主迴圈：逐組消融配置跑完整 5-Fold CV
# ============================================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
all_ablation_results = []

for config in ABLATION_CONFIGS:
    config_name = config["name"]
    use_bio      = config["use_bio"]
    use_behavior = config["use_behavior"]
    config_dir   = os.path.join(OUTPUT_DIR, config_name.replace(" ", "_").replace("+", "plus"))
    os.makedirs(config_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"消融組合：{config_name}")
    print(f"  use_bio={use_bio}  use_behavior={use_behavior}")
    print(f"{'='*60}")

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_post_embs, labels), 1):
        print(f"\n--- Fold {fold}/{N_FOLDS} ---")

        num_scaler = StandardScaler().fit(num_feats[train_idx])

        train_ds = UserDataset(all_post_embs[train_idx], bio_embs[train_idx],
                               num_scaler.transform(num_feats[train_idx]),
                               labels[train_idx], augment=True)
        val_ds   = UserDataset(all_post_embs[val_idx],   bio_embs[val_idx],
                               num_scaler.transform(num_feats[val_idx]),
                               labels[val_idx], augment=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

        model = AblationBiLSTMClassifier(
            embed_dim=EMBED_DIM, num_num=NUM_FEATS_DIM,
            use_bio=use_bio, use_behavior=use_behavior,
            lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS
        ).to(DEVICE)

        pos_w = torch.tensor([
            max(1.0, (len(train_idx) - labels[train_idx].sum()) / labels[train_idx].sum())
        ]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        es = EarlyStopping(EARLY_STOPPING_PATIENCE, MIN_DELTA)

        best_f1, best_epoch = 0.0, 0
        for epoch in range(1, EPOCHS + 1):
            tr_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            acc, f1, prec, rec, auc, val_loss, _, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step(f1)
            print(f"  Epoch {epoch:2d} | Loss: {tr_loss:.4f} | ValLoss: {val_loss:.4f} | "
                  f"Acc: {acc:.4f} | F1: {f1:.4f} | P: {prec:.4f} | R: {rec:.4f} | AUC: {auc:.4f}")
            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
                torch.save(model.state_dict(), os.path.join(config_dir, f"fold_{fold}_best.pt"))
            if es(f1):
                print(f"  Early stopping at epoch {epoch}"); break

        model.load_state_dict(torch.load(os.path.join(config_dir, f"fold_{fold}_best.pt")))
        acc, f1, prec, rec, auc, val_loss, preds, trues, probs = evaluate(model, val_loader, criterion)

        print(f"\n  Fold {fold} Best (Epoch {best_epoch}): "
              f"Acc={acc:.4f} F1={f1:.4f} P={prec:.4f} R={rec:.4f} AUC={auc:.4f}")
        print(classification_report(trues, preds, target_names=["Non-Depression","Depression"]))

        pd.DataFrame({
            "y_true": trues, "y_pred": preds, "probability": probs, "fold": fold
        }).to_csv(os.path.join(config_dir, f"fold_{fold}_predictions.csv"), index=False)

        fold_results.append({
            "fold": fold, "best_epoch": best_epoch,
            "val_acc": acc, "val_f1": f1,
            "val_precision": prec, "val_recall": rec, "val_auc": auc
        })
        joblib.dump(num_scaler, os.path.join(config_dir, f"fold_{fold}_num_scaler.pkl"))

    # 每組消融的統計摘要
    res_df = pd.DataFrame(fold_results)
    res_df.to_csv(os.path.join(config_dir, "cv_results.csv"), index=False)

    summary = {
        "config": config_name,
        "use_bio": use_bio,
        "use_behavior": use_behavior,
        "mean_acc":  float(res_df["val_acc"].mean()),
        "std_acc":   float(res_df["val_acc"].std()),
        "mean_f1":   float(res_df["val_f1"].mean()),
        "std_f1":    float(res_df["val_f1"].std()),
        "mean_prec": float(res_df["val_precision"].mean()),
        "std_prec":  float(res_df["val_precision"].std()),
        "mean_rec":  float(res_df["val_recall"].mean()),
        "std_rec":   float(res_df["val_recall"].std()),
        "mean_auc":  float(res_df["val_auc"].mean()),
        "std_auc":   float(res_df["val_auc"].std()),
        "fold_f1s":  res_df["val_f1"].tolist(),
    }
    all_ablation_results.append(summary)

    print(f"\n  [{config_name}] Mean F1 = {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")

# ============================================================
# 9. 最終彙總輸出
# ============================================================
print(f"\n{'='*60}\n消融實驗總結\n{'='*60}")

# 找完整模型的 F1 作為基準
baseline = next(s for s in all_ablation_results if s["config"] == "Text + Bio + Behavior")
baseline_f1 = baseline["mean_f1"]

rows = []
for s in all_ablation_results:
    delta = s["mean_f1"] - baseline_f1
    rows.append({
        "特徵組合":     s["config"],
        "use_bio":     s["use_bio"],
        "use_behavior":s["use_behavior"],
        "Accuracy":    f"{s['mean_acc']:.4f} ± {s['std_acc']:.4f}",
        "F1":          f"{s['mean_f1']:.4f} ± {s['std_f1']:.4f}",
        "Precision":   f"{s['mean_prec']:.4f} ± {s['std_prec']:.4f}",
        "Recall":      f"{s['mean_rec']:.4f} ± {s['std_rec']:.4f}",
        "ROC-AUC":     f"{s['mean_auc']:.4f} ± {s['std_auc']:.4f}",
        "△F1 vs 完整": f"{delta:+.4f}",
    })
    print(f"  {s['config']:30s} | F1={s['mean_f1']:.4f}±{s['std_f1']:.4f} | △F1={delta:+.4f}")

summary_df = pd.DataFrame(rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "ablation_summary.csv"), index=False, encoding="utf-8-sig")

with open(os.path.join(OUTPUT_DIR, "ablation_summary.json"), "w", encoding="utf-8") as f:
    json.dump(all_ablation_results, f, ensure_ascii=False, indent=2)

print(f"\n✓ 所有消融結果儲存至: {OUTPUT_DIR}")
print("  ablation_summary.csv | ablation_summary.json | <各組>/cv_results.csv")