# ============================================================
# Baseline：Chinese BERT / MacBERT
# 對齊 general_llm_model_LSTM_.py 的資料格式與評估流程
# 預設使用 hfl/chinese-macbert-base（可改為 bert-base-chinese）
# ============================================================
import os, json, random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# Config ← 修改路徑
# 若要換模型：將 BERT_MODEL_NAME 改為 "bert-base-chinese"
# -----------------------
DATA_PATH      = "C:/Users/USER/Desktop/課程/專題/模型訓練/data/new_user_data.xlsx"
OUTPUT_DIR     = "C:/Users/USER/Desktop/課程/專題/模型訓練/results_baseline_macbert"
BERT_MODEL_NAME= "hfl/chinese-macbert-base"   # 或 "bert-base-chinese"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LENGTH  = 512       # BERT 最大 token 數
BATCH_SIZE  = 8         # GPU 記憶體不足時改為 4
EPOCHS      = 10
LR          = 2e-5
WEIGHT_DECAY= 1e-4
SEED        = 42
N_FOLDS     = 5
EARLY_STOPPING_PATIENCE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ============================================================
# 1. Load & Group Data
# ============================================================
print("Loading data...")
df = pd.read_excel(DATA_PATH)
for c in ["username", "bio", "post_content", "post_date", "depression_or_not"]:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

df["bio"]               = df["bio"].fillna("")
df["post_content"]      = df["post_content"].fillna("")
df["post_date"]         = pd.to_datetime(df["post_date"], errors="coerce")
df["depression_or_not"] = pd.to_numeric(df["depression_or_not"], errors="coerce").fillna(0).astype(int)

grouped = df.sort_values(["username", "post_date"]).groupby("username").agg({
    "bio":               "first",
    "post_content":      list,
    "post_date":         list,
    "depression_or_not": "max"
}).reset_index()

print(f"Total users: {len(grouped)}")
print(f"Depression: {(grouped['depression_or_not']==1).sum()} | Non-depression: {(grouped['depression_or_not']==0).sum()}")

# ============================================================
# 2. 文字處理：將每位 user 的所有貼文 + bio 合併，截斷至 MAX_LENGTH
# ============================================================
def build_text(post_list, bio):
    """bio + 所有貼文串接，BERT tokenizer 會自行截斷"""
    return str(bio) + " [SEP] " + " ".join([str(p) for p in post_list])

texts  = [build_text(r["post_content"], r["bio"]) for _, r in grouped.iterrows()]
labels = grouped["depression_or_not"].astype(int).values

# ============================================================
# 3. Dataset
# ============================================================
class BertUserDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids":      self.encodings["input_ids"][i],
            "attention_mask": self.encodings["attention_mask"][i],
            "label":          self.labels[i]
        }

# ============================================================
# 4. Model：BERT + MLP head
# ============================================================
class BertClassifier(nn.Module):
    def __init__(self, bert_model_name, dropout=0.3):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(bert_model_name)
        hidden_size  = self.bert.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),          nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb  = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        return self.mlp(cls_emb).squeeze(-1)

# ============================================================
# 5. Early Stopping
# ============================================================
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
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
# 6. Train / Evaluate
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train(); total_loss = 0.0
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbl  = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(ids, mask), lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * ids.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval(); preds, trues, probs_all = [], [], []; total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbl  = batch["label"].to(DEVICE)
            logits = model(ids, mask)
            total_loss += criterion(logits, lbl).item() * ids.size(0)
            prob = torch.sigmoid(logits).cpu().numpy()
            probs_all.extend(prob.tolist())
            preds.extend((prob >= 0.5).astype(int).tolist())
            trues.extend(lbl.cpu().numpy().astype(int).tolist())
    acc  = accuracy_score(trues, preds)
    f1   = f1_score(trues, preds, zero_division=0)
    prec = precision_score(trues, preds, zero_division=0)
    rec  = recall_score(trues, preds, zero_division=0)
    try:
        auc = roc_auc_score(trues, probs_all)
    except Exception:
        auc = float("nan")
    return acc, f1, prec, rec, total_loss / len(loader.dataset), preds, trues, probs_all, auc

# ============================================================
# 7. Cross-Validation
# ============================================================
print(f"\n{'='*60}\nStarting {N_FOLDS}-Fold CV — {BERT_MODEL_NAME}\n{'='*60}\n")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
    print(f"\n{'='*60}\nFOLD {fold}/{N_FOLDS}\n{'='*60}")

    train_texts = [texts[i] for i in train_idx]
    val_texts   = [texts[i] for i in val_idx]

    train_ds = BertUserDataset(train_texts, labels[train_idx], tokenizer, MAX_LENGTH)
    val_ds   = BertUserDataset(val_texts,   labels[val_idx],   tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model = BertClassifier(BERT_MODEL_NAME).to(DEVICE)

    n_neg = (labels[train_idx] == 0).sum()
    n_pos = (labels[train_idx] == 1).sum()
    pos_w = torch.tensor([max(1.0, n_neg / n_pos)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    es = EarlyStopping(EARLY_STOPPING_PATIENCE)

    best_f1, best_epoch = 0.0, 0
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        acc, f1, prec, rec, val_loss, _, _, _, auc = evaluate(model, val_loader, criterion)
        scheduler.step(f1)
        print(f"Epoch {epoch:2d} | TrainLoss: {tr_loss:.4f} | ValLoss: {val_loss:.4f} | "
              f"Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"fold_{fold}_best.pt"))
        if es(f1):
            print(f"Early stopping at epoch {epoch}"); break

    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"fold_{fold}_best.pt")))
    acc, f1, prec, rec, val_loss, preds, trues, probs, auc = evaluate(model, val_loader, criterion)

    print(f"\nFold {fold} Best (Epoch {best_epoch}): Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")
    print(classification_report(trues, preds, target_names=["Non-Depression", "Depression"]))

    pd.DataFrame({
        "y_true": trues, "y_pred": preds, "probability": probs, "fold": fold
    }).to_csv(os.path.join(OUTPUT_DIR, f"fold_{fold}_predictions.csv"), index=False)

    fold_results.append({
        "fold": fold, "best_epoch": best_epoch,
        "val_acc": acc, "val_f1": f1,
        "val_precision": prec, "val_recall": rec, "val_auc": auc
    })

# ============================================================
# 8. Summary
# ============================================================
results_df = pd.DataFrame(fold_results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "cv_results.csv"), index=False)
print(f"\n{'='*60}\nCROSS-VALIDATION SUMMARY — {BERT_MODEL_NAME}\n{'='*60}")
print(results_df.to_string(index=False))
print(f"\nMean ± Std:")
for col in ["val_acc", "val_f1", "val_precision", "val_recall", "val_auc"]:
    print(f"  {col.replace('val_',''):12s}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")

meta = {
    "model_name":    BERT_MODEL_NAME,
    "max_length":    MAX_LENGTH,
    "batch_size":    BATCH_SIZE,
    "epochs":        EPOCHS,
    "lr":            LR,
    "n_folds":       N_FOLDS,
    "seed":          SEED,
    "avg_acc":  float(results_df["val_acc"].mean()),
    "std_acc":  float(results_df["val_acc"].std()),
    "avg_f1":   float(results_df["val_f1"].mean()),
    "std_f1":   float(results_df["val_f1"].std()),
    "avg_prec": float(results_df["val_precision"].mean()),
    "avg_rec":  float(results_df["val_recall"].mean()),
    "avg_auc":  float(results_df["val_auc"].mean()),
}
with open(os.path.join(OUTPUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n✓ 所有結果儲存至: {OUTPUT_DIR}")
print("  fold_X_best.pt | fold_X_predictions.csv | cv_results.csv | meta.json")
