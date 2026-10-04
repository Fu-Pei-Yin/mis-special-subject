# ============================================================
# 外部驗證（Hold-out External Test）
# 使用原始 5-Fold 訓練好的 General BiLSTM 模型
# 對全新爬取帳號進行預測，評估模型泛化能力
#
# 前置條件：
#   1. 原始訓練已完成，results_D_general_bilstm/ 下有 fold_1~5_best.pt
#   2. 新資料格式：username, display_name, bio, post_content, post_date, depression_or_not
#   3. 新資料帳號與原始 593 人完全不重疊
# ============================================================

import os, json, random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score,
                             average_precision_score, confusion_matrix,
                             classification_report)
import joblib

# ============================================================
# Config  ← 修改這三個路徑
# ============================================================
EXTERNAL_DATA_PATH = "C:/Users/USER/Desktop/課程/專題/模型訓練/data/external_data.xlsx"
TRAIN_MODEL_DIR    = "C:/Users/USER/Desktop/課程/專題/模型訓練/results_D_general_bilstm"
OUTPUT_DIR         = "C:/Users/USER/Desktop/課程/專題/模型訓練/results_external_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM   = 384
MAX_POSTS   = 32
BATCH_SIZE  = 4
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
N_FOLDS     = 5
SEED        = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ============================================================
# 1. 載入外部資料
# ============================================================
print("Loading external data...")
df = pd.read_excel(EXTERNAL_DATA_PATH)
for c in ["username", "display_name", "bio", "post_content", "post_date", "depression_or_not"]:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

df["bio"]               = df["bio"].fillna("")
df["post_content"]      = df["post_content"].fillna("")
df["post_date"]         = pd.to_datetime(df["post_date"], errors="coerce")
df["depression_or_not"] = pd.to_numeric(df["depression_or_not"], errors="coerce").fillna(0).astype(int)

grouped = df.sort_values(["username", "post_date"]).groupby("username").agg({
    "display_name":     "first",
    "bio":              "first",
    "post_content":     list,
    "post_date":        list,
    "depression_or_not":"max"
}).reset_index()

labels = grouped["depression_or_not"].astype(int).values
print(f"外部資料：共 {len(grouped)} 位使用者")
print(f"  憂鬱傾向：{(labels == 1).sum()} 人")
print(f"  非憂鬱：  {(labels == 0).sum()} 人")

# ============================================================
# 2. 特徵建構（與訓練時完全相同）
# ============================================================
def compute_numeric(post_texts, bio, post_dates):
    n = len(post_texts)
    avg_len = np.mean([len(str(t)) for t in post_texts]) if n > 0 else 0.0
    max_len = max([len(str(t)) for t in post_texts]) if n > 0 else 0.0
    min_len = min([len(str(t)) for t in post_texts]) if n > 0 else 0.0
    night   = sum(1 for d in post_dates if pd.notna(d) and 0 <= pd.to_datetime(d).hour <= 5)
    return [n, avg_len, len(str(bio)), night / n if n > 0 else 0.0, max_len, min_len]

num_feats = np.array([
    compute_numeric(r["post_content"], r["bio"], r["post_date"])
    for _, r in grouped.iterrows()
])

st_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Encoding bios...")
bio_embs = np.array(
    st_model.encode(grouped["bio"].tolist(), show_progress_bar=True),
    dtype=np.float32
)

print("Encoding posts...")
all_post_embs = []
for post_list in tqdm(grouped["post_content"].tolist()):
    texts = [str(t) for t in post_list][:MAX_POSTS]
    if not texts:
        all_post_embs.append(np.zeros((MAX_POSTS, EMBED_DIM), dtype=np.float32))
        continue
    emb = st_model.encode(texts, show_progress_bar=False).astype(np.float32)
    if emb.shape[0] < MAX_POSTS:
        emb = np.vstack([emb, np.zeros((MAX_POSTS - emb.shape[0], EMBED_DIM), dtype=np.float32)])
    all_post_embs.append(emb[:MAX_POSTS])
all_post_embs = np.stack(all_post_embs)

# ============================================================
# 3. Dataset
# ============================================================
class UserDataset(Dataset):
    def __init__(self, post_embs, bio_embs, num_feats, labels):
        self.posts  = torch.tensor(post_embs)
        self.bio    = torch.tensor(bio_embs)
        self.num    = torch.tensor(num_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"posts": self.posts[i], "bio": self.bio[i],
                "num": self.num[i], "label": self.labels[i]}

# ============================================================
# 4. 模型定義（與訓練時完全相同）
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
    def __init__(self, embed_dim, num_num, lstm_hidden=256, lstm_layers=2,
                 hidden=128, dropout=0.4):
        super().__init__()
        self.posts_encoder = PostsLSTMEncoder(embed_dim, lstm_hidden, lstm_layers, dropout)
        self.bio_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        total_dim = self.posts_encoder.output_dim + embed_dim + num_num
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden),    nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))
    def forward(self, posts, bio, num_feats):
        x = torch.cat([self.posts_encoder(posts), self.bio_proj(bio), num_feats], dim=1)
        return self.mlp(x).squeeze(-1)

# ============================================================
# 5. 用 5 個 Fold 模型各自預測，取機率平均（Ensemble）
#    每個 Fold 的 num_scaler 來自該 Fold 的訓練集，保持一致
# ============================================================
print(f"\n{'='*60}")
print("開始外部驗證：載入 5 個 Fold 模型逐一預測")
print(f"{'='*60}")

fold_probs_list = []  # 每個 fold 的預測機率

for fold in range(1, N_FOLDS + 1):
    print(f"\n--- Fold {fold} ---")

    # 載入該 Fold 訓練時的 num_scaler
    scaler_path = os.path.join(TRAIN_MODEL_DIR, f"fold_{fold}_num_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"找不到 {scaler_path}，請確認訓練結果目錄正確")
    num_scaler = joblib.load(scaler_path)

    # 用訓練時的 scaler 轉換外部資料的行為特徵
    num_scaled = num_scaler.transform(num_feats)

    dataset = UserDataset(all_post_embs, bio_embs, num_scaled, labels)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 載入模型權重
    model = GeneralBiLSTMClassifier(EMBED_DIM, num_feats.shape[1],
                                    LSTM_HIDDEN, LSTM_LAYERS).to(DEVICE)
    model_path = os.path.join(TRAIN_MODEL_DIR, f"fold_{fold}_best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到 {model_path}，請確認訓練結果目錄正確")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    probs = []
    with torch.no_grad():
        for batch in loader:
            posts = batch["posts"].to(DEVICE)
            bio   = batch["bio"].to(DEVICE)
            num   = batch["num"].to(DEVICE)
            logits = model(posts, bio, num)
            prob   = torch.sigmoid(logits).cpu().numpy()
            probs.extend(prob.tolist())

    fold_probs_list.append(probs)
    fold_preds = (np.array(probs) >= 0.5).astype(int)
    fold_f1  = f1_score(labels, fold_preds, zero_division=0)
    fold_acc = accuracy_score(labels, fold_preds)
    print(f"  Fold {fold} 單獨預測：Acc={fold_acc:.4f}  F1={fold_f1:.4f}")

# ============================================================
# 6. Ensemble：5 個 Fold 機率平均後取最終預測
# ============================================================
ensemble_probs = np.mean(fold_probs_list, axis=0)
ensemble_preds = (ensemble_probs >= 0.5).astype(int)

acc  = accuracy_score(labels, ensemble_preds)
f1   = f1_score(labels, ensemble_preds, zero_division=0)
prec = precision_score(labels, ensemble_preds, zero_division=0)
rec  = recall_score(labels, ensemble_preds, zero_division=0)
try:
    auc    = roc_auc_score(labels, ensemble_probs)
    pr_auc = average_precision_score(labels, ensemble_probs)
except Exception:
    auc = pr_auc = float("nan")

cm = confusion_matrix(labels, ensemble_preds)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

print(f"\n{'='*60}")
print("外部驗證最終結果（5-Fold Ensemble）")
print(f"{'='*60}")
print(f"  樣本數：{len(labels)} 人（憂鬱 {labels.sum()} / 非憂鬱 {len(labels)-labels.sum()}）")
print(f"  Accuracy    : {acc:.4f}")
print(f"  F1-Score    : {f1:.4f}")
print(f"  Precision   : {prec:.4f}")
print(f"  Recall      : {rec:.4f}")
print(f"  Specificity : {specificity:.4f}")
print(f"  ROC-AUC     : {auc:.4f}")
print(f"  PR-AUC      : {pr_auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"             預測非憂鬱  預測憂鬱")
print(f"  實際非憂鬱   TN={tn:3d}      FP={fp:3d}")
print(f"  實際憂鬱     FN={fn:3d}      TP={tp:3d}")
print(f"\n{classification_report(labels, ensemble_preds, target_names=['Non-Depression','Depression'])}")

# ============================================================
# 7. 儲存結果
# ============================================================
pred_df = pd.DataFrame({
    "username":    grouped["username"].tolist(),
    "y_true":      labels.tolist(),
    "y_pred":      ensemble_preds.tolist(),
    "probability": ensemble_probs.tolist(),
})
pred_df.to_csv(os.path.join(OUTPUT_DIR, "external_predictions.csv"), index=False, encoding="utf-8-sig")

results = {
    "n_users":      int(len(labels)),
    "n_depression": int(labels.sum()),
    "n_normal":     int(len(labels) - labels.sum()),
    "accuracy":     float(acc),
    "f1":           float(f1),
    "precision":    float(prec),
    "recall":       float(rec),
    "specificity":  float(specificity),
    "roc_auc":      float(auc),
    "pr_auc":       float(pr_auc),
    "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
    "fold_f1s": [
        float(f1_score(labels, (np.array(p) >= 0.5).astype(int), zero_division=0))
        for p in fold_probs_list
    ]
}
with open(os.path.join(OUTPUT_DIR, "external_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✓ 結果儲存至：{OUTPUT_DIR}")
print("  external_predictions.csv | external_results.json")
