# ============================================================
# Baseline：TF-IDF + SVM & XGBoost
# 對齊 general_llm_model_LSTM_.py 的資料格式與評估流程
# ============================================================
import os, json, random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# Config ← 修改路徑
# -----------------------
DATA_PATH  = "C:/Users/USER/Desktop/課程/專題/模型訓練/data/new_user_data.xlsx"
OUTPUT_DIR = "C:/Users/USER/Desktop/課程/專題/模型訓練/results_baseline_tfidf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED    = 42
N_FOLDS = 5
random.seed(SEED); np.random.seed(SEED)

# ============================================================
# 1. Load & Group Data（對齊 LSTM 腳本的處理方式）
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
    "bio":              "first",
    "post_content":     list,
    "post_date":        list,
    "depression_or_not":"max"
}).reset_index()

print(f"Total users: {len(grouped)}")
print(f"Depression: {(grouped['depression_or_not']==1).sum()} | Non-depression: {(grouped['depression_or_not']==0).sum()}")

# ============================================================
# 2. Feature Engineering
#    文字特徵：將每位使用者的所有貼文 + bio 合併為一段文字
#    行為特徵：post_count, avg_len, bio_len, night_ratio, max_len, min_len
# ============================================================
def build_text(post_list, bio):
    """將 bio + 所有貼文串接為單一字串"""
    all_text = str(bio) + " " + " ".join([str(p) for p in post_list])
    return all_text.strip()

def compute_numeric(post_texts, bio, post_dates):
    n       = len(post_texts)
    avg_len = np.mean([len(str(t)) for t in post_texts]) if n > 0 else 0.0
    max_len = max([len(str(t)) for t in post_texts]) if n > 0 else 0.0
    min_len = min([len(str(t)) for t in post_texts]) if n > 0 else 0.0
    night   = sum(1 for d in post_dates if pd.notna(d) and 0 <= pd.to_datetime(d).hour <= 5)
    return [n, avg_len, len(str(bio)), night / n if n > 0 else 0.0, max_len, min_len]

texts    = [build_text(r["post_content"], r["bio"]) for _, r in grouped.iterrows()]
num_feats= np.array([compute_numeric(r["post_content"], r["bio"], r["post_date"])
                     for _, r in grouped.iterrows()], dtype=np.float32)
labels   = grouped["depression_or_not"].astype(int).values

# ============================================================
# 3. 共用評估函數
# ============================================================
def evaluate_predictions(y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return acc, f1, prec, rec, auc

# ============================================================
# 4. 模型 A：TF-IDF + SVM
# ============================================================
print(f"\n{'='*60}\nBaseline A: TF-IDF + SVM\n{'='*60}\n")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
svm_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
    print(f"  Fold {fold}/{N_FOLDS} ...", end=" ")

    # TF-IDF 只在訓練集上 fit
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform([texts[i] for i in train_idx])
    X_val_tfidf   = tfidf.transform([texts[i] for i in val_idx])

    # 行為特徵 normalize
    scaler  = StandardScaler()
    X_train_num = csr_matrix(scaler.fit_transform(num_feats[train_idx]))
    X_val_num   = csr_matrix(scaler.transform(num_feats[val_idx]))

    X_train = hstack([X_train_tfidf, X_train_num])
    X_val   = hstack([X_val_tfidf,   X_val_num])

    # 計算 class weight 對應 pos_weight 設計
    n_neg = (labels[train_idx] == 0).sum()
    n_pos = (labels[train_idx] == 1).sum()
    cw    = {0: 1.0, 1: max(1.0, n_neg / n_pos)}

    clf = SVC(kernel="rbf", C=1.0, probability=True, class_weight=cw, random_state=SEED)
    clf.fit(X_train, labels[train_idx])

    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]
    y_true = labels[val_idx]

    acc, f1, prec, rec, auc = evaluate_predictions(y_true, y_pred, y_prob)
    print(f"Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Non-Depression", "Depression"]))

    pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "probability": y_prob, "fold": fold
    }).to_csv(os.path.join(OUTPUT_DIR, f"svm_fold_{fold}_predictions.csv"), index=False)

    svm_results.append({"fold": fold, "val_acc": acc, "val_f1": f1,
                         "val_precision": prec, "val_recall": rec, "val_auc": auc})

svm_df = pd.DataFrame(svm_results)
svm_df.to_csv(os.path.join(OUTPUT_DIR, "svm_cv_results.csv"), index=False)
print(f"\nSVM Mean ± Std:")
for col in ["val_acc", "val_f1", "val_precision", "val_recall", "val_auc"]:
    print(f"  {col.replace('val_',''):12s}: {svm_df[col].mean():.4f} ± {svm_df[col].std():.4f}")

# ============================================================
# 5. 模型 B：TF-IDF + XGBoost
# ============================================================
print(f"\n{'='*60}\nBaseline B: TF-IDF + XGBoost\n{'='*60}\n")
xgb_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
    print(f"  Fold {fold}/{N_FOLDS} ...", end=" ")

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform([texts[i] for i in train_idx])
    X_val_tfidf   = tfidf.transform([texts[i] for i in val_idx])

    scaler  = StandardScaler()
    X_train_num = csr_matrix(scaler.fit_transform(num_feats[train_idx]))
    X_val_num   = csr_matrix(scaler.transform(num_feats[val_idx]))

    X_train = hstack([X_train_tfidf, X_train_num]).toarray()
    X_val   = hstack([X_val_tfidf,   X_val_num]).toarray()

    n_neg = (labels[train_idx] == 0).sum()
    n_pos = (labels[train_idx] == 1).sum()
    scale_pos = max(1.0, n_neg / n_pos)

    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos, use_label_encoder=False,
        eval_metric="logloss", random_state=SEED, verbosity=0,
        tree_method="hist"
    )
    clf.fit(X_train, labels[train_idx])

    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]
    y_true = labels[val_idx]

    acc, f1, prec, rec, auc = evaluate_predictions(y_true, y_pred, y_prob)
    print(f"Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Non-Depression", "Depression"]))

    pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "probability": y_prob, "fold": fold
    }).to_csv(os.path.join(OUTPUT_DIR, f"xgb_fold_{fold}_predictions.csv"), index=False)

    xgb_results.append({"fold": fold, "val_acc": acc, "val_f1": f1,
                          "val_precision": prec, "val_recall": rec, "val_auc": auc})

xgb_df = pd.DataFrame(xgb_results)
xgb_df.to_csv(os.path.join(OUTPUT_DIR, "xgb_cv_results.csv"), index=False)
print(f"\nXGBoost Mean ± Std:")
for col in ["val_acc", "val_f1", "val_precision", "val_recall", "val_auc"]:
    print(f"  {col.replace('val_',''):12s}: {xgb_df[col].mean():.4f} ± {xgb_df[col].std():.4f}")

# ============================================================
# 6. Summary JSON
# ============================================================
summary = {
    "TF-IDF+SVM": {
        "avg_acc":  float(svm_df["val_acc"].mean()),   "std_acc":  float(svm_df["val_acc"].std()),
        "avg_f1":   float(svm_df["val_f1"].mean()),    "std_f1":   float(svm_df["val_f1"].std()),
        "avg_prec": float(svm_df["val_precision"].mean()),
        "avg_rec":  float(svm_df["val_recall"].mean()),
        "avg_auc":  float(svm_df["val_auc"].mean()),
    },
    "TF-IDF+XGBoost": {
        "avg_acc":  float(xgb_df["val_acc"].mean()),   "std_acc":  float(xgb_df["val_acc"].std()),
        "avg_f1":   float(xgb_df["val_f1"].mean()),    "std_f1":   float(xgb_df["val_f1"].std()),
        "avg_prec": float(xgb_df["val_precision"].mean()),
        "avg_rec":  float(xgb_df["val_recall"].mean()),
        "avg_auc":  float(xgb_df["val_auc"].mean()),
    }
}
with open(os.path.join(OUTPUT_DIR, "baseline_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n✓ 所有結果儲存至: {OUTPUT_DIR}")
print("  svm_cv_results.csv | xgb_cv_results.csv | baseline_summary.json")
print("  svm_fold_X_predictions.csv | xgb_fold_X_predictions.csv")
