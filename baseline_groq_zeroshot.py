# ============================================================
# Baseline：Groq (llama-3.1-8b-instant) Zero-Shot / Few-Shot 直接分類
# 對齊 general_llm_model_LSTM_.py 的資料格式與評估流程
# 目的：證明 LLM 標註後用 BiLSTM 推論的優勢（成本與效能）
# ============================================================
import os, json, random, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from groq import Groq
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# Config ← 修改路徑與 API key
# -----------------------
DATA_PATH    = "C:/Users/USER/Desktop/課程/專題/模型訓練/data/new_user_data.xlsx"
OUTPUT_DIR   = "C:/Users/USER/Desktop/課程/專題/模型訓練/results_baseline_groq"
GROQ_API_KEY = "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"   # ← 填入你的 Groq API key
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED      = 42
N_FOLDS   = 5
MAX_POSTS = 10    # 每位 user 最多取幾則貼文（節省 token）
MODE      = "few_shot"  # "zero_shot" 或 "few_shot"
SLEEP_SEC = 0.5          # 每次 API 呼叫後等待秒數

random.seed(SEED); np.random.seed(SEED)
client = Groq(api_key=GROQ_API_KEY)

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
# 2. Prompt 設計
# ============================================================
SYSTEM_PROMPT = """你是一位心理健康風險評估研究員。
你的任務是根據社群媒體使用者的自我介紹與近期貼文，判斷該使用者是否具有高憂鬱風險。

注意事項：
1. 僅依據文字內容判斷，不進行臨床診斷
2. 判斷標準：貼文中是否持續出現憂鬱傾向語言（如情緒低落、無望感、社交退縮、睡眠問題等）
3. 只輸出數字 0 或 1，絕對不要有其他說明、解釋或標點符號
   0 = 低憂鬱風險
   1 = 高憂鬱風險"""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": """自介：心很累，活著沒什麼意思
近期貼文：
- 今天又沒有睡好，腦袋一片空白
- 覺得自己很沒用，什麼都做不好
- 不想出門，一個人待著比較好"""
    },
    {"role": "assistant", "content": "1"},
    {
        "role": "user",
        "content": """自介：喜歡旅遊和美食
近期貼文：
- 今天去吃了超好吃的拉麵！
- 週末要去爬山，好期待
- 新買的相機拍起來超美"""
    },
    {"role": "assistant", "content": "0"},
]

def build_user_message(bio, post_list):
    posts_text = "\n".join([f"- {str(p)}" for p in post_list[:MAX_POSTS]])
    return f"自介：{str(bio)}\n近期貼文：\n{posts_text}"

def call_groq(bio, post_list, mode="zero_shot", retries=3):
    user_msg = build_user_message(bio, post_list)

    if mode == "few_shot":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + \
                   FEW_SHOT_EXAMPLES + \
                   [{"role": "user", "content": user_msg}]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg}
        ]

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=5,
                temperature=0.0,   # 確保輸出穩定
            )
            text = response.choices[0].message.content.strip()
            for c in text:
                if c in "01":
                    return int(c), 1.0 if c == "1" else 0.0
            return 0, 0.0
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return 0, 0.0

# ============================================================
# 3. 全量推論（有快取，中斷後重跑不重複消耗 quota）
# ============================================================
print(f"\n{'='*60}\nBaseline: Groq llama-3.1-8b {MODE}\n{'='*60}\n")

cache_path = os.path.join(OUTPUT_DIR, f"groq_{MODE}_all_predictions.csv")
if os.path.exists(cache_path):
    print(f"Found cache: {cache_path}, loading...")
    all_pred_df = pd.read_csv(cache_path)
    all_preds = all_pred_df["y_pred"].values
    all_probs = all_pred_df["probability"].values
else:
    all_preds, all_probs = [], []
    for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Calling Groq"):
        pred, prob = call_groq(row["bio"], row["post_content"], mode=MODE)
        all_preds.append(pred)
        all_probs.append(prob)
        time.sleep(SLEEP_SEC)

    pd.DataFrame({
        "username":    grouped["username"].values,
        "y_true":      grouped["depression_or_not"].values,
        "y_pred":      all_preds,
        "probability": all_probs
    }).to_csv(cache_path, index=False)
    print(f"  Saved predictions to {cache_path}")

labels    = grouped["depression_or_not"].astype(int).values
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# ============================================================
# 4. 用相同 StratifiedKFold 計算 fold-wise metrics
# ============================================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(labels, labels), 1):
    y_true = labels[val_idx]
    y_pred = all_preds[val_idx]
    y_prob = all_probs[val_idx]

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    print(f"Fold {fold}: Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Non-Depression", "Depression"]))

    pd.DataFrame({
        "y_true": y_true, "y_pred": y_pred, "probability": y_prob, "fold": fold
    }).to_csv(os.path.join(OUTPUT_DIR, f"fold_{fold}_predictions.csv"), index=False)

    fold_results.append({
        "fold": fold, "val_acc": acc, "val_f1": f1,
        "val_precision": prec, "val_recall": rec, "val_auc": auc
    })

# ============================================================
# 5. Summary
# ============================================================
results_df = pd.DataFrame(fold_results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "cv_results.csv"), index=False)
print(f"\nGroq llama-3.1-8b {MODE} Mean ± Std:")
for col in ["val_acc", "val_f1", "val_precision", "val_recall", "val_auc"]:
    print(f"  {col.replace('val_',''):12s}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")

meta = {
    "model_name":  "groq/llama-3.1-8b-instant",
    "mode":         MODE,
    "max_posts":    MAX_POSTS,
    "n_folds":      N_FOLDS,
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
print("  cv_results.csv | meta.json | groq_zero_shot_all_predictions.csv")
print("\n注意：推論結果已快取，重跑時直接讀取快取，不再呼叫 API。")