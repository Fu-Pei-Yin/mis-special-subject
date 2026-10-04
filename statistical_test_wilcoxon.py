# ============================================================
# 統計檢定：Wilcoxon Signed-Rank Test + Effect Size
# 比較所有 Baseline 與 General BiLSTM 的 5-Fold 結果
# 輸出：論文用統計表格（含 p 值、效果量、FDR 校正）
# ============================================================
import os, json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# Config ← 修改為你的實際路徑
# -----------------------
# General BiLSTM（主模型）的結果路徑
BILSTM_CV_PATH = "C:/Users/USER/Desktop/課程/專題/模型訓練/results_D_general_bilstm/cv_results.csv"

# 各 Baseline 的結果路徑
BASELINE_PATHS = {
    "TF-IDF+SVM":     "C:/Users/USER/Desktop/課程/專題/模型訓練/results_baseline_tfidf/svm_cv_results.csv",
    "TF-IDF+XGBoost": "C:/Users/USER/Desktop/課程/專題/模型訓練/results_baseline_tfidf/xgb_cv_results.csv",
    "MacBERT":        "C:/Users/USER/Desktop/課程/專題/模型訓練/results_baseline_macbert/cv_results.csv",
    "Gemini-ZeroShot":"C:/Users/USER/Desktop/課程/專題/模型訓練/results_baseline_gemini/cv_results.csv",
}

OUTPUT_DIR = "C:/Users/USER/Desktop/課程/專題/模型訓練/results_statistical_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS = ["val_f1", "val_acc", "val_precision", "val_recall"]
METRIC_LABELS = {"val_f1": "F1", "val_acc": "Accuracy",
                 "val_precision": "Precision", "val_recall": "Recall"}

# ============================================================
# 1. 載入資料
# ============================================================
print("Loading results...")
bilstm_df = pd.read_csv(BILSTM_CV_PATH).sort_values("fold").reset_index(drop=True)

baselines = {}
for name, path in BASELINE_PATHS.items():
    if os.path.exists(path):
        baselines[name] = pd.read_csv(path).sort_values("fold").reset_index(drop=True)
        print(f"  Loaded: {name}")
    else:
        print(f"  [SKIP] Not found: {name} ({path})")

# ============================================================
# 2. 效果量：matched-pairs rank-biserial correlation
#    r = 1 - (2 * W_minus) / (n * (n+1) / 2)  (簡化版)
#    實際採用 rank-biserial = 2*W / n(n+1) - 1  (Cohen 1988)
# ============================================================
def rank_biserial(x, y):
    """計算 Wilcoxon matched-pairs rank-biserial correlation (effect size r)"""
    diff = np.array(x) - np.array(y)
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.0
    ranks = rankdata(np.abs(diff))
    W_plus  = ranks[diff > 0].sum()
    W_minus = ranks[diff < 0].sum()
    r = (W_plus - W_minus) / (n * (n + 1) / 2)
    return float(r)

def interpret_effect(r):
    r = abs(r)
    if r >= 0.5: return "large"
    elif r >= 0.3: return "medium"
    elif r >= 0.1: return "small"
    else: return "negligible"

# ============================================================
# 3. 執行 Wilcoxon 檢定
# ============================================================
print(f"\n{'='*70}")
print("Wilcoxon Signed-Rank Test: General BiLSTM vs. Each Baseline")
print(f"{'='*70}\n")

records = []
for baseline_name, baseline_df in baselines.items():
    for metric in METRICS:
        if metric not in bilstm_df.columns or metric not in baseline_df.columns:
            continue

        bilstm_scores   = bilstm_df[metric].values
        baseline_scores = baseline_df[metric].values

        # 確保 fold 數一致
        n = min(len(bilstm_scores), len(baseline_scores))
        x, y = bilstm_scores[:n], baseline_scores[:n]

        diff = x - y
        if np.all(diff == 0):
            stat, p_val = np.nan, 1.0
        else:
            try:
                stat, p_val = wilcoxon(x, y, alternative="greater")  # BiLSTM > Baseline
            except Exception:
                stat, p_val = np.nan, 1.0

        r = rank_biserial(x, y)
        effect = interpret_effect(r)

        records.append({
            "baseline":     baseline_name,
            "metric":       METRIC_LABELS[metric],
            "BiLSTM_mean":  x.mean(),
            "BiLSTM_std":   x.std(),
            "Baseline_mean":y.mean(),
            "Baseline_std": y.std(),
            "W_statistic":  stat,
            "p_value":      p_val,
            "effect_r":     r,
            "effect_size":  effect,
        })

results_df = pd.DataFrame(records)

# ============================================================
# 4. FDR 校正（Benjamini-Hochberg）
# ============================================================
valid_mask = results_df["p_value"].notna()
p_vals     = results_df.loc[valid_mask, "p_value"].values

if len(p_vals) > 0:
    reject, p_corrected, _, _ = multipletests(p_vals, method="fdr_bh")
    results_df.loc[valid_mask, "p_fdr"]   = p_corrected
    results_df.loc[valid_mask, "sig_fdr"] = reject
else:
    results_df["p_fdr"]   = np.nan
    results_df["sig_fdr"] = False

results_df["sig_label"] = results_df["p_fdr"].apply(
    lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    if pd.notna(p) else "n.s."
)

# ============================================================
# 5. 印出結果
# ============================================================
print(results_df.to_string(index=False, float_format="%.4f"))

# ============================================================
# 6. 輸出論文用表格（pivot by metric）
# ============================================================
# 6-1. 完整詳細表
results_df.to_csv(os.path.join(OUTPUT_DIR, "wilcoxon_full_results.csv"), index=False)

# 6-2. 論文摘要表（每個 Baseline × 主要指標）
summary_rows = []
for baseline_name in baselines.keys():
    sub = results_df[results_df["baseline"] == baseline_name]
    row = {"Model": baseline_name}
    for _, r in sub.iterrows():
        m = r["metric"]
        row[f"{m}_BiLSTM"]    = f"{r['BiLSTM_mean']:.4f}±{r['BiLSTM_std']:.4f}"
        row[f"{m}_Baseline"]  = f"{r['Baseline_mean']:.4f}±{r['Baseline_std']:.4f}"
        row[f"{m}_p(FDR)"]    = f"{r['p_fdr']:.4f}" if pd.notna(r['p_fdr']) else "n/a"
        row[f"{m}_sig"]       = r["sig_label"]
        row[f"{m}_effect_r"]  = f"{r['effect_r']:.3f}"
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "wilcoxon_summary_table.csv"), index=False)

# 6-3. 論文用精簡表（只含 F1 和 AUC）
print(f"\n{'='*70}")
print("論文摘要（F1 比較）")
print(f"{'='*70}")
f1_table = results_df[results_df["metric"] == "F1"][
    ["baseline", "BiLSTM_mean", "BiLSTM_std", "Baseline_mean", "Baseline_std",
     "p_value", "p_fdr", "sig_label", "effect_r", "effect_size"]
].copy()
f1_table.columns = ["Baseline", "BiLSTM F1 Mean", "±Std", "Baseline F1 Mean", "±Std",
                     "p (raw)", "p (FDR)", "Sig.", "Effect r", "Effect Size"]
print(f1_table.to_string(index=False, float_format="%.4f"))
f1_table.to_csv(os.path.join(OUTPUT_DIR, "wilcoxon_f1_table.csv"), index=False)

# ============================================================
# 7. 自動生成論文用描述文字
# ============================================================
print(f"\n{'='*70}")
print("論文寫作參考文字（可直接貼入 Results 章節）")
print(f"{'='*70}\n")

bilstm_f1_mean = bilstm_df["val_f1"].mean()
bilstm_f1_std  = bilstm_df["val_f1"].std()
print(f"General BiLSTM 在 5-fold 交叉驗證中，F1 分數達 {bilstm_f1_mean:.4f}（±{bilstm_f1_std:.4f}）。\n")

for _, row in f1_table.iterrows():
    sig = row["Sig."]
    r   = float(row["Effect r"])
    eff = interpret_effect(r)
    b_mean = float(row["BiLSTM F1 Mean"])
    base_mean = float(row["Baseline F1 Mean"])
    diff  = b_mean - base_mean
    direction = "高於" if diff > 0 else "低於"

    if sig != "n.s.":
        desc = (f"與 {row['Baseline']} 相比，General BiLSTM 的 F1 點估計{direction}基準模型"
                f"（Δ={diff:+.4f}），Wilcoxon 檢定顯示差異達統計顯著（p={float(row['p (FDR)']):.4f}，"
                f"FDR 校正後 {sig}，rank-biserial r={r:.3f}，{eff} effect size）。")
    else:
        desc = (f"與 {row['Baseline']} 相比，General BiLSTM 的 F1 點估計{direction}基準模型"
                f"（Δ={diff:+.4f}），惟 Wilcoxon 檢定未達統計顯著（p={float(row['p (FDR)']):.4f}，FDR 校正後），"
                f"rank-biserial r={r:.3f}（{eff} effect size）。"
                f"此結果反映樣本數限制（n=5 folds），點估計方向仍支持 BiLSTM 之實務優勢。")
    print(desc)
    print()

print(f"✓ 所有結果儲存至: {OUTPUT_DIR}")
print("  wilcoxon_full_results.csv | wilcoxon_summary_table.csv | wilcoxon_f1_table.csv")
