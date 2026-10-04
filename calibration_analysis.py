# ============================================================
# 機率校準與閾值敏感度分析
# 輸入：5個 fold_X_predictions.csv（含 y_true, y_pred, probability）
# 輸出：Brier Score, ECE, Reliability Diagram, 閾值敏感度表
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from sklearn.metrics import (brier_score_loss, f1_score, accuracy_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.calibration import calibration_curve

# -----------------------
# Config ← 修改路徑
# -----------------------
PRED_DIR   = "C:/Users/USER/Desktop/課程/專題/模型訓練/results_D_general_bilstm"
OUTPUT_DIR = "C:/Users/USER/Desktop/課程/專題/模型訓練/results_calibration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FOLDS    = 5
N_BINS     = 10       # Reliability Diagram 分組數
THRESHOLDS = np.arange(0.3, 0.75, 0.05).round(2)   # 0.30 ~ 0.70

# ============================================================
# 1. 載入所有 fold 預測結果
# ============================================================
all_dfs = []
for fold in range(1, N_FOLDS + 1):
    path = os.path.join(PRED_DIR, f"fold_{fold}_predictions.csv")
    df   = pd.read_csv(path)
    df["fold"] = fold
    all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)
y_true_all = combined["y_true"].values
y_prob_all = combined["probability"].values
y_pred_all = combined["y_pred"].values

print(f"Total samples: {len(combined)}")
print(f"Positive rate: {y_true_all.mean():.4f}")

# ============================================================
# 2. Brier Score（越低越好，完美=0，隨機~0.25）
# ============================================================
brier = brier_score_loss(y_true_all, y_prob_all)
print(f"\nBrier Score: {brier:.4f}")

# Per-fold Brier Score
fold_brier = []
for fold in range(1, N_FOLDS + 1):
    sub  = combined[combined["fold"] == fold]
    bs   = brier_score_loss(sub["y_true"].values, sub["probability"].values)
    fold_brier.append({"fold": fold, "brier_score": bs})
    print(f"  Fold {fold} Brier Score: {bs:.4f}")

fold_brier_df = pd.DataFrame(fold_brier)
print(f"  Mean ± Std: {fold_brier_df['brier_score'].mean():.4f} ± {fold_brier_df['brier_score'].std():.4f}")

# ============================================================
# 3. ECE（Expected Calibration Error）
# ============================================================
def compute_ece(y_true, y_prob, n_bins=10):
    bins      = np.linspace(0, 1, n_bins + 1)
    ece       = 0.0
    bin_stats = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            bin_stats.append({"bin_lower": bins[i], "bin_upper": bins[i+1],
                              "count": 0, "avg_confidence": 0, "avg_accuracy": 0})
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc  = y_true[mask].mean()
        ece     += (mask.sum() / len(y_true)) * abs(avg_conf - avg_acc)
        bin_stats.append({"bin_lower": round(bins[i], 2), "bin_upper": round(bins[i+1], 2),
                          "count": int(mask.sum()),
                          "avg_confidence": round(avg_conf, 4),
                          "avg_accuracy":   round(avg_acc,  4),
                          "gap": round(abs(avg_conf - avg_acc), 4)})
    return ece, pd.DataFrame(bin_stats)

ece, bin_df = compute_ece(y_true_all, y_prob_all, N_BINS)
print(f"\nECE (Expected Calibration Error): {ece:.4f}")
print(bin_df.to_string(index=False))
bin_df.to_csv(os.path.join(OUTPUT_DIR, "ece_bin_stats.csv"), index=False)

# ============================================================
# 4. Reliability Diagram
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左：整體 Reliability Diagram
ax = axes[0]
fraction_pos, mean_pred = calibration_curve(y_true_all, y_prob_all, n_bins=N_BINS, strategy="uniform")
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)
ax.plot(mean_pred, fraction_pos, "o-", color="#2563EB", linewidth=2,
        markersize=7, label=f"General BiLSTM\n(ECE={ece:.4f}, Brier={brier:.4f})")
ax.fill_between(mean_pred, fraction_pos, mean_pred,
                alpha=0.15, color="#2563EB", label="Calibration gap")
ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Fraction of Positives", fontsize=12)
ax.set_title("Reliability Diagram (Overall)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(alpha=0.3)

# 右：每個 fold 的 calibration curve
ax2 = axes[1]
ax2.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
colors = ["#2563EB","#16A34A","#DC2626","#D97706","#7C3AED"]
for fold in range(1, N_FOLDS + 1):
    sub = combined[combined["fold"] == fold]
    fp, mp = calibration_curve(sub["y_true"].values, sub["probability"].values,
                               n_bins=N_BINS, strategy="uniform")
    ax2.plot(mp, fp, "o-", color=colors[fold-1], linewidth=1.5,
             markersize=5, label=f"Fold {fold}", alpha=0.8)
ax2.set_xlabel("Mean Predicted Probability", fontsize=12)
ax2.set_ylabel("Fraction of Positives", fontsize=12)
ax2.set_title("Reliability Diagram (Per Fold)", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9)
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.grid(alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "reliability_diagram.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ Reliability Diagram saved: {fig_path}")

# ============================================================
# 5. 閾值敏感度分析（0.30 ~ 0.70）
# ============================================================
print(f"\n{'='*70}\n閾值敏感度分析\n{'='*70}")
threshold_records = []

for thresh in THRESHOLDS:
    y_pred_t = (y_prob_all >= thresh).astype(int)
    acc  = accuracy_score(y_true_all, y_pred_t)
    f1   = f1_score(y_true_all, y_pred_t, zero_division=0)
    prec = precision_score(y_true_all, y_pred_t, zero_division=0)
    rec  = recall_score(y_true_all, y_pred_t, zero_division=0)
    try:
        auc = roc_auc_score(y_true_all, y_prob_all)
    except:
        auc = float("nan")
    tp = int(((y_pred_t == 1) & (y_true_all == 1)).sum())
    fp = int(((y_pred_t == 1) & (y_true_all == 0)).sum())
    fn = int(((y_pred_t == 0) & (y_true_all == 1)).sum())
    tn = int(((y_pred_t == 0) & (y_true_all == 0)).sum())

    threshold_records.append({
        "threshold": thresh, "accuracy": acc, "f1": f1,
        "precision": prec, "recall": rec, "auc": auc,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn
    })
    print(f"  θ={thresh:.2f} | Acc={acc:.4f} F1={f1:.4f} P={prec:.4f} R={rec:.4f} "
          f"| TP={tp} FP={fp} FN={fn} TN={tn}")

thresh_df = pd.DataFrame(threshold_records)
thresh_df.to_csv(os.path.join(OUTPUT_DIR, "threshold_sensitivity.csv"), index=False)

# 閾值敏感度圖
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresh_df["threshold"], thresh_df["f1"],       "o-", color="#2563EB", linewidth=2, label="F1")
ax.plot(thresh_df["threshold"], thresh_df["precision"],"s-", color="#16A34A", linewidth=2, label="Precision")
ax.plot(thresh_df["threshold"], thresh_df["recall"],   "^-", color="#DC2626", linewidth=2, label="Recall")
ax.plot(thresh_df["threshold"], thresh_df["accuracy"], "D-", color="#D97706", linewidth=2, label="Accuracy")
ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1.2, label="Default θ=0.5")
best_f1_thresh = thresh_df.loc[thresh_df["f1"].idxmax(), "threshold"]
ax.axvline(x=best_f1_thresh, color="#7C3AED", linestyle=":", linewidth=1.5,
           label=f"Best F1 θ={best_f1_thresh:.2f}")
ax.set_xlabel("Classification Threshold (θ)", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Threshold Sensitivity Analysis", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
plt.tight_layout()
thresh_fig_path = os.path.join(OUTPUT_DIR, "threshold_sensitivity.png")
plt.savefig(thresh_fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ Threshold Sensitivity Plot saved: {thresh_fig_path}")

# ============================================================
# 6. 完整摘要 JSON
# ============================================================
summary = {
    "brier_score":        round(float(brier), 4),
    "brier_mean_per_fold":round(float(fold_brier_df["brier_score"].mean()), 4),
    "brier_std_per_fold": round(float(fold_brier_df["brier_score"].std()),  4),
    "ece":                round(float(ece), 4),
    "best_f1_threshold":  float(best_f1_thresh),
    "default_threshold":  0.5,
    "threshold_sensitivity": threshold_records,
}
import json
with open(os.path.join(OUTPUT_DIR, "calibration_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n{'='*70}")
print(f"校準分析摘要")
print(f"{'='*70}")
print(f"  Brier Score       : {brier:.4f}  (越低越好，隨機分類器 ≈ 0.25)")
print(f"  ECE               : {ece:.4f}  (越低越好，完美校準 = 0)")
print(f"  Best F1 Threshold : {best_f1_thresh:.2f}")
print(f"\n✓ 所有結果儲存至: {OUTPUT_DIR}")
print("  reliability_diagram.png | threshold_sensitivity.png")
print("  ece_bin_stats.csv | threshold_sensitivity.csv | calibration_summary.json")
