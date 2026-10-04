"""
黃金標準測試集：七組模型效能指標之分層 Bootstrap 95% 信賴區間
========================================================
用途：回應教授對 4.7 章節之補充要求 ——
      針對 Recall / Precision / F1 / AUC 計算 bootstrap 95% CI，
      讓表１７的點估計值附上不確定性範圍。

方法：分層重抽樣（Stratified Bootstrap）
      - 在 high（真實高風險）與 low（真實低風險）兩個子集內，
        各自獨立做「有放回抽樣」，抽出與原始子集相同的人數，
        再合併成一組模擬樣本。
      - 確保每次重抽樣的 high:low 比例都與原始黃金標準一致
        （BiLSTM 等六模型為 14:97，Gemini 為其可比較子集的比例），
        避免類別比例隨機浮動帶來的額外變異混入信賴區間。
      - 重複 N_BOOTSTRAP 次（預設 2000 次），取每個指標的
        2.5% 與 97.5% 百分位數作為 95% CI 上下界。

輸入檔案（須與本程式放在同一目錄）：
  gold_standard_candidates_identity.csv   人工黃金標籤
  gold_standard_pred_bilstm.csv
  gold_standard_pred_macbert.csv
  gold_standard_pred_tfidf_svm.csv
  gold_standard_pred_tfidf_xgb.csv
  gold_standard_pred_groq_zeroshot.csv
  gold_standard_pred_groq_fewshot.csv
  gold_standard_pred_gemini.csv

輸出：
  table17_with_bootstrap_ci.csv   含點估計值與 95% CI 之完整表格
  bootstrap_distributions.csv     各模型各指標之完整 bootstrap 分布（供檢查用）
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score
)

# ── 設定 ──────────────────────────────────────────────────────────
BASE_DIR = Path("C:\\Users\\eva19\\Desktop\\課程\\專題\\獨立人工仲裁黃金標準")          # 預設讀取程式所在目錄；可改成絕對路徑
N_BOOTSTRAP = 2000            # 重抽樣次數
CI_LOW, CI_HIGH = 2.5, 97.5   # 95% CI 對應的百分位數
RANDOM_SEED = 42              # 固定種子，確保結果可重現


# ── 讀取人工黃金標籤（111 人，evaluation_status == included）──────
def load_gold():
    gs = pd.read_csv(BASE_DIR / "gold_standard_candidates_identity.csv",
                      encoding="utf-8-sig")
    df = gs[gs["evaluation_status"] == "included"].copy()
    df["target"] = df["risk_final"].map({"high": 1, "low": 0})
    df = df[["username", "target"]].drop_duplicates().reset_index(drop=True)
    return df


# ── 分層 Bootstrap 核心函式 ─────────────────────────────────────────
def stratified_bootstrap_ci(y_true, y_pred, y_prob, n_boot=N_BOOTSTRAP,
                             seed=RANDOM_SEED):
    """
    對單一模型的預測結果做分層 bootstrap，回傳四個指標的點估計值、
    95% CI 上下界，以及完整的 bootstrap 分布（供繪圖或檢查用）。

    y_true, y_pred : 0/1 array，人工標籤與模型預測標籤
    y_prob         : 模型輸出機率（用於計算 AUC）
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    # 找出 high（target=1）與 low（target=0）各自的索引位置
    idx_high = np.where(y_true == 1)[0]
    idx_low = np.where(y_true == 0)[0]
    n_high, n_low = len(idx_high), len(idx_low)

    boot_recall, boot_precision, boot_f1, boot_auc = [], [], [], []

    for _ in range(n_boot):
        # 在 high 子集與 low 子集內，各自獨立做「有放回抽樣」
        # 抽出數量與原始子集相同，確保類別比例不變
        resample_high = rng.choice(idx_high, size=n_high, replace=True)
        resample_low = rng.choice(idx_low, size=n_low, replace=True)
        resample_idx = np.concatenate([resample_high, resample_low])

        yt = y_true[resample_idx]
        yp = y_pred[resample_idx]
        ypr = y_prob[resample_idx]

        # 邊界情況：若這次重抽樣後 high 或 low 子集裡的預測標籤
        # 全部相同（例如全部判 0），部分指標會無法定義，
        # 以 zero_division=0 / try-except 安全處理，不中斷迴圈
        boot_recall.append(recall_score(yt, yp, zero_division=0))
        boot_precision.append(precision_score(yt, yp, zero_division=0))
        boot_f1.append(f1_score(yt, yp, zero_division=0))
        try:
            boot_auc.append(roc_auc_score(yt, ypr))
        except ValueError:
            # 該次重抽樣若 yt 全部同一類別，AUC 無法計算，捨棄此次抽樣
            boot_auc.append(np.nan)

    def summarize(arr, point_estimate):
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]  # 排除 AUC 無法計算的少數情況
        lo, hi = np.percentile(arr, [CI_LOW, CI_HIGH])
        return {
            "point": round(point_estimate, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "ci_str": f"{point_estimate:.4f} ({lo:.4f}–{hi:.4f})",
        }

    # 點估計值用「原始未重抽樣」的完整樣本計算，bootstrap 只用於估計不確定性範圍
    point_recall = recall_score(y_true, y_pred, zero_division=0)
    point_precision = precision_score(y_true, y_pred, zero_division=0)
    point_f1 = f1_score(y_true, y_pred, zero_division=0)
    point_auc = roc_auc_score(y_true, y_prob)

    result = {
        "Recall": summarize(boot_recall, point_recall),
        "Precision": summarize(boot_precision, point_precision),
        "F1": summarize(boot_f1, point_f1),
        "AUC": summarize(boot_auc, point_auc),
    }
    distributions = {
        "Recall": boot_recall, "Precision": boot_precision,
        "F1": boot_f1, "AUC": boot_auc,
    }
    return result, distributions


# ── 主流程 ───────────────────────────────────────────────────────────
def main():
    df_gold = load_gold()
    print(f"黃金標準：{len(df_gold)} 人，"
          f"high {df_gold['target'].sum()} / low {(df_gold['target']==0).sum()}\n")

    all_results = []
    all_distributions = {}

    # ── 1. BiLSTM ──────────────────────────────────────────────────
    b = pd.read_csv(BASE_DIR / "gold_standard_pred_bilstm.csv")
    m = df_gold.merge(b, on="username", how="inner")
    res, dist = stratified_bootstrap_ci(
        m["target"], m["bilstm_label"], m["bilstm_prob_mean"])
    all_results.append(("BiLSTM（主模型）", len(m), res))
    all_distributions["BiLSTM"] = dist

    # ── 2. MacBERT ─────────────────────────────────────────────────
    b = pd.read_csv(BASE_DIR / "gold_standard_pred_macbert.csv")
    m = df_gold.merge(b, on="username", how="inner")
    res, dist = stratified_bootstrap_ci(
        m["target"], m["macbert_label"], m["macbert_prob_mean"])
    all_results.append(("MacBERT", len(m), res))
    all_distributions["MacBERT"] = dist

    # ── 3. TF-IDF + SVM ────────────────────────────────────────────
    b = pd.read_csv(BASE_DIR / "gold_standard_pred_tfidf_svm.csv")
    m = df_gold.merge(b, on="username", how="inner")
    res, dist = stratified_bootstrap_ci(
        m["target"], m["svm_label"], m["svm_prob_mean"])
    all_results.append(("TF-IDF+SVM", len(m), res))
    all_distributions["TF-IDF+SVM"] = dist

    # ── 4. TF-IDF + XGBoost ────────────────────────────────────────
    b = pd.read_csv(BASE_DIR / "gold_standard_pred_tfidf_xgb.csv")
    m = df_gold.merge(b, on="username", how="inner")
    res, dist = stratified_bootstrap_ci(
        m["target"], m["xgb_label"], m["xgb_prob_mean"])
    all_results.append(("TF-IDF+XGBoost", len(m), res))
    all_distributions["TF-IDF+XGBoost"] = dist

    # ── 5. Groq Zero-Shot ──────────────────────────────────────────
    b = pd.read_csv(BASE_DIR / "gold_standard_pred_groq_zeroshot.csv")
    m = df_gold.merge(b, on="username", how="inner")
    m["groq_label"] = m["groq_label"].astype(int)
    res, dist = stratified_bootstrap_ci(
        m["target"], m["groq_label"], m["groq_probability"])
    all_results.append(("Groq Zero-Shot", len(m), res))
    all_distributions["Groq Zero-Shot"] = dist

    # ── 6. Groq Few-Shot ───────────────────────────────────────────
    b = pd.read_csv(BASE_DIR / "gold_standard_pred_groq_fewshot.csv")
    m = df_gold.merge(b, on="username", how="inner")
    m["groq_label"] = m["groq_label"].astype(int)
    res, dist = stratified_bootstrap_ci(
        m["target"], m["groq_label"], m["groq_probability"])
    all_results.append(("Groq Few-Shot", len(m), res))
    all_distributions["Groq Few-Shot"] = dist

    # ── 7. Gemini（排除 observe 與 api_failed，僅留可比較樣本）────────
    b = pd.read_csv(BASE_DIR / "gold_standard_pred_gemini.csv")
    m = df_gold.merge(b, on="username", how="inner")
    m = m[~m["gemini_label"].str.contains("observe", na=False, case=False)]
    m = m[m["gemini_label"] != "api_failed"].copy()
    m["pred"] = m["gemini_label"].map({
        "high_depression_risk_user": 1,
        "low_depression_risk_user": 0,
    })
    m = m.dropna(subset=["target", "pred"])
    m["target"] = m["target"].astype(int)
    m["pred"] = m["pred"].astype(int)
    res, dist = stratified_bootstrap_ci(
        m["target"], m["pred"], m["gemini_event_ratio"])
    all_results.append(("Gemini-2.5-flash", len(m), res))
    all_distributions["Gemini-2.5-flash"] = dist

    # ── 整理輸出表格 ─────────────────────────────────────────────────
    rows = []
    for name, n, res in all_results:
        rows.append({
            "Model": name,
            "N": n,
            "Recall (95% CI)": res["Recall"]["ci_str"],
            "Precision (95% CI)": res["Precision"]["ci_str"],
            "F1 (95% CI)": res["F1"]["ci_str"],
            "AUC (95% CI)": res["AUC"]["ci_str"],
        })
        print(f"--- {name} (N={n}) ---")
        for metric in ["Recall", "Precision", "F1", "AUC"]:
            r = res[metric]
            print(f"  {metric:10s} = {r['point']:.4f}  "
                  f"95% CI: [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
        print()

    df_out = pd.DataFrame(rows)
    df_out.to_csv(BASE_DIR / "table17_with_bootstrap_ci.csv",
                  index=False, encoding="utf-8-sig")
    print(f"✅ 已輸出：table17_with_bootstrap_ci.csv")
    print(df_out.to_string(index=False))

    # ── 額外輸出完整 bootstrap 分布（供畫圖或進一步檢查用）─────────────
    dist_rows = []
    for model_name, dist in all_distributions.items():
        for metric, values in dist.items():
            for v in values:
                dist_rows.append({"Model": model_name, "Metric": metric, "Value": v})
    pd.DataFrame(dist_rows).to_csv(
        BASE_DIR / "bootstrap_distributions.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 已輸出：bootstrap_distributions.csv（{len(dist_rows)} 列，供檢查/繪圖用）")


if __name__ == "__main__":
    main()