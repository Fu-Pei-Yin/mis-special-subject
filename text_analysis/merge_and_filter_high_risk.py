# 02_merge_and_filter_high_risk.py
import pandas as pd
import glob
import os

DATA_DIR = "depression_data"
OUTPUT_PATH = "high_risk_posts.csv"

# ==============================
# 讀取所有 CSV
# ==============================
all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
if not all_files:
    raise RuntimeError("找不到任何 CSV 檔案")

dfs = []
for f in all_files:
    df = pd.read_csv(f, encoding="utf-8-sig")
    df["source_file"] = os.path.basename(f)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# ==============================
# 基本清理
# ==============================
df_all["username"] = df_all["username"].astype(str).str.strip()

LABEL_COLS = [
    "gemini_user_status_label",
    "human_user_status_label"
]
for col in LABEL_COLS:
    if col not in df_all.columns:
        df_all[col] = ""
    df_all[col] = df_all[col].astype(str).str.strip().str.lower()

# ==============================
# 建立 final_depression_label
# ==============================
def resolve_final_label(row):
    if row["gemini_user_status_label"]:
        return row["gemini_user_status_label"]
    return row["human_user_status_label"]

df_all["final_depression_label"] = df_all.apply(resolve_final_label, axis=1)

# ==============================
# 篩選高憂鬱風險
# ==============================
high_risk_df = df_all[
    df_all["final_depression_label"] == "high_depression_risk_user"
].copy()

print(f"高憂鬱風險貼文數：{len(high_risk_df)}")
print(f"高憂鬱風險使用者數：{high_risk_df['username'].nunique()}")

# ==============================
# 僅保留分析必要欄位
# ==============================
KEEP_COLS = [
    "username",
    "post_date",
    "post_content",
    "bio",
    "matched_keyword"
]

high_risk_df = high_risk_df[KEEP_COLS]

# ==============================
# 輸出
# ==============================
high_risk_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print(f"已輸出分析檔案：{OUTPUT_PATH}")