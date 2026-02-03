import pandas as pd

# ==============================
# 讀取資料
# ==============================
CSV_PATH = "user_status_label_depression.csv"
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

# 必要欄位檢查
REQUIRED_COLUMNS = [
    "username",
    "gemini_account_type_result",
    "gemini_user_status_label",
    "human_account_type_result",
    "human_user_status_label"
]
for col in REQUIRED_COLUMNS:
    if col not in df.columns:
        raise ValueError(f"缺少必要欄位：{col}")

# 欄位清理
df["username"] = df["username"].astype(str).str.strip()
for col in ["gemini_account_type_result", "gemini_user_status_label",
            "human_account_type_result", "human_user_status_label"]:
    df[col] = df[col].astype(str).str.strip().str.lower()

# ==============================
# 整體唯一使用者數
# ==============================
total_unique_users = df["username"].nunique()

# ==============================
# 最終帳號身分（包含人工審查）
# ==============================
def final_account_type(row):
    if row["gemini_account_type_result"] != "needs_manual_review":
        return row["gemini_account_type_result"]
    else:
        return row["human_account_type_result"]

df["final_account_type"] = df.apply(final_account_type, axis=1)

ACCOUNT_TYPE_CATEGORIES = [
    "not_enough_posts",
    "business",
    "non_life",
    "pass",
    "needs_manual_review",
]

account_type_user_counts = (
    df.groupby("final_account_type")["username"]
    .nunique()
    .reindex(ACCOUNT_TYPE_CATEGORIES, fill_value=0)
)

# ==============================
# 最終憂鬱風險（包含人工審查）
# ==============================
def final_depression_label(row):
    if row["gemini_user_status_label"] != "needs_manual_review":
        return row["gemini_user_status_label"]
    else:
        return row["human_user_status_label"]

df["final_depression_label"] = df.apply(final_depression_label, axis=1)

DEPRESSION_RISK_CATEGORIES = [
    "high_depression_risk_user",
    "low_depression_risk_user",
    "non_depression_risk_user",
    "needs_manual_review"
]

depression_risk_user_counts = (
    df.groupby("final_depression_label")["username"]
    .nunique()
    .reindex(DEPRESSION_RISK_CATEGORIES, fill_value=0)
)

# ==============================
# Gemini API 覆蓋率計算 (以唯一 user 為單位)
# ==============================
gemini_account_api_count = df.groupby("username")["gemini_account_type_result"].first().ne("needs_manual_review").sum()
gemini_depression_api_count = df.groupby("username")["gemini_user_status_label"].first().ne("needs_manual_review").sum()

gemini_account_coverage = gemini_account_api_count / total_unique_users
gemini_depression_coverage = gemini_depression_api_count / total_unique_users

# ==============================
# 輸出結果
# ==============================
print("\n" + "=" * 70)
print("【整體唯一使用者數】")
print("=" * 70)
print(f"Unique users (by username): {total_unique_users}")

print("\n" + "=" * 70)
print("【最終帳號身分（包含人工審查）】")
print("=" * 70)
for k, v in account_type_user_counts.items():
    print(f"{k:25s} : {v}")
print(f"Gemini API 覆蓋率：{gemini_account_coverage:.1%}")

print("\n" + "=" * 70)
print("【最終憂鬱風險（包含人工審查）】")
print("=" * 70)
for k, v in depression_risk_user_counts.items():
    print(f"{k:25s} : {v}")
print(f"Gemini API 覆蓋率：{gemini_depression_coverage:.1%}")
