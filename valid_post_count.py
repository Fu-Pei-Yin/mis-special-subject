import pandas as pd

# ==========================================
# 判斷任意連續 14 天內是否有 >= 3 篇貼文
# ==========================================
def has_three_posts_within_14_days(user_df):
    # 轉成 datetime，並去除無效值
    user_df = user_df.copy()
    user_df["post_date"] = pd.to_datetime(user_df["post_date"], errors="coerce")
    user_df = user_df.dropna(subset=["post_date"])
    
    # 排序，從最新到最舊
    user_df = user_df.sort_values("post_date", ascending=False).reset_index(drop=True)
    
    dates = user_df["post_date"].tolist()
    
    # 至少要三篇貼文才能檢查
    if len(dates) < 3:
        return False
    
    # 滑動 14 天窗口檢查
    for i in range(len(dates)):
        window_start = dates[i]
        window_end = window_start - pd.Timedelta(days=13)
        # 取出該 14 天期間內的所有貼文
        window_df = [d for d in dates if window_end <= d <= window_start]
        if len(window_df) >= 3:
            return True
    
    return False

# ==========================================
# 對整個 DataFrame 標註 not_enough_posts
# ==========================================
def mark_not_enough_posts(df):
    # 初始化欄位
    if "gemini_account_type_result" not in df.columns:
        df["gemini_account_type_result"] = ""
    
    for user in df["username"].unique():
        user_df = df[df["username"] == user]
        if not has_three_posts_within_14_days(user_df):
            df.loc[df["username"] == user, "gemini_account_type_result"] = "not_enough_posts"
    return df

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    df = pd.read_csv("threads_data_zh_2_20251117.csv", encoding="utf-8-sig")
    
    df = mark_not_enough_posts(df)
    
    df.to_csv("posts_count_checked_2_20251117.csv", index=False, encoding="utf-8-sig")
    print("任意14天貼文檢查完成")