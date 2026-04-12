import pandas as pd
import os

# 原始資料檔案
INPUT_FILE = "user_status_label_normal.csv"

# 要輸出的無風險名單
OUTPUT_FILE = "non_depression_risk_user_list.csv"

def main():
    # 讀取原始資料
    df = pd.read_csv(INPUT_FILE)

    # 保留無風險使用者的資料
    df_non = df[(df["gemini_user_status_label"] == "non_depression_risk_user") |
        (
            (df["gemini_user_status_label"] == "needs_manual_review") &
            (df["human_user_status_label"] == "non_depression_risk_user")
        )]

    if df_non.empty:
        print("沒有任何被標註為 non_depression_risk_user 的使用者")
        return

    # 取得每位使用者的第一筆資料(最新/日期最晚的貼文)
    df_latest_post = (
        df_non.sort_values(by=["username", "post_date"])
            .groupby("username", as_index=False)
            .last()
    )

    # 並保留指定欄位
    new_df = df_latest_post[
        ["username", "display_name", "bio", "post_date"]
    ].copy()

    # 若 OUTPUT_FILE 已存在 → 讀舊資料，並避免重複紀錄
    if os.path.exists(OUTPUT_FILE):
        old_df = pd.read_csv(OUTPUT_FILE)[new_df.columns]

        # 避免重複寫入同樣的 username
        new_df = new_df[~new_df["username"].isin(old_df["username"])]

        # 若沒有新的需要寫入的資料
        if new_df.empty:
            print("所有 non_depression_risk_user 都已存在於名單中，無新增記錄。")
            return

        # 合併舊資料 + 新資料
        final_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        # 第一次建立檔案
        final_df = new_df

    # ================================
    # 寫入輸出檔案
    # ================================
    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"完成！已新增 {len(new_df)} 位 non_depression_risk_user 至名單中。")
    print(f"目前名單總共 {len(final_df)} 位使用者。")

if __name__ == "__main__":
    main()