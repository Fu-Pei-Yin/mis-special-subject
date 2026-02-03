import pandas as pd

# 讀取 CSV 檔案
csv_path = "threads_data_depression.csv"  # 請替換成你的檔案路徑
df = pd.read_csv(csv_path, encoding="utf-8-sig")

# 計算 username 的唯一值數量
unique_user_count = df["username"].nunique()

print(f"username 的唯一值數量：{unique_user_count}")