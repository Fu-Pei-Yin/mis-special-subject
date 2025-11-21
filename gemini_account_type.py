from dotenv import load_dotenv
import os
import pandas as pd
import time
from google import genai
import random

MODEL = "gemini-2.0-flash"

# 載入 API 金鑰
load_dotenv()
API_KEY = os.getenv("STATUS_API_KEY")
client = genai.Client(api_key=API_KEY)

# ==========================================
# 讀取資料
# ==========================================
df = pd.read_csv("posts_count_checked_2_20251117.csv", encoding="utf-8-sig")
df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")

if "gemini_account_type_result" not in df.columns:
    df["gemini_account_type_result"] = ""

# ==========================================
# 判斷任意 14 天內有無 3 篇貼文
# ==========================================
def find_14_days_window(user_df):
    user_df = user_df.sort_values("post_date", ascending=False).reset_index(drop=True)
    dates = user_df["post_date"].tolist()
    for i in range(len(dates)):
        window_start = dates[i]
        window_end = window_start - pd.Timedelta(days=13)
        window_df = user_df[
            (user_df["post_date"] <= window_start) & 
            (user_df["post_date"] >= window_end)
        ]
        if len(window_df) >= 3:
            return window_df  # 回傳整個視窗（不限篇數）
    return None

# ==========================================
# Gemini 的prompt設計
# ==========================================
RULES = """
你必須依照以下條例審查帳號，並判定 gemini_account_type_result。
審查資料包含：
- display_name
- bio
- post_content

================================================
第一階段：帳號身份標註（gemini_account_type_result）
================================================
你需要嚴格依照以下條例逐條判斷：

------------------------------------------------
【1. 企業 / 品牌 / 店家 / 機構主導帳號 → 若符合任一條件 → business】 
------------------------------------------------
1.1 display_name、bio、post_content 內容若明顯屬於企業／品牌／店家／機構／官方。
    包含：店家名稱、地點、營業時間、服務項目、品牌小編等。

1.2 bio或超過50%的post_content以「商品銷售、服務推廣、預約諮詢、徵才」為主

1.3 bio或post_content中出現任一商業行為（合作洽談、預約、訂購）或商業聯絡方式（Email／LINE／電話／連結）

1.4 商業性標籤貼文（hashtag），如：#合作／#贊助／#銷售口號等，出現次數超過貼文總量50%

------------------------------------------------
【2. 公眾人物 / KOL / 網紅 / 個人商業目的帳號 → 若符合任一條件 → business】
------------------------------------------------
2.1 表面為個人帳號，但post_content主要用於商品或服務宣傳
例如：出現次數超過貼文總量50%為商品推廣、團購、置入行銷、品牌曝光、活動宣傳等

2.2 帳號互動模式以粉絲經營、抽獎、合作、求職為主

------------------------------------------------
【3. 特殊排除情況 → 若符合 → non_life】
------------------------------------------------
3.1 若 80% 以上post_content屬於以下類型：
    - 政治相關、社會議題
    - 偶像／運動員／明星／虛擬角色沉迷相關（含：二創文章、應援）
    - 腥羶色或徵友、徵陪伴之文章
    - 與個人生活、個人情緒無關的內容，例如：
        ‧ 心靈成長／勵志文章
        ‧ 職場經驗分享／產業分析
        ‧ 教育／學習資源分享
        ‧ 知識性文章(冷知識、科普等)
        ‧ 書摘／影評心得
        ‧ 各類不含個人生活經驗與情緒抒發的文字創作(與個人生活連結低之內容)

------------------------------------------------
【4. 若不符合以上所有條例 → pass】
------------------------------------------------

================================================
最終輸出規範
================================================
你只能輸出以下三種標籤之一（不得添加任何文字）：
business
non_life
pass
"""

def classify_user(profile_text):
    prompt = f"""
請依照以下完整條例進行帳號審查，並只輸出一個標籤。

條例：
{RULES}

------------------------------------
以下是此使用者的可審查內容：
------------------------------------
{profile_text}
"""
    for retry in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            wait_time = 2 ** retry  # 指數退避
            print(f"!! API錯誤，等待 {wait_time} 秒重試... ({retry+1}/5)")
            time.sleep(wait_time)
    return None  # 若失敗則回 None，稍後補跑

# ==========================================
# 主流程（批次處理，保留完整貼文）
# ==========================================
batch_size = 10
all_users = df["username"].unique()
processed = 0
failed_users = []

for i in range(0, len(all_users), batch_size):
    batch_users = all_users[i:i+batch_size]
    print(f"\n===== 處理 batch {i//batch_size + 1} ({len(batch_users)} users) =====\n")

    for user in batch_users:
        user_df = df[df["username"] == user]
        print(f"處理使用者：{user}")

        existing = user_df["gemini_account_type_result"].iloc[0]
        if isinstance(existing, str) and existing.strip() != "":
            print(f"→ gemini_account_type_result 已存在 ({existing})，略過 API")
            continue

        window = find_14_days_window(user_df)
        if window is None:
            df.loc[df["username"] == user, "gemini_account_type_result"] = "not_enough_posts"
            print("→ 任意14天不足3篇，略過 API")
            continue

        # 保留完整貼文
        posts_text = "\n---\n".join(window["post_content"].astype(str).tolist())

        display_name = str(user_df["display_name"].iloc[0])
        bio = str(user_df["bio"].iloc[0])

        merged_text = f"""
Display Name: {display_name}
Bio: {bio}

符合條件的 14 天貼文視窗（共 {len(window)} 篇）：
{posts_text}
"""

        # 呼叫 Gemini API
        result = classify_user(merged_text)
        if result is None:
            failed_users.append(user)
            print("→ API_error，稍後補跑")
        else:
            df.loc[df["username"] == user, "gemini_account_type_result"] = result
            print(f"→ Gemini 判斷：{result}")

        processed += 1
        time.sleep(random.uniform(2, 3))

    # 每 batch 結束後休息 5~10 秒
    batch_sleep = random.uniform(5, 10)
    print(f"△ Batch 完成，休息 {batch_sleep:.1f} 秒...")
    time.sleep(batch_sleep)

# ==========================================
# 補跑 API_error 的使用者
# ==========================================
print(f"\n===== 補跑失敗的 {len(failed_users)} users =====\n")
for user in failed_users:
    user_df = df[df["username"] == user]
    window = find_14_days_window(user_df)
    if window is None:
        df.loc[df["username"] == user, "gemini_account_type_result"] = "not_enough_posts"
        continue

    posts_text = "\n---\n".join(window["post_content"].astype(str).tolist())
    display_name = str(user_df["display_name"].iloc[0])
    bio = str(user_df["bio"].iloc[0])
    merged_text = f"""
Display Name: {display_name}
Bio: {bio}

符合條件的 14 天貼文視窗（共 {len(window)} 篇）：
{posts_text}
"""
    result = classify_user(merged_text)
    if result is None:
        df.loc[df["username"] == user, "gemini_account_type_result"] = "API_error"
    else:
        df.loc[df["username"] == user, "gemini_account_type_result"] = result
    time.sleep(random.uniform(2, 3))

# ==========================================
# 最終輸出
# ==========================================
df.to_csv("account_type_result_2_20251117.csv", index=False, encoding="utf-8-sig")
print("完成！結果已輸出到 account_type_result_2_20251117.csv")