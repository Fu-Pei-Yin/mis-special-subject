from dotenv import load_dotenv
import os
import pandas as pd
import time
from google import genai
import random

# ==========================================
# API 設定
# ==========================================
MODEL = "gemini-2.0-flash"

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

# ==========================================
# 載入資料、初始化欄位
# ==========================================
df = pd.read_csv("threads_data_depression.csv", encoding="utf-8-sig") #若為重跑同csv，將檔名改為與結果相同之檔名
df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")

string_columns = ["gemini_account_type_result"]

for col in string_columns:
    if col not in df.columns:
        df[col] = pd.Series(dtype="string")
    else:
        df[col] = df[col].astype("string")

# ==========================================
# Prompt Rules
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
2.1 表面為個人帳號，但post_content主要用於介紹商品或服務宣傳
例如：出現次數超過貼文總量50%為商品推廣、團購、置入行銷、品牌曝光、活動宣傳等

2.2 帳號互動模式以粉絲經營、抽獎、合作、求職為主

------------------------------------------------
【3. 特殊排除情況 → 若符合 → non_life】
------------------------------------------------
3.1 若 80% 以上post_content屬於以下類型：
    - 政治相關、社會議題
    - 沉迷偶像／運動員／明星／虛擬角色（含：二創文章、應援等）
    - 性暗示或徵友、徵陪伴之文章
    - 與個人生活或個人情緒無關的內容，例如：
        ‧ 心靈成長／勵志文章
        ‧ 職場經驗分享
        ‧ 產業分析(如:股票、加密貨幣等)
        ‧ 學習資源分享(如:語言學習、技能提升等)
        ‧ 知識性分享(如:科普、學術名詞介紹等)
        ‧ 看書／電影心得
        ‧ 各類不含個人生活與情緒抒發的文字創作(詩歌、小說、音樂推薦等)

------------------------------------------------
【4. 若不符合以上所有條列 → pass】
------------------------------------------------

================================================
最終輸出規範
================================================
你只能輸出以下三種標籤之一（不得添加任何文字）：
business
non_life
pass
"""

# 組合送給API之函式
def build_profile_text(user_df):
    posts_text = "\n---\n".join(user_df["post_content"].astype(str).tolist())
    display_name = str(user_df["display_name"].iloc[0])
    bio = str(user_df["bio"].iloc[0])
    return f"""
Display Name: {display_name}
Bio: {bio}

14 天內貼文（共 {len(user_df)} 篇）：
{posts_text}
"""

# 呼叫API之函式
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
        except Exception:
            wait_time = 2 ** retry  # 指數退避
            print(f"!! API錯誤，等待 {wait_time} 秒重試... ({retry+1}/5)")
            time.sleep(wait_time)
    return None  # 若五次仍失敗 → 回 None

# ==========================================
# 主流程（批次處理）
# ==========================================
batch_size = 10
# 移除無效username
df = df.dropna(subset=["username"])
df = df[df["username"].astype(str).str.strip() != ""]
df["username"] = df["username"].astype(str)
all_users = df["username"].unique()
processed = 0
failed_users = []

for i in range(0, len(all_users), batch_size):
    batch_users = all_users[i:i+batch_size]
    print(f"\n===== 處理 batch {i//batch_size + 1} ({len(batch_users)} users) =====\n")

    for user in batch_users:
        user_df = df[df["username"] == user]
        print(f"處理使用者：{user}")

        if user_df.empty:
            print(f"→ 無資料 username：{user} ，跳過")
            continue

        # 已標註過 → 跳過(補跑)
        existing = user_df["gemini_account_type_result"].iloc[0]
        if isinstance(existing, str) and existing.strip() != "":
            print(f"→ gemini_account_type_result 已標註 ({existing})，略過 API")
            continue

        # 貼文數不足3篇 → 標註not_enough_posts，跳過 API
        if len(user_df) < 3:
                df.loc[df["username"] == user, "gemini_account_type_result"] = "not_enough_posts"
                print("→ 貼文不足 3 篇，標註 not_enough_posts，略過 API")
                continue

        # 組合使用者資料
        merged_text = build_profile_text(user_df)
        # 呼叫Gemini API
        result = classify_user(merged_text)
        if result is None:
            failed_users.append(user)
            print("→ API_error，稍後補跑")
        else:
            df.loc[df["username"] == user, "gemini_account_type_result"] = result
            print(f"→ Gemini 判斷：{result}")

        processed += 1
        time.sleep(random.uniform(2, 3))

    df.to_csv("account_type_result_depression.csv", index=False, encoding="utf-8-sig")

    if i + batch_size < len(all_users):
        batch_sleep = random.uniform(5, 10)
        print(f"△ Batch 完成，休息 {batch_sleep:.1f} 秒...")
        time.sleep(batch_sleep)
    else:
        print("△ 最後一個 batch 完成！")

# ==========================================
# 補跑API_error的使用者
# ==========================================
if failed_users:
    print(f"\n===== 補跑 {len(failed_users)} 個 API error 使用者 =====\n")
    for user in failed_users:
        user_df = df[df["username"] == user]
        print(f"補跑使用者：{user}")

        # 組合使用者資料
        merged_text = build_profile_text(user_df)
        # 呼叫Gemini API
        result = classify_user(merged_text)

        if result is not None:
            df.loc[df["username"] == user, "gemini_account_type_result"] = result
            print(f"→ 補跑成功：{result}")
        else:
            df.loc[df["username"] == user, "gemini_account_type_result"] = "API_error"
            print("→ 補跑仍失敗，標記為 API_error")
            
        time.sleep(random.uniform(1.5, 3))
        
    df.to_csv("account_type_result_depression.csv", index=False, encoding="utf-8-sig")
else:
    print("無需補跑API error之使用者")

print("完成！已進行帳號審查標記")