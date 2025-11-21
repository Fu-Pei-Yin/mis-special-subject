from google import genai
from datetime import timedelta
import pandas as pd
from dotenv import load_dotenv
import os
import json
import time
import random

MODEL = "gemini-2.0-flash"

# ==========================================
# 1. 初始化 Gemini API
# ==========================================
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

# ==========================================
# 2. 情緒prompt：Depressive / Other Negative / General
# ==========================================
FULL_RULES = r"""
你是一個 Threads 貼文的「情緒傾向審查模型」，必須嚴格依以下條列進行分類，不得推論未提供內容。

審查原則：
- 僅根據觀察視窗內的貼文內容與出現頻率判斷。
- 出現心理諮商、醫療、疾病相關字詞 **若未達頻率標準，不可直接判為憂鬱傾向**。

=====================================================
【子階段一：「疑似有憂鬱傾向的用戶」（depressive_user）】
需符合以下 **任一條件**，且該條件在觀察視窗內：**兩週內（14天）出現 ≥3 則** 或 **占貼文數 ≥ 50%**

1. 負面身心狀況與語言特徵
1.1 強烈負面情緒詞：悲傷、無助、沮喪、痛苦、焦慮、孤獨、絕望、崩潰等  
     例：「撐不住」「活著沒意思」「很痛苦」「崩潰」「徹底失望」
1.2 生理症狀：哭泣、失眠、早醒、疲倦、沒食慾、注意力不集中、記憶力下降等  
     例：「哭到睡不著」「沒胃口」「累到沒力氣」「腦袋停不下來」
1.3 自責與價值感低落  
     例：「我很沒用」「總是辜負大家」「沒資格存在」
1.4 興趣/動機喪失  
     例：「什麼都沒意思」「提不起勁」「以前喜歡現在完全沒感覺」

2. 社交退縮與孤立
2.1 不願社交或感覺被忽略：  
     例：「不想見人」「沒朋友」「好像沒人找我聊天了」
2.2 以「自己 vs 過去」做負面比較  
     例：「以前很樂觀，現在只想消失」「最近越來越不想發文」

3. 明顯求助/自傷/自殺意圖
3.1 自傷、自殺意圖：  
     例：「想消失」「不想活」「活著太痛苦」「想結束一切」
3.2 主動表達想尋求諮商、身心科治療  
3.3 坦述被診斷憂鬱症/正在接受治療  
**若僅符合 3.2、3.3，但未達 1.X~3.1 任一條之頻率，應排除，不列為 depressive_user。**

=====================================================
【子階段二：「非憂鬱但有負面情緒」（other_negative_user）】
符合以下條件：
- 負面情緒存在（例如孤單、焦慮、憤怒、傷心、害怕、壓力）
- 但 **未達子階段一任何條例之頻率標準**
- 例：「有點孤單」「壓力有點大」「不太想聊天但還好」

=====================================================
【子階段三：「一般狀態用戶」（general_user）】
符合以下條件：
- 不含明顯憂鬱或強烈負面症狀
- 內容主要為日常生活、分享、聊天、中性描述或正向主題
- 若有輕微負面情緒，僅為偶發且不持續

=====================================================
【只能輸出以下三種之一】（不得加入解釋、理由或其他文字）
"depressive_user"
"other_negative_user"
"general_user"
=====================================================
"""

# ==========================================
# 3. Prompt 模板（禁止隨意解釋）
# ==========================================
STATUS_PROMPT = """
你是一個不可以自行推論的 Threads 貼文審查模型，只能依照條列規則判斷。

===【審查規範】===
{status_rules}

===【14天觀察視窗內貼文（日期→內容）】===
{posts}

請嚴格依規範判定，用 JSON 格式輸出（不能加入其他文字）：
{{
  "gemini_user_status_label": "<只能三選一>"
}}
"""

# ==========================================
# 4. 載入資料
# ==========================================
df = pd.read_csv("account_type_result_1_20251112.csv", encoding="utf-8-sig")
df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")

if "gemini_user_status_label" not in df.columns:
    df["gemini_user_status_label"] = ""

# ==========================================
# 5. 續跑機制
# ==========================================
partial_map = {}
if os.path.exists("user_status_result_partial.csv"):
    partial_df = pd.read_csv("user_status_result_partial.csv", encoding="utf-8-sig")
    for _, row in partial_df.iterrows():
        partial_map[row["username"]] = row["gemini_user_status_label"]

# ==========================================
# 6. Gemini API 呼叫
# ==========================================
def classify_user_one_shot(user_posts):
    posts_text = "\n".join(
        [f"{row.post_date.date()}｜{row.post_content}" for _, row in user_posts.iterrows()]
    )
    prompt = STATUS_PROMPT.format(status_rules=FULL_RULES, posts=posts_text)

    for retry in range(5):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            resp_text = getattr(response, "text", "").strip()
            if not resp_text:
                raise ValueError("Empty API Response")
            cleaned = resp_text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except Exception as e:
            wait_time = 2 ** retry + random.uniform(0, 1)
            print(f"API error, 等待 {wait_time:.1f} 秒重試... ({retry+1}/5)")
            time.sleep(wait_time)
            if "429" in str(e).lower():
                return {"gemini_user_status_label": "quota_exceeded"}
    return {"gemini_user_status_label": "error"}

# ==========================================
# 7. 批次處理 + 補跑失敗 user
# ==========================================
batch_size = 5
save_interval = 20
processed_count = 0
all_users = df["username"].unique()
failed_users = []

for i in range(0, len(all_users), batch_size):
    batch_users = all_users[i:i+batch_size]
    print(f"\n===== 處理 batch {i//batch_size+1} ({len(batch_users)} users) =====\n")

    for user in batch_users:
        user_rows = df[df["username"] == user]
        print(f"處理使用者：{user}")

        # 非 pass 身分類型不處理
        acc_type = str(user_rows.iloc[0]["gemini_account_type_result"]).strip().lower()
        if acc_type != "pass":
            df.loc[df["username"] == user, "gemini_user_status_label"] = "skipped"
            continue

        # 續跑模式
        if user in partial_map:
            df.loc[df["username"] == user, "gemini_user_status_label"] = partial_map[user]
            continue

        # 已有結果
        current_status = str(user_rows.iloc[0]["gemini_user_status_label"])
        if current_status not in ["", "nan", None]:
            continue

        # 取最新貼文→往前推 14 天
        latest_post_date = user_rows["post_date"].max()
        window_start = latest_post_date - timedelta(days=14)
        window_posts = user_rows[user_rows["post_date"].between(window_start, latest_post_date)]

        if len(window_posts) < 3:
            df.loc[df["username"] == user, "gemini_user_status_label"] = "general_user"
            continue

        # 呼叫 Gemini API
        result = classify_user_one_shot(window_posts)
        label = result.get("gemini_user_status_label", "error")

        if label in ["error", "quota_exceeded"]:
            failed_users.append(user)
            print("→ API_error，稍後補跑")
        else:
            df.loc[df["username"] == user, "gemini_user_status_label"] = label
            print(f"→ Gemini 判斷：{result}")

        processed_count += 1
        if processed_count % save_interval == 0:
            df.to_csv("user_status_result_partial.csv", index=False, encoding="utf-8-sig")

        # 每個 user 間隔
        time.sleep(random.uniform(2, 3))

    # 每 batch 結束後休息
    batch_sleep = random.uniform(5, 10)
    print(f"△ Batch 完成，休息 {batch_sleep:.1f} 秒...")
    time.sleep(batch_sleep)

# 補跑失敗 user
print(f"\n===== 補跑失敗的 {len(failed_users)} users =====\n")
for user in failed_users:
    user_rows = df[df["username"] == user]
    latest_post_date = user_rows["post_date"].max()
    window_start = latest_post_date - timedelta(days=14)
    window_posts = user_rows[user_rows["post_date"].between(window_start, latest_post_date)]
    if len(window_posts) < 3:
        df.loc[df["username"] == user, "gemini_user_status_label"] = "general_user"
        continue

    result = classify_user_one_shot(window_posts)
    label = result.get("gemini_user_status_label", "error")
    df.loc[df["username"] == user, "gemini_user_status_label"] = label
    time.sleep(random.uniform(2, 3))

# ==========================================
# 8. 最終輸出
# ==========================================
df.to_csv("user_status_result_1_20251112.csv", index=False, encoding="utf-8-sig")
print("完成！")