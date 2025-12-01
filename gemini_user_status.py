from google import genai
import pandas as pd
from dotenv import load_dotenv
import os
import json
import time
import random

MODEL = "gemini-2.5-flash"

# ==========================================
# 初始化Gemini API
# ==========================================
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

# ==========================================
# 載入資料
# ==========================================
df = pd.read_csv("account_type_result_depression.csv", encoding="utf-8-sig")
df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")

# ==========================================
# 初始化欄位格式
# ==========================================
string_columns = ["gemini_user_status_label", "gemini_symptom_counts"]

for col in string_columns:
    if col not in df.columns:
        df[col] = pd.Series(dtype="string")
    else:
        df[col] = df[col].astype("string")

# ==========================================
# PHQ-9症狀判定規則
# ==========================================
SYMPTOM_DETECTION_RULES = r"""
你是一個專業的 Threads 用戶情緒狀態判定模型。

【任務】
分析使用者14天內貼文內容，判定是否具有憂鬱傾向事件。僅標記使用者「自身表達的主觀感受」，不包含引用、創作、他人故事。

【判斷邏輯】
- 同一貼文，只要符合任一症狀，即計為1個憂鬱事件。
- 但仍需輸出各症狀的次數（研究分析用）。

【症狀分類】
1. 失去興趣/缺乏動機
   - 範例：「提不起勁」「什麼都沒意思」「以前喜歡現在不想做」
2. 明顯負向情緒
   - 包含悲傷、無助、沮喪、痛苦、崩潰、焦慮、絕望
   - 範例：「每天都快撐不下去」「一直想哭」「覺得人生沒希望」「情緒很糟」
3. 睡眠困擾
   - 失眠、早醒、過度睡眠
   - 範例：「整晚睡不著」「一直失眠」「一天都只想睡」
4. 疲倦/沒活力
   - 很累、體力不足、倦怠
   - 範例：「累到不想動」「醒來像被抽乾」
5. 食慾改變
   - 食慾下降或暴食
   - 範例：「完全不想吃」「一整天都在吃」
6. 自責與低自尊
   - 自我貶低、罪惡感、無價值、覺得自己讓人失望
   - 範例：「我好沒用」「對不起大家」「覺得自己一無是處」
7. 注意力不佳/思考遲鈍
   - 注意力困難、記憶差、思緒混亂
   - 範例：「腦袋遲鈍」「專注不到三秒」
8. 動作變慢或躁動
   - 行為緩慢、坐立不安、明顯變快/變慢
   - 範例：「慢半拍」「煩到坐不住」
9. 自傷/自殺想法
   - 想死、傷害自己、希望自己消失
   - 範例：「我只想消失」「死了比較輕鬆」「割手」
10. 表達想尋求心理治療/諮商
    - 範例：「我想去看心身心科/精神科」「我需要心理諮商」
11. 明確醫療診斷或用藥資訊
    - 範例：「被診斷出憂鬱症」「有吃抗憂鬱藥」

【JSON 輸出規則】
- 一律輸出完整 1~11 號症狀，即使次數為 0 也必須出現。
- 禁止省略任何欄位。
- 只輸出JSON，不要解釋，不要加文字。

【輸出範例（請照此完整格式）】
{
  "symptom_counts": {
    "1": 1, "2": 0, "3": 2, "4": 0, "5": 0,
    "6": 0, "7": 0, "8": 0, "9": 0, "10": 0, "11": 0
  },
  "post_flags": [true, false, true, ...]
}
"""

SYMPTOM_PROMPT_TEMPLATE = """
{rules}

===【14天觀察視窗內貼文（日期→內容）】===
{posts}

請嚴格依照上述規則，輸出各症狀次數 JSON。
"""

# 貼文層級頻率判定
def calculate_frequency_label(per_post_flags, symptom_counts, total_posts):
    if total_posts == 0:
        return "general_user", "無貼文", 0, symptom_counts

    total_events = sum(1 for flag in per_post_flags if flag)
    event_ratio = total_events / total_posts
    if event_ratio >= 0.50:
        return "high_depression_risk_user", f"症狀貼文占比 {event_ratio:.0%} ≥50%", total_events, symptom_counts
    elif event_ratio >= 0.25:
        return "low_depression_risk_user", f"症狀貼文占比 {event_ratio:.0%} 介於25%–49%", total_events, symptom_counts
    else:
        return "non_depression_risk_user", f"症狀貼文占比 {event_ratio:.0%} <25%", total_events, symptom_counts

# 呼叫Gemini API函式
def classify_user_with_frequency(user_posts):
    posts_text = "\n".join(
        [f"{row.post_date.date()}｜{row.post_content}"
         for _, row in user_posts.iterrows()]
    )

    prompt = SYMPTOM_PROMPT_TEMPLATE.format(
        rules=SYMPTOM_DETECTION_RULES,
        posts=posts_text
    )

    for retry in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
            resp_text = getattr(response, "text", "").strip()
            cleaned = resp_text.replace("```json", "").replace("```", "").strip()
            symptom_json = json.loads(cleaned)
            symptom_counts = symptom_json.get("symptom_counts", {})
            symptom_counts = {str(i): symptom_counts.get(str(i), 0) for i in range(1, 12)}
            per_post_flags = symptom_json.get("post_flags", [])

            total_posts = len(user_posts)

            # 防止API漏回post_flags導致比例失真
            if len(per_post_flags) != total_posts:
                per_post_flags = per_post_flags + [False] * (total_posts - len(per_post_flags))


            label, reason, total_events, symptom_counts = calculate_frequency_label(
                per_post_flags, symptom_counts, total_posts
            )

            print(f"→ 事件數: {total_events}, 總發文數 {total_posts} , {reason}")
            print(f"→ 症狀分布: {json.dumps(symptom_counts, ensure_ascii=False)}")

            return {
                "gemini_user_status_label": label,
                "gemini_symptom_counts": json.dumps(symptom_counts, ensure_ascii=False)
            }

        except Exception as e:
            print(f"API error: {e}")
            time.sleep(1 + retry)

    return {"gemini_user_status_label": "error", "gemini_symptom_counts": "{}"}

# ==========================================
# 主流程(批次分析)
# ==========================================
batch_size = 10
df = df.dropna(subset=["username"])
df = df[df["username"].astype(str).str.strip() != ""]
df["username"] = df["username"].astype(str)
all_users = df["username"].unique()
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
        existing_labels = user_df["gemini_user_status_label"].dropna().astype(str).str.strip()
        if existing_labels.str.len().gt(0).any():
            print(f"→ 使用者已有標註（{existing_labels.iloc[0]}），略過")
            continue

        # 根據帳號類型決定是否跳過
        acc_type = str(user_df.iloc[0].get("gemini_account_type_result", "")).lower()
        SKIP_TYPES = {"not_enough_posts", "business", "non_life", "api_error", "", "nan"}
        if acc_type in SKIP_TYPES:
            df.loc[df["username"] == user, "gemini_user_status_label"] = "skipped"
            print(f"→ 帳號類型為 {acc_type} ，跳過分析")
            continue

        result = classify_user_with_frequency(user_df)

        if result["gemini_user_status_label"] == "error":
            failed_users.append(user)
            print(f"→ API error，稍後補跑")
        else:
            df.loc[df["username"] == user, "gemini_user_status_label"] = result["gemini_user_status_label"]
            # df.loc[df["username"] == user, "gemini_symptom_counts"] = result["gemini_symptom_counts"]
            print(f"→ 標註為 {result['gemini_user_status_label']}")

        time.sleep(random.uniform(1.5, 3))

    df.to_csv("user_status_label_depression.csv", index=False, encoding="utf-8-sig")

    if i + batch_size < len(all_users):
        batch_sleep = random.uniform(5, 10)
        print(f"△ Batch完成，休息{batch_sleep:.1f}秒...")
        time.sleep(batch_sleep)
    else:
        print("△ 最後一個batch完成！")

# ==========================================
# 補跑API_error使用者
# ==========================================
if failed_users:
    print(f"\n===== 補跑 {len(failed_users)} 個 API error 使用者 =====\n")
    for user in failed_users:
        user_df = df[df["username"] == user]
        print(f"補跑使用者：{user}")

        retry_count = 0
        max_retry = 5
        success = False

        while retry_count < max_retry and not success:
            result = classify_user_with_frequency(user_df)
            if result["gemini_user_status_label"] != "error":
                df.loc[df["username"] == user, "gemini_user_status_label"] = result["gemini_user_status_label"]
                # df.loc[df["username"] == user, "gemini_symptom_counts"] = result["gemini_symptom_counts"]
                print(f"→ 標註為 {result['gemini_user_status_label']}")
                success = True
                print(f"→ 補跑成功：{result['gemini_user_status_label']}")
            else:
                retry_count += 1
                wait_time = 2 ** retry_count
                print(f"→ 補跑失敗 {retry_count}/{max_retry}，等待 {wait_time} 秒後重試...")
                time.sleep(wait_time)

        if not success:
            df.loc[df["username"] == user, "gemini_user_status_label"] = "API_error"
            print("→ 補跑仍失敗，標記為 API_error")

        time.sleep(random.uniform(1.5, 3))
    df.to_csv("user_status_label_depression.csv", index=False, encoding="utf-8-sig")
else:
    print("無需補跑API error之使用者")

print("\n完成 user_status_label + symptom_counts 判定！")