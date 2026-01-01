from google import genai
import pandas as pd
from dotenv import load_dotenv
import os
import json
import time
import random

# ==========================================
# API 設定
# ==========================================
MODEL = "gemini-2.5-flash"
load_dotenv()

API_KEYS = [
    os.getenv("PRO_API_KEY"),
    os.getenv("MIS1_GEMINI_KEY"),
    os.getenv("TEST2_GEMINI_KEY"),
    os.getenv("TEST3_GEMINI_KEY"),
    os.getenv("TEST4_GEMINI_KEY"),
    os.getenv("FU2_API_KEY"),
    os.getenv("FU_API_KEY"),
    os.getenv("MISEE_API_KEY"),
    os.getenv("TEST1_GEMINI_KEY"),
    os.getenv("CAMP_API_KEY"),
    os.getenv("GOOGLE_API_KEY"),
]

invalid_keys = set()
API_KEYS = [k for k in API_KEYS if k and k.strip()]
if not API_KEYS:
    raise RuntimeError("無可用API Key，請檢查.env設定")

current_key_index = 0
GLOBAL_BACKOFF_UNTIL = 0
key_cooldowns = {i: 0 for i in range(len(API_KEYS))}

def get_client_with_validation():
    global current_key_index

    tried = 0
    max_keys = len(API_KEYS)

    while tried < max_keys:
        # 若此 key 已被標記為無效，直接跳過
        if current_key_index in invalid_keys:
            rotate_api_key()
            tried += 1
            continue

        key = API_KEYS[current_key_index]

        # ---------- 基本型別與空值防呆 ----------
        if not isinstance(key, str) or not key.strip():
            print(f"⚠ API Key #{current_key_index+1} 為空或無效，永久停用")
            invalid_keys.add(current_key_index)
            rotate_api_key()
            tried += 1
            continue

        # ---------- 冷卻檢查 ----------
        now = time.time()
        cooldown_until = key_cooldowns.get(current_key_index, 0)
        if now < cooldown_until:
            wait_time = cooldown_until - now
            print(f"→ API Key #{current_key_index+1} 冷卻中，等待 {wait_time:.1f}s")
            time.sleep(wait_time)

        # ---------- 嘗試初始化 Client ----------
        try:
            client = genai.Client(api_key=key.strip())
            print(f"→ 使用 API Key #{current_key_index+1}")
            return client

        except Exception as e:
            print(f"!! API Key #{current_key_index+1} 初始化失敗：{e}")
            # 初始化失敗視為永久無效 key
            invalid_keys.add(current_key_index)
            rotate_api_key()
            time.sleep(1)
            tried += 1

    # 若所有 key 都嘗試過仍失敗
    raise RuntimeError("所有 API Key 均無法初始化（已全部失效或不可用）")

def rotate_api_key():
    global current_key_index
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    print(f"→ 已切換至 API Key #{current_key_index + 1}")

client = get_client_with_validation()

# PHQ-9症狀判定規則
SYMPTOM_DETECTION_RULES = r"""
你是一個用於「高純度憂鬱文本訓練資料蒐集」的嚴格標註模型。

【資料用途聲明（非常重要）】
本任務的輸出將直接用於機器學習模型訓練，而非臨床篩檢或風險預測。
因此你必須採用「極端保守策略」：
- 寧可漏判（False Negative）
- 絕對不可誤判（False Positive）
- 若無法高度確信，請一律標記為 0
請不要嘗試同理、推測或補完使用者狀態。

----------------------------------------------------------------

【任務】
分析同一位使用者在「連續14天內」的Threads貼文內容，判斷每篇貼文是否包含「可被高度確信的憂鬱相關症狀事件」。

僅標記：
- 使用者以第一人稱描述「自身」的心理或生理狀態
不包含：
- 引用他人、創作、歌詞、轉貼
- 假設性語句、玩笑或反諷
- 對社會、新聞或他人的評論

----------------------------------------------------------------

【單篇貼文判定原則】
- 每篇貼文作為計數單位
- 同一篇貼文，只要符合任一症狀，即計為1個憂鬱事件
- 判定需考慮貼文內語意前後脈絡，但不必跨貼文整合
- 只標記第一人稱、自述心理/生理狀態的明確語句
- 模糊負向情緒或單句情緒詞不得計數
- 非短期情緒、非事件反應、非比喻或誇飾語氣

----------------------------------------------------------------

【症狀分類及範例】
1. 失去興趣 / 缺乏動機  
   - 範例：「提不起勁」「什麼都沒意思」「以前喜歡現在不想做」
2. 明顯且持續的負向情緒  
   - 範例：「每天都快撐不下去」「一直想哭」「覺得人生沒希望」「情緒很糟」
3. 睡眠困擾  
   - 範例：「整晚睡不著」「一直失眠」「一天都只想睡」
4. 疲倦 / 沒活力  
   - 範例：「累到不想動」「醒來像被抽乾」
5. 食慾明顯改變  
   - 範例：「完全不想吃」「一整天都在吃」
6. 自責與低自尊  
   - 範例：「我好沒用」「對不起大家」「覺得自己一無是處」
7. 注意力不佳 / 思考遲鈍  
   - 範例：「腦袋遲鈍」「專注不到三秒」
8. 動作變慢或躁動  
   - 範例：「慢半拍」「煩到坐不住」
9. 自傷或自殺相關想法  
   - 範例：「我只想消失」「死了比較輕鬆」「割手」
10. 表達想尋求心理治療或諮商  
    - 範例：「我想去看心身科/精神科」「我需要心理諮商」
11. 明確醫療診斷或用藥資訊  
    - 範例：「被診斷出憂鬱症」「有吃抗憂鬱藥」

----------------------------------------------------------------

【輸出規則】
- 一律輸出完整 1~11 號症狀，即使次數為0
- 禁止省略任何欄位
- 每篇貼文若有症狀，請累加對應症狀次數
- post_flags[i] = True 表示該篇至少有一個症狀
- post_flags[i] = False 表示該篇沒有任何症狀
- 輸出的 post_flags 與 symptom_counts 必須完全一致
- 僅輸出 JSON，不要附加任何解釋文字

----------------------------------------------------------------

【JSON 輸出範例】
{
  "symptom_counts": {
    "1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
    "6": 0, "7": 0, "8": 0, "9": 0, "10": 0, "11": 0
  },
  "post_flags": [true, false, false, ...]
}
"""

SYMPTOM_PROMPT_TEMPLATE = """
{rules}

===【14天觀察視窗內貼文（日期→內容）】===
{posts}

請嚴格依照上述規則，輸出各症狀次數 JSON。
"""

def call_gemini_with_rotation(prompt: str):
    global client, current_key_index, GLOBAL_BACKOFF_UNTIL
    for _ in range(len(API_KEYS)):
        now = time.time()
        if now < GLOBAL_BACKOFF_UNTIL:
            sleep_time = GLOBAL_BACKOFF_UNTIL - now
            print(f"⚠ 全域退避中，等待 {sleep_time:.1f}s")
            time.sleep(sleep_time)
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
            key_cooldowns[current_key_index] = time.time() + random.uniform(18, 30)
            return response.text.strip()
        except Exception as e:
            msg = str(e).lower()
            print(f"!! API錯誤（key #{current_key_index+1}）：{e}")
            if "503" in msg or "overloaded" in msg:
                backoff = random.uniform(40, 90)
                GLOBAL_BACKOFF_UNTIL = time.time() + backoff
                print(f"⚠ 模型過載，啟動全域退避 {backoff:.1f}s")
                continue
            if "429" in msg or "rate" in msg or "quota" in msg:
                print(f"⚠ Key #{current_key_index+1} 額度用盡（429），立刻輪替")
                rotate_api_key()
                client = get_client_with_validation()
                continue
            if "401" in msg or "403" in msg:
                print(f"⚠ Key #{current_key_index+1} 權限錯誤，輪轉")
                rotate_api_key()
                client = get_client_with_validation()
                continue
            print("⚠ 非預期錯誤，短暫等待後重試")
            time.sleep(random.uniform(5, 10))
            continue
    return None

def calculate_frequency_label(per_post_flags, symptom_counts, total_posts):
    if total_posts == 0:
        return "general_user", "無貼文", 0, 0.0, symptom_counts
    total_events = sum(1 for flag in per_post_flags if flag)
    event_ratio = total_events / total_posts
    if event_ratio >= 0.50:
        return "high_depression_risk_user", f"症狀貼文占比 {event_ratio:.0%} ≥50%", total_events, event_ratio, symptom_counts
    elif event_ratio >= 0.25:
        return "low_depression_risk_user", f"症狀貼文占比 {event_ratio:.0%} 介於25%–49%", total_events, event_ratio, symptom_counts
    else:
        return "non_depression_risk_user", f"症狀貼文占比 {event_ratio:.0%} <25%", total_events, event_ratio, symptom_counts

def classify_user_status(user_posts: pd.DataFrame):
    posts_text = "\n".join(
        f"{row.post_date.date()}｜{row.post_content}"
        for _, row in user_posts.iterrows()
    )
    prompt = SYMPTOM_PROMPT_TEMPLATE.format(
        rules=SYMPTOM_DETECTION_RULES,
        posts=posts_text
    )
    raw = call_gemini_with_rotation(prompt)
    if not raw:
        return None
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        print("!! JSON 解析失敗")
        return None
    symptom_counts = {str(i): data.get("symptom_counts", {}).get(str(i), 0) for i in range(1, 12)}
    post_flags = data.get("post_flags", [])
    total_posts = len(user_posts)
    if len(post_flags) < total_posts:
        post_flags += [False] * (total_posts - len(post_flags))
    else:
        post_flags = post_flags[:total_posts]
    label, reason, total_events, event_ratio, symptom_counts = calculate_frequency_label(
        post_flags, symptom_counts, total_posts
    )
    print(f"→ 事件數: {total_events}, 總發文數: {total_posts}, {reason}")
    print(f"→ 症狀分布: {json.dumps(symptom_counts, ensure_ascii=False)}")
    return {"label": label, "symptom_counts": json.dumps(symptom_counts, ensure_ascii=False)}

# ==========================================
# 主流程
# ==========================================
batch_size = 10
df = pd.read_csv("account_type_result_depression_19.csv", encoding="utf-8-sig")
df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")
df["username"] = df["username"].astype(str)

string_columns = ["gemini_user_status_label", "human_user_status_label"]
for col in string_columns:
    if col not in df.columns:
        df[col] = pd.Series(dtype="string")
    else:
        df[col] = df[col].astype("string")

# 基本資料清理
df = df.dropna(subset=["username"])
df = df[df["username"].str.strip() != ""]
all_users = df["username"].unique()

failed_users = []

for i in range(0, len(all_users), batch_size):
    batch_users = all_users[i:i + batch_size]
    print(f"\n===== 處理 batch {i // batch_size + 1} ({len(batch_users)} users) =====\n")

    for user in batch_users:
        user_df = df[df["username"] == user]
        print(f"處理使用者：{user}")

        # 已有標註 → 直接略過
        existing_labels = (user_df["gemini_user_status_label"].dropna().astype(str).str.strip())
        if existing_labels.str.len().gt(0).any():
            print(f"→ 使用者已有標註（{existing_labels.iloc[0]}），略過")
            continue

        # 身分 Gate 判斷
        gemini_acc = str(
            user_df.iloc[0].get("gemini_account_type_result", "")
        ).strip().lower()

        human_acc = str(
            user_df.iloc[0].get("human_user_status_label", "")
        ).strip().lower()

        allow_analysis = False

        if gemini_acc == "pass":
            allow_analysis = True

        elif gemini_acc == "needs_manual_review":
            if human_acc == "pass":
                allow_analysis = True
            else:
                df.loc[df["username"] == user, "gemini_user_status_label"] = "skipped"
                print("→ 身分需人工複審且未通過（非 pass），跳過憂鬱分析")
                continue

        else:
            df.loc[df["username"] == user, "gemini_user_status_label"] = "skipped"
            print(f"→ 非可分析之帳號類型 {gemini_acc}，跳過分析")
            continue

        # 進行憂鬱傾向判斷
        if not allow_analysis:
            # 理論上不會進到這裡，防呆
            df.loc[df["username"] == user, "gemini_user_status_label"] = "skipped"
            print("→ 未通過身分 gate，跳過分析")
            continue

        result = classify_user_status(user_df)

        if not result:
            failed_users.append(user)
            print("→ API error，稍後補跑")
            continue

        df.loc[
            df["username"] == user,
            "gemini_user_status_label"
        ] = result["label"]
        print(f"→ 標註為 {result['label']}")

        time.sleep(random.triangular(12, 25, 18))

    # 每個 batch 結束即寫檔
    df.to_csv("user_status_label_depression.csv",index=False,encoding="utf-8-sig")

    if i + batch_size < len(all_users):
        batch_sleep = random.uniform(20, 40)
        print(f"△ Batch完成，休息 {batch_sleep:.1f} 秒...")
        time.sleep(batch_sleep)
    else:
        print("△ 最後一個 batch 完成！")

# 補跑流程
if failed_users:
    print(f"\n===== 補跑 {len(failed_users)} 位失敗使用者 =====\n")
    for user in failed_users:
        user_df = df[df["username"] == user]
        print(f"補跑使用者：{user}")
        result = classify_user_status(user_df)
        if result:
            df.loc[df["username"] == user, "gemini_user_status_label"] = result["label"]
            print(f"→ 補跑成功：{result}")
        else:
            df.loc[df["username"] == user, "gemini_user_status_label"] = "needs_manual_review"
            print("→ 補跑仍失敗，標記為 needs_manual_review（人工複審）")
        time.sleep(random.triangular(12, 25, 18))
    df.to_csv("user_status_label_depression.csv", index=False, encoding="utf-8-sig")
else:
    print("無需補跑之使用者")

print("完成 user_status_label + symptom_counts 判定")