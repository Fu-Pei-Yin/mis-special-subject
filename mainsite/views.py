from django.shortcuts import render, redirect
from crawler.threads_crawler import ThreadsCrawler
from core.predict import predict_username
from datetime import datetime, date
from core.dataset import (
    compute_user_numeric_features,
    extract_logic_features,
    NEG_WORDS,
    POS_WORDS,
    BIO_RISK_WORDS,
)
from django.utils.timezone import now
from requests.exceptions import ReadTimeout, ConnectionError
import socket
import json
import re
from django.http import JsonResponse

# --- 工具類別：解決 JSON 序列化問題 ---
class DatetimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.strftime("%Y-%m-%d %H:%M")
        return super().default(obj)


def _fmt_date(raw):
    if isinstance(raw, (datetime, date)):
        return raw.strftime("%Y-%m-%d %H:%M")
    return str(raw) if raw else ""


# --- 視圖函數 ---


def index(request):
    return render(request, "index.html")

ALLOWED_CHARS_REGEX = re.compile(r'^[A-Za-z0-9._]+$')
def is_valid_dot_rule(username: str) -> bool:
    if username.startswith("."):
        return False
    if username.endswith("."):
        return False
    if ".." in username:
        return False
    return True
def analyze(request):
    if request.method != "POST":
        return redirect("index")
    username = request.POST.get("username", "").strip()
    # 空值
    if not username:
        return render(request, "errors/input_error.html", {
            "message": "尚未輸入使用者名稱"
        })
    # 非法字元
    if not ALLOWED_CHARS_REGEX.match(username):
        return render(request, "errors/input_error.html", {
            "message": "僅允許英文字母、數字、底線(_)與句點(.)"
        })
    # 句點規則
    if not is_valid_dot_rule(username):
        return render(request, "errors/input_error.html", {
            "message": "句點(.)不可開頭、結尾或連續出現"
        })

    return render(request, "scanning.html", {"username": username})


def result(request):
    if request.method != "POST":
        return redirect("index")
    try:
        username = request.POST.get("username")
        try:
            crawler = ThreadsCrawler(username, headless=True)
            # 請確保此路徑在您的環境中正確
            crawler.auto_login(cookie_path="C:/Users/USER/Desktop/課程/專題/threads_depression_web/crawler/cookies.pkl")
            user_data = crawler.crawl_user()
        except (ReadTimeout, TimeoutError, socket.timeout, ConnectionError):
            return render(request, "errors/system_busy.html")

        if not user_data or not user_data.get('posts'):
            return render(request, "errors/no_data.html", {"username": username})

        try:
            risk_prob = predict_username(user_data)
            if risk_prob >= 0.66:
                risk_level = "高風險"
            elif risk_prob >= 0.33:
                risk_level = "中風險"
            else:
                risk_level = "低風險"
        except Exception:
            return render(request, "errors/analysis_failed.html")

        bio = str(user_data.get('bio', ''))
        post_texts = [str(p['content']) for p in user_data['posts']]
        post_dates = [_fmt_date(p.get('post_date', '')) for p in user_data['posts']]

        num_features = compute_user_numeric_features(post_texts, bio, post_dates)
        logic_features = extract_logic_features(post_texts, bio)

        # 提取特徵 (注意索引需與 dataset.py 定義一致)
        # num_features: [post_count, avg_post_len, bio_len, night_ratio, max, min]
        # logic_features: [neg_count, pos_count, logic_score, bio_flag, neg_ratio]
        night_ratio_val = num_features[3]
        neg_ratio_val = logic_features[4]

        # 標記關鍵詞
        posts_with_keywords = []
        for text, str_date in zip(post_texts, post_dates):
            neg_hits = [w for w in NEG_WORDS if w in text]
            pos_hits = [w for w in POS_WORDS if w in text]
            posts_with_keywords.append({
                "post_date": str_date,
                "content": text,
                "neg_words": neg_hits,
                "pos_words": pos_hits,
                "has_risk_word": bool(neg_hits),
            })

        # 打包白箱數據 (此 JSON 將傳往 analyze_result.html 的隱藏欄位)
        whitebox_data = {
            "neg_count": int(logic_features[0]),
            "pos_count": int(logic_features[1]),
            "logic_score": int(logic_features[2]),
            "bio_flag": bool(logic_features[3]),
            "neg_ratio": round(neg_ratio_val * 100, 1),
            "night_ratio": round(night_ratio_val * 100, 1),
            "post_count": int(num_features[0]),
            "avg_post_len": round(float(num_features[1]), 1),
            "bio_risk_words": [w for w in BIO_RISK_WORDS if w in bio],
            "posts_keywords": posts_with_keywords,
        }
        whitebox_payload = json.dumps(whitebox_data, ensure_ascii=False, cls=DatetimeEncoder)
        
        # 【新增】將 whitebox_payload 存入 session 作為備用
        request.session['whitebox_payload'] = whitebox_payload
        
        return render(request, "analyze_result.html", {
            "username": username,
            "display_name": user_data.get("display_name", username),
            "bio": bio,
            "risk_prob": round(risk_prob * 100, 1),
            "risk_level": risk_level,
            "posts": user_data["posts"],
            "now": datetime.now(),
            "Nighttime_activity": round(night_ratio_val * 100, 1),
            "emotional_negativity": round(neg_ratio_val * 100, 1),
            "whitebox_payload": whitebox_payload, # 關鍵：傳給前端
        })
    # 全域未知錯誤
    except Exception as e:
        print("UNEXPECTED ERROR:", e)
        return render(request, "errors/error.html")


def suggest_report(request):
    if request.method != "POST":
        return redirect("index")

    # 1. 取得基礎資訊
    username = request.POST.get("username")
    risk_level = request.POST.get("risk_level")
    # 確保從字串轉回 float，處理可能遺失的情形
    try:
        risk_prob_pct = float(request.POST.get("risk_prob", 0))
        night_pct = float(request.POST.get("Nighttime_activity", 0))
        neg_pct = float(request.POST.get("emotional_negativity", 0))
    except ValueError:
        risk_prob_pct = night_pct = neg_pct = 0

    # 2. 【修正】優先檢查 POST，沒有再檢查 session 備用
    raw_payload = request.POST.get("whitebox_payload")
    if not raw_payload:  # 如果 POST 沒有，檢查 session
        raw_payload = request.session.get("whitebox_payload")
    
    try:
        if raw_payload:
            whitebox = json.loads(raw_payload)
            print(123)
        else:
            whitebox = {}
    except (json.JSONDecodeError, TypeError):
        whitebox = {}
        print(456)

    # 3. 提取白箱欄位用於邏輯判斷與顯示
    neg_count = whitebox.get("neg_count", 0)
    logic_score = whitebox.get("logic_score", 0)
    bio_flag = whitebox.get("bio_flag", False)
    bio_risk_words = whitebox.get("bio_risk_words", [])

    # 4. 產出執行摘要 (Summary)
    summary_parts = [
        f"根據對使用者 {username} 近期的社群行為數據分析，系統評估其心理風險為 {risk_level} ({risk_prob_pct}%)。"
    ]
    if night_pct >= 60:
        summary_parts.append(f"偵測到顯著的夜間活動偏移 ({night_pct}%)，這通常與睡眠品質不佳相關。")
    if neg_pct >= 30:
        summary_parts.append(f"語義分析顯示負面文字能量較高 ({neg_pct}%)，建議關注情緒穩定性。")
    if bio_risk_words:
        summary_parts.append(f"個人簡介中出現關鍵字：{'、'.join(bio_risk_words)}。")
    summary = " ".join(summary_parts)

    # 5. 行為模式 (Behavior Patterns)
    behavior_patterns = []
    if night_pct >= 60:
        behavior_patterns.append({"title": "晝夜節律延遲", "content": "作息集中於深夜，生理時鐘可能失調。"})
    if neg_pct >= 30:
        behavior_patterns.append({"title": "情緒負向偏好", "content": "貼文內容包含較多負面情感詞彙。"})
    behavior_patterns.append({"title": "自我敘事傾向", "content": "貼文多以第一人稱出發，表現出高度自我覺察或沉思。"})

    # 6. 風險因子 (Risk Factors)
    risk_factors = []
    if night_pct >= 60: risk_factors.append("長期熬夜導致的生理疲勞")
    if neg_pct >= 30: risk_factors.append("負面思考循環")
    if risk_level == "高風險": risk_factors.append("社交退縮與心理韌性臨界點")
    if not risk_factors: risk_factors.append("目前數據顯示心理韌性尚屬穩定")

    # 7. AI 建議 (AI Guidelines)
    ai_guidelines = []
    if risk_level in ["高風險", "中風險"]:
        ai_guidelines.append("建議安排專業心理諮商或使用 PHQ-9 量表進行自評。")
    if night_pct >= 50:
        ai_guidelines.append("嘗試建立規律作息，減少深夜使用社群媒體的時間。")
    ai_guidelines.append("建議增加戶外運動或與信任的親友進行實體互動。")

    # 8. 回傳模板
    context = {
        "username": username,
        "period": "Past 14 Days",
        "generated_time": now(),
        "risk_level": risk_level,
        "risk_prob": risk_prob_pct,
        "Nighttime_activity": night_pct,
        "emotional_negativity": neg_pct,
        "summary": summary,
        "behavior_patterns": behavior_patterns,
        "risk_factors": risk_factors,
        "ai_guidelines": ai_guidelines,
        "whitebox": whitebox, # 直接傳入 dict 供模板遍歷
    }

    return render(request, "suggest.html", context)

def custom_404_view(request, exception):
    if "application/json" in request.headers.get("Accept", ""):
        return JsonResponse ({
        "error": "Page not found",
        "status_code": 404,
        "detail": "The requested URL was not found."
        }, status=404)
    else:
        return render(request, "errors/error_404.html", status=404)