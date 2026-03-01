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
from django.core.cache import cache
from requests.exceptions import ReadTimeout, ConnectionError
import socket
import json
import re
import threading
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

    if not username:
        return render(request, "errors/input_error.html", {
            "message": "尚未輸入使用者名稱"
        })
    if not ALLOWED_CHARS_REGEX.match(username):
        return render(request, "errors/input_error.html", {
            "message": "僅允許英文字母、數字、底線(_)與句點(.)"
        })
    if not is_valid_dot_rule(username):
        return render(request, "errors/input_error.html", {
            "message": "句點(.)不可開頭、結尾或連續出現"
        })

    # 初始化 cache
    cache_key = f"crawl_progress_{username}"
    done_key  = f"crawl_done_{username}"
    data_key  = f"crawl_data_{username}"
    error_key = f"crawl_error_{username}"

    cache.set(cache_key, [], timeout=300)
    cache.set(done_key,  False, timeout=300)
    cache.delete(data_key)
    cache.delete(error_key)

    # 背景執行爬蟲
    def run_crawler():
        def on_progress(msg):
            messages = cache.get(cache_key) or []
            messages.append(str(msg))
            cache.set(cache_key, messages, timeout=300)

        try:
            crawler = ThreadsCrawler(username, headless=True, progress_callback=on_progress)
            crawler.auto_login(
                cookie_path="C:/Users/USER/Desktop/課程/專題/threads_depression_web/crawler/cookies.pkl"
            )
            user_data = crawler.crawl_user()
            cache.set(data_key, user_data, timeout=300)
        except (ReadTimeout, TimeoutError, socket.timeout, ConnectionError):
            cache.set(error_key, "network_error", timeout=300)
        except Exception as e:
            cache.set(error_key, "unknown_error", timeout=300)
        finally:
            cache.set(done_key, True, timeout=300)

    thread = threading.Thread(target=run_crawler, daemon=True)
    thread.start()

    return render(request, "scanning.html", {"username": username})


# --- 前端輪詢進度 API ---
def crawl_progress(request, username):
    cache_key = f"crawl_progress_{username}"
    done_key  = f"crawl_done_{username}"
    error_key = f"crawl_error_{username}"

    # 用 session 記錄已讀取的訊息位置，避免重複顯示
    read_index = request.session.get(f"progress_index_{username}", 0)

    messages = cache.get(cache_key) or []
    new_messages = messages[read_index:]
    request.session[f"progress_index_{username}"] = len(messages)

    done  = cache.get(done_key) or False
    error = cache.get(error_key)

    return JsonResponse({
        "messages": new_messages,
        "done": done,
        "error": error,
    })


def result(request):
    if request.method != "POST":
        return redirect("index")

    try:
        username = request.POST.get("username")

        # 從 cache 取得爬蟲結果（由 analyze 的背景執行緒寫入）
        data_key  = f"crawl_data_{username}"
        error_key = f"crawl_error_{username}"

        error = cache.get(error_key)
        if error == "network_error":
            return render(request, "errors/system_busy.html")
        if error:
            return render(request, "errors/error.html")

        user_data = cache.get(data_key)

        # cache 不存在時（例如直接 POST 到 result）退回重新掃描
        if not user_data:
            return render(request, "scanning.html", {"username": username})

        if not user_data.get('posts'):
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

        num_features   = compute_user_numeric_features(post_texts, bio, post_dates)
        logic_features = extract_logic_features(post_texts, bio)

        night_ratio_val = num_features[3]
        neg_ratio_val   = logic_features[4]

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

        whitebox_data = {
            "neg_count":    int(logic_features[0]),
            "pos_count":    int(logic_features[1]),
            "logic_score":  int(logic_features[2]),
            "bio_flag":     bool(logic_features[3]),
            "neg_ratio":    round(neg_ratio_val * 100, 1),
            "night_ratio":  round(night_ratio_val * 100, 1),
            "post_count":   int(num_features[0]),
            "avg_post_len": round(float(num_features[1]), 1),
            "bio_risk_words":  [w for w in BIO_RISK_WORDS if w in bio],
            "posts_keywords":  posts_with_keywords,
        }
        whitebox_payload = json.dumps(whitebox_data, ensure_ascii=False, cls=DatetimeEncoder)

        request.session['whitebox_payload'] = whitebox_payload

        # 用完就清除 cache，釋放記憶體
        cache.delete(data_key)
        cache.delete(error_key)

        return render(request, "analyze_result.html", {
            "username":           username,
            "display_name":       user_data.get("display_name", username),
            "bio":                bio,
            "risk_prob":          round(risk_prob * 100, 1),
            "risk_level":         risk_level,
            "posts":              user_data["posts"],
            "now":                datetime.now(),
            "Nighttime_activity": round(night_ratio_val * 100, 1),
            "emotional_negativity": round(neg_ratio_val * 100, 1),
            "whitebox_payload":   whitebox_payload,
        })

    except Exception as e:
        print("UNEXPECTED ERROR:", e)
        return render(request, "errors/error.html")


def suggest_report(request):
    if request.method != "POST":
        return redirect("index")

    username = request.POST.get("username")
    risk_level = request.POST.get("risk_level")

    try:
        risk_prob_pct = float(request.POST.get("risk_prob", 0))
        night_pct     = float(request.POST.get("Nighttime_activity", 0))
        neg_pct       = float(request.POST.get("emotional_negativity", 0))
    except ValueError:
        risk_prob_pct = night_pct = neg_pct = 0

    raw_payload = request.POST.get("whitebox_payload")
    if not raw_payload:
        raw_payload = request.session.get("whitebox_payload")

    try:
        whitebox = json.loads(raw_payload) if raw_payload else {}
    except (json.JSONDecodeError, TypeError):
        whitebox = {}

    neg_count      = whitebox.get("neg_count", 0)
    logic_score    = whitebox.get("logic_score", 0)
    bio_flag       = whitebox.get("bio_flag", False)
    bio_risk_words = whitebox.get("bio_risk_words", [])

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

    behavior_patterns = []
    if night_pct >= 60:
        behavior_patterns.append({"title": "晝夜節律延遲", "content": "作息集中於深夜，生理時鐘可能失調。"})
    if neg_pct >= 30:
        behavior_patterns.append({"title": "情緒負向偏好", "content": "貼文內容包含較多負面情感詞彙。"})
    behavior_patterns.append({"title": "自我敘事傾向", "content": "貼文多以第一人稱出發，表現出高度自我覺察或沉思。"})

    risk_factors = []
    if night_pct >= 60:  risk_factors.append("長期熬夜導致的生理疲勞")
    if neg_pct >= 30:    risk_factors.append("負面思考循環")
    if risk_level == "高風險": risk_factors.append("社交退縮與心理韌性臨界點")
    if not risk_factors: risk_factors.append("目前數據顯示心理韌性尚屬穩定")

    ai_guidelines = []
    if risk_level in ["高風險", "中風險"]:
        ai_guidelines.append("建議安排專業心理諮商或使用 PHQ-9 量表進行自評。")
    if night_pct >= 50:
        ai_guidelines.append("嘗試建立規律作息，減少深夜使用社群媒體的時間。")
    ai_guidelines.append("建議增加戶外運動或與信任的親友進行實體互動。")

    context = {
        "username":           username,
        "period":             "Past 14 Days",
        "generated_time":     now(),
        "risk_level":         risk_level,
        "risk_prob":          risk_prob_pct,
        "Nighttime_activity": night_pct,
        "emotional_negativity": neg_pct,
        "summary":            summary,
        "behavior_patterns":  behavior_patterns,
        "risk_factors":       risk_factors,
        "ai_guidelines":      ai_guidelines,
        "whitebox":           whitebox,
    }

    return render(request, "suggest.html", context)


def custom_404_view(request, exception):
    if "application/json" in request.headers.get("Accept", ""):
        return JsonResponse({
            "error": "Page not found",
            "status_code": 404,
            "detail": "The requested URL was not found."
        }, status=404)
    else:
        return render(request, "errors/error_404.html", status=404)