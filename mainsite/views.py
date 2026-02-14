from django.shortcuts import render, redirect
from crawler.threads_crawler import ThreadsCrawler
from core.predict import predict_username
from datetime import datetime
from core.dataset import compute_user_numeric_features, extract_logic_features
from django.utils.timezone import now
from requests.exceptions import ReadTimeout, ConnectionError
import socket
import re

USERNAME_REGEX = re.compile(r'^(?!.*\.\.)(?!\.)(?!.*\.$)[A-Za-z0-9._]+$')

def index(request):
    return render(request, "index.html")

def analyze(request):
    if request.method != "POST":
        return redirect("index")
    username = request.POST.get("username", "").strip()
    if not username:
        return render(request, "errors/input_error.html", {
            "message": "尚未輸入使用者名稱"
        })
    if not USERNAME_REGEX.match(username):
        return render(request, "errors/input_error.html", {
            "message": "Threads 使用者名稱僅允許英文字母、數字、底線(_)與句點(.)"
        })
    return render(request,"scanning.html",locals())

def result(request):
    if request.method != "POST":
        return redirect("index")
    username = request.POST.get("username")
    try:
        crawler = ThreadsCrawler(username, headless=True)
        crawler.auto_login(cookie_path="C:/Users/eva19/Desktop/graduate_project/threads_depression_web/crawler/cookies.pkl")
        user_data = crawler.crawl_user()
    except (ReadTimeout, TimeoutError, socket.timeout, ConnectionError):
        return render(request, "errors/system_busy.html")
    
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
    if not user_data or not user_data['posts']:
        return render(request, "errors/no_data.html", {"username": username})
    post_texts = [p['content'] for p in user_data['posts']]
    post_dates = [p['post_date'] for p in user_data['posts']]
    num_features = compute_user_numeric_features(post_texts, user_data['bio'], post_dates)
    logic_features = extract_logic_features(post_texts, user_data['bio'])
    Nighttime_activity=num_features[3]
    emotional_negativity=logic_features[4]
    return render(
        request,"analyze_result.html",
        {
            "username": username,
            "display_name":user_data["display_name"],
            "bio":user_data["bio"],
            "risk_prob": round(risk_prob * 100, 1),
            "risk_level": risk_level,
            "posts": user_data["posts"],
            "now":datetime.now(),
            "Nighttime_activity":round(Nighttime_activity * 100, 1),
            "emotional_negativity": round(emotional_negativity * 100, 1)
        }
    )

def suggest_report(request):
    if request.method != "POST":
        return redirect("index")
    username = request.POST.get("username")
    risk_prob = float(request.POST.get("risk_prob")) / 100
    risk_level = request.POST.get("risk_level")
    Nighttime_activity = float(request.POST.get("Nighttime_activity"))
    emotional_negativity = float(request.POST.get("emotional_negativity"))
    summary_parts = []
    summary_parts.append(
        f"根據對使用者 {username} 近兩週社群行為數據之分析，"
        f"系統評估其心理風險等級為 {risk_level}（{round(risk_prob*100,1)}%）。"
    )
    if Nighttime_activity >= 60:
        summary_parts.append(
            f"使用者呈現顯著的晝夜節律延遲現象（夜間活動比例 {Nighttime_activity}%），"
            "可能與睡眠品質下降或作息失衡相關。"
        )
    if emotional_negativity >= 30:
        summary_parts.append(
            f"語義分析顯示負面情緒表達比例達 {emotional_negativity}%，"
            "高於一般社群使用者基準值，需留意潛在情緒困擾風險。"
        )
    summary = " ".join(summary_parts)
    behavior_patterns = []
    if Nighttime_activity >= 60:
        behavior_patterns.append({
            "title": "晝夜節律延遲（Circadian Rhythm Delay）",
            "content": "大量互動集中於深夜時段，顯示生理節律可能長期偏移。"
        })
    if emotional_negativity >= 30:
        behavior_patterns.append({
            "title": "情緒負向表達偏高（Negative Affect）",
            "content": "貼文中負面語彙與低情緒能量語句頻繁出現。"
        })
    behavior_patterns.append({
        "title": "自我導向敘事（Self-focused Narrative）",
        "content": "語句結構以第一人稱為主，顯示高度內在反思傾向。"
    })
    risk_factors = []
    if Nighttime_activity >= 60:
        risk_factors.append("長期作息失衡與睡眠節律紊亂")
    if emotional_negativity >= 30:
        risk_factors.append("負面情緒累積與情緒調節能力下降")
    if risk_level == "高風險":
        risk_factors.append("社會互動退縮與心理韌性下降風險")
    ai_guidelines = []
    if risk_level in ["高風險", "中風險"]:
        ai_guidelines.append(
            "建議進一步進行標準化心理量表評估（如 BDI-II 或 PHQ-9）。"
        )
    if Nighttime_activity >= 60:
        ai_guidelines.append(
            "可考慮導入睡眠節律調整策略（CBT-I 或行為作息重建）。"
        )
    if emotional_negativity >= 30:
        ai_guidelines.append(
            "建議練習情緒覺察與認知重構技巧，以降低負向思維循環。"
        )
    ai_guidelines.append(
        "逐步增加低壓力社交互動，有助於恢復外在情緒回饋機制。"
    )
    return render(
        request,
        "suggest.html",
        {
            "username": username,
            "period": "Past 14 Days",
            "generated_time": now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "risk_level": risk_level,
            "summary": summary,
            "behavior_patterns": behavior_patterns,
            "risk_factors": risk_factors,
            "ai_guidelines": ai_guidelines
        }
    )

