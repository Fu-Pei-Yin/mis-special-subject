from selenium import webdriver
from bs4 import BeautifulSoup
import time
import pandas as pd
import langid
import os
from datetime import datetime, timedelta

def safe_parse_datetime(date_text):
    """安全解析 Threads 日期，避免 index error / format error"""
    try:
        parts = date_text.split(" ")
        if len(parts) < 3:
            return None

        date_raw = parts[0]
        time_raw = parts[2]

        date_combined = f"{date_raw} {time_raw}"
        date_combined = date_combined.replace("上午", "AM").replace("下午", "PM")

        dt = datetime.strptime(date_combined, "%Y年%m月%d日 %p%I:%M")
        return dt
    except:
        return None


def safe_lang_classify(text):
    """避免 too many values to unpack"""
    try:
        result = langid.classify(text)
        if isinstance(result, tuple) and len(result) >= 1:
            return result[0]
        return "unknown"
    except:
        return "unknown"


def safe_get_post_text(body):
    """安全提取貼文內容"""
    try:
        span_texts = body.find_all('span')
        lines = []
        for s in span_texts:
            text = s.get_text(strip=True)
            class_attr = s.get("class", [])
            if text and text != "翻譯" and "xeuugli" not in class_attr:
                lines.append(text)
        return "\n".join(lines)
    except:
        return ""


def save_to_excel(data, filename):
    df = pd.DataFrame(data)
    if os.path.exists(filename):
        with pd.ExcelWriter(filename, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    else:
        df.to_excel(filename, index=False)


# ====== 主程式開始 ======
driver = webdriver.Chrome()
driver.get('https://www.threads.com/login')
print("登入成功後按 Enter 繼續")
input()

times = 150

for _ in range(times):
    driver.get('https://www.threads.com')
    time.sleep(5)

    data_zh = []
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    posts = soup.find_all('div', class_="x78zum5 xdt5ytf")

    for post in posts:
        try:
            # ---- 取得 username ----
            username_element = post.find('span', class_="x1lliihq x193iq5w x6ikm8r x10wlt62 xlyipyv xuxw1ft")
            username = username_element.get_text(strip=True) if username_element else None
            if not username:
                continue

            # ---- 解析最初的貼文日期 ----
            created_at_element = post.find('time', class_="x1rg5ohu xnei2rj x2b8uid xuxw1ft")
            created_at_raw = created_at_element.get('title', '').strip() if created_at_element else None
            test_dt = safe_parse_datetime(created_at_raw)
            if not test_dt:
                continue

            # ---- 取得貼文內容 ----
            body = post.find('div', class_="x1a6qonq x6ikm8r x10wlt62 xj0a0fe x126k92a x6prxxf x7r5mf7")
            test_post_text = safe_get_post_text(body) if body else ""

            # ---- 中文判斷 ----
            language = safe_lang_classify(test_post_text)
            if language != 'zh':
                continue

            # ---- 進入使用者頁面 ----
            driver.get(f'https://www.threads.com/@{username}')
            for _ in range(3):
                for _ in range(2):
                    time.sleep(5)
                    soup = BeautifulSoup(driver.page_source, 'html.parser')

                # ---- 使用者名稱 ----
                div = soup.find('div', class_='xcrlgei')
                h1 = div.find('h1') if div else None
                display_name = h1.get_text(strip=True) if h1 else ""

                # ---- 自介 ----
                intro_site = soup.find('div', class_='xw7yly9')
                intro_tag = intro_site.find('span') if intro_site else None
                bio = intro_tag.get_text(strip=True) if intro_tag else ""

                # ---- 滑動載入 ----
                for _ in range(2):
                    driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
                    time.sleep(5)

                user_soup = BeautifulSoup(driver.page_source, 'html.parser')
                user_posts = user_soup.find_all('div', class_="x78zum5 xdt5ytf")
                post_count = len(user_posts)
                if post_count < 3:
                    continue

                user_list = []
                count_non_zh = 0

                # ---- 處理使用者每一則貼文 ----
                for upost in user_posts:
                    created_el = upost.find('time', class_="x1rg5ohu xnei2rj x2b8uid xuxw1ft")
                    created_raw = created_el.get('title', '').strip() if created_el else None
                    dt = safe_parse_datetime(created_raw)
                    if not dt:
                        continue

                    post_text = ""
                    if timedelta(0) <= (test_dt - dt) <= timedelta(days=14):
                        body = upost.find('div', class_="x1a6qonq x6ikm8r x10wlt62 xj0a0fe x126k92a x6prxxf x7r5mf7")
                        post_text = safe_get_post_text(body) if body else ""

                    if post_text:
                        lang = safe_lang_classify(post_text)
                        if lang != 'zh':
                            count_non_zh += 1

                        user_list.append({
                            'username': username,
                            'display_name': display_name,
                            'post_content': post_text,
                            'post_date': dt.strftime("%Y-%m-%d %H:%M"),
                            'bio': bio
                        })

                # ---- 中文比例判斷 ----
                if count_non_zh / post_count <= 0.5:
                    sorted_user_list = sorted(
                        user_list,
                        key=lambda x: datetime.strptime(x['post_date'], "%Y-%m-%d %H:%M"),
                        reverse=True
                    )
                    data_zh.extend(sorted_user_list)
                    print(username, display_name, post_count, f"非中文比例: {count_non_zh}/{post_count}")

        except Exception as e:
            print("錯誤處理貼文:", e)
            continue

    save_to_excel(data_zh, 'threads_data_normal_zh.xlsx')

driver.quit()
os.startfile('threads_data_normal_zh.xlsx')
