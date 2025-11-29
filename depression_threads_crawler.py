from selenium import webdriver
from bs4 import BeautifulSoup
import time
import pandas as pd
import langid
import os
from datetime import datetime, timedelta
query_list =["沒意義","需要陪伴","撐不住","活著好痛苦","提不起勁","身心科","失眠","心悸","胸悶","疲憊","活著好累","空虛","冷漠","恐慌","心理疾病","心情低落","內耗","孤獨","絕望","不該存在","推薦心理諮商","想死","精神科","負面情緒","憂鬱","悲傷","壓抑","坐立不安","脆弱","情緒崩潰","記憶力變差"]
def save_to_excel(data, filename):
    df = pd.DataFrame(data)
    if os.path.exists(filename):
        with pd.ExcelWriter(filename, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    else:
        df.to_excel(filename, index=False)
driver = webdriver.Chrome()
driver.get('https://www.threads.com/login')
print("登入成功後按 Enter 繼續")
input()
query=""
for query in query_list:
    data_zh = []
    driver.get(f'https://www.threads.com/search?q={query}')
    time.sleep(5)
    for _ in range(2):
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
        time.sleep(5)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    posts = soup.find_all('div', class_="x78zum5 xdt5ytf")
    for post in posts:
        try:
            username_element = post.find('span', class_="x1lliihq x193iq5w x6ikm8r x10wlt62 xlyipyv xuxw1ft")
            username = username_element.get_text(strip=True) if username_element else 'unknown'
            if username == 'unknown':
                continue
            test_created_at_element = post.find('time', class_="x1rg5ohu xnei2rj x2b8uid xuxw1ft")
            test_created_at = test_created_at_element.get('title','').strip() if test_created_at_element else 'unknown'
            date_part = test_created_at.split(" ")[0] + " " + test_created_at.split(" ")[2]
            date_part = date_part.replace("上午","AM").replace("下午","PM")
            test_dt = datetime.strptime(date_part,"%Y年%m月%d日 %p%I:%M")
            test_created_at = test_dt.strftime('%Y-%m-%d %H:%M')
            if datetime.now() - test_dt > timedelta(days=14):
                continue
            body = post.find('div', class_="x1a6qonq x6ikm8r x10wlt62 xj0a0fe x126k92a x6prxxf x7r5mf7")
            test_post_text = ''
            if body:
                span_texts = body.find_all('span')
                lines = []
                for s in span_texts:
                    text = s.get_text(strip=True)
                    class_attr = s.get("class",[])
                    if text and text != "翻譯" and "xeuugli" not in class_attr:
                        lines.append(text)
                test_post_text = '\n'.join(lines)
            query_in_post = any(q in test_post_text for q in query_list)
            if not query_in_post:
                continue
            language, _ = langid.classify(str(test_post_text)) if test_post_text else 'unknown'
            if language != 'zh':
                continue
            driver.get(f'https://www.threads.com/@{username}')
            time.sleep(5)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            div = soup.find('div', class_='xcrlgei')
            h1 = div.find('h1') if div else None
            display_name = h1.get_text(strip=True) if h1 else ''
            intro_site = soup.find('div', class_='xw7yly9')
            intro_tag = intro_site.find('span') if intro_site else None
            bio = intro_tag.get_text(strip=True) if intro_tag else ''
            for _ in range(2):
                for _ in range(2):
                    driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
                    time.sleep(5)
                user_soup = BeautifulSoup(driver.page_source,'html.parser')
                user_posts = user_soup.find_all('div', class_="x78zum5 xdt5ytf")
                post_count = len(user_posts)
                if post_count < 3:
                    continue
                user_list = []
                count = 0
                for post in user_posts:
                    post_text = ''
                    created_at_element = post.find('time', class_="x1rg5ohu xnei2rj x2b8uid xuxw1ft")
                    created_at = created_at_element.get('title','').strip() if created_at_element else 'unknown'
                    date_part = created_at.split(" ")[0] + " " + created_at.split(" ")[2]
                    date_part = date_part.replace("上午","AM").replace("下午","PM")
                    dt = datetime.strptime(date_part,"%Y年%m月%d日 %p%I:%M")
                    created_at = dt.strftime('%Y-%m-%d %H:%M')
                    if timedelta(0) <= (test_dt - dt) <= timedelta(days=14):
                        body = post.find('div', class_="x1a6qonq x6ikm8r x10wlt62 xj0a0fe x126k92a x6prxxf x7r5mf7")
                        if body:
                            span_texts = body.find_all('span')
                            lines = []
                            for s in span_texts:
                                text = s.get_text(strip=True)
                                class_attr = s.get("class",[])
                                if text and text != "翻譯" and "xeuugli" not in class_attr:
                                    lines.append(text)
                            post_text = '\n'.join(lines)
                    if post_text:
                        language,_ = langid.classify(str(post_text))
                        if language != 'zh':
                            count += 1
                        user_list.append({
                            'username': username,
                            'display_name': display_name,
                            'post_content': post_text,
                            'post_date': created_at,
                            'bio': bio,
                            'matched_keyword': query,
                        })
                if count/post_count <= 0.5:
                    sorted_user_list = sorted(
                        user_list,
                        key=lambda x: datetime.strptime(x['post_date'], "%Y-%m-%d %H:%M")
                    )
                    data_zh.extend(sorted_user_list)
        except Exception as e:
            print(f"錯誤處理貼文: {e}")
            continue
    save_to_excel(data_zh, 'threads_data_zh.xlsx')
    print(f"已處理關鍵字：{query}")
driver.quit()
os.startfile('threads_data_zh.xlsx')
