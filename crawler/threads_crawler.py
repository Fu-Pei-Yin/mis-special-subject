from selenium import webdriver
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import langid

# 安全日期解析
def safe_parse_datetime(date_text):
    try:
        parts = date_text.split(" ")
        if len(parts) < 3:
            return None
        date_raw = parts[0]
        time_raw = parts[2]
        date_combined = f"{date_raw} {time_raw}".replace("上午", "AM").replace("下午", "PM")
        dt = datetime.strptime(date_combined, "%Y年%m月%d日 %p%I:%M")
        return dt
    except:
        return None

# 安全語言判斷
def safe_lang_classify(text):
    try:
        result = langid.classify(text)
        if isinstance(result, tuple) and len(result) >= 1:
            return result[0]
        return "unknown"
    except:
        return "unknown"

# 安全取得貼文內容
def safe_get_post_text(body):
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

# 檢查是否為置頂貼文
def is_pinned_post(post_element):
    try:
        # 尋找包含"已釘選"文字的span標籤
        pinned_span = post_element.find('span', string="已釘選")
        if pinned_span:
            return True
        
        # 檢查是否有特定類別的span標籤包含"已釘選"
        for span in post_element.find_all('span'):
            if span.get_text(strip=True) == "已釘選":
                return True
        
        # 檢查class屬性中是否包含特定標記（根據提供的HTML結構）
        pinned_class = post_element.find(class_="x1lliihq x1plvlek xryxfnj x1n2onr6 xyejjpt x15dsfln x193iq5w xeuugli x1fj9vlw x13faqbe x1vvkbs x1s928wv xhkezso x1gmr53x x1cpjm7i x1fgarty x1943h6x x1i0vuye xmd891q xo1l8bm xhgddhk x1yc453h")
        if pinned_class:
            return True
            
        return False
    except:
        return False

class ThreadsCrawler:
    def __init__(self, username, headless=True):
        self.username = username
        self.display_name = ""
        self.bio = ""
        self.posts = []

        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        self.driver = webdriver.Chrome(options=options)

    '''
    def login(self):
        self.driver = webdriver.Chrome()
        self.driver.get("https://www.threads.com/login")
        print("請登入 Threads，登入完成後按 Enter 繼續")
        input()
    '''
    def auto_login(self):
        self.driver = webdriver.Chrome()
        self.driver.get("https://www.threads.com/login")
        import pickle
        with open("C:/Users/USER/Desktop/課程/專題/threads_depression_detector/cookies.pkl", "rb") as f:
            cookies = pickle.load(f)
        for cookie in cookies:
            self.driver.add_cookie(cookie)
        self.driver.refresh()
        time.sleep(5)
        print("自動登入成功")

    def crawl_user(self):
        url = f"https://www.threads.com/@{self.username}"
        self.driver.get(url)
        time.sleep(5)
        
        # 初始化變數
        first_non_pinned_date = None
        found_first_non_pinned = False
        cutoff_date = None
        
        # 用集合來記錄已抓取的貼文內容，避免重複
        seen_posts = set()
        max_scroll_attempts = 50  # 最多嘗試下滑次數
        scroll_attempts = 0
        last_post_count = 0
        no_new_posts_count = 0
        
        while scroll_attempts < max_scroll_attempts:
            # 下滑到底部
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            
            # 解析當前頁面
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # 獲取顯示名稱和個人簡介（僅第一次）
            if not self.display_name:
                div = soup.find('div', class_='xcrlgei')
                h1 = div.find('h1') if div else None
                self.display_name = h1.get_text(strip=True) if h1 else ""
            
            if not self.bio:
                intro_site = soup.find('div', class_='xw7yly9')
                intro_tag = intro_site.find('span') if intro_site else None
                self.bio = intro_tag.get_text(strip=True) if intro_tag else ""
            
            # 解析貼文
            posts_raw = soup.find_all('div', class_="x78zum5 xdt5ytf")
            
            for upost in posts_raw:
                # 檢查是否為置頂貼文
                if is_pinned_post(upost):
                    print("發現置頂貼文，跳過...")
                    continue
                
                created_el = upost.find('time', class_="x1rg5ohu xnei2rj x2b8uid xuxw1ft")
                created_raw = created_el.get('title', '').strip() if created_el else None
                dt = safe_parse_datetime(created_raw)
                
                if not dt:
                    continue
                
                body = upost.find('div', class_="x1a6qonq x6ikm8r x10wlt62 xj0a0fe x126k92a x6prxxf x7r5mf7")
                post_text = safe_get_post_text(body) if body else ""
                
                if not post_text or post_text in seen_posts:
                    continue
                    
                seen_posts.add(post_text)
                
                # 如果還沒找到第一篇非置頂貼文，設定為第一篇非置頂貼文
                if not found_first_non_pinned:
                    first_non_pinned_date = dt
                    found_first_non_pinned = True
                    cutoff_date = first_non_pinned_date - timedelta(days=14)
                    print(f"找到第一篇非置頂貼文，日期: {first_non_pinned_date.strftime('%Y-%m-%d %H:%M')}")
                    print(f"開始抓取，截止日期: {cutoff_date.strftime('%Y-%m-%d %H:%M')}")
                
                # 檢查是否在14天範圍內
                if dt < cutoff_date:
                    print(f"發現超過14天的貼文 ({dt.strftime('%Y-%m-%d %H:%M')})，停止抓取")
                    self.driver.quit()
                    return {
                        "username": self.username,
                        "display_name": self.display_name,
                        "bio": self.bio,
                        "posts": self.posts
                    }
                
                # 檢查語言並添加到結果
                lang = safe_lang_classify(post_text)
                if lang == "zh":
                    self.posts.append({
                        "content": post_text, 
                        "post_date": dt.strftime("%Y-%m-%d %H:%M")
                    })
            
            # 檢查是否還有新貼文加載
            current_post_count = len(self.posts)
            if current_post_count == last_post_count:
                no_new_posts_count += 1
                if no_new_posts_count >= 3:  # 連續3次下滑沒有新貼文，則停止
                    print("連續3次下滑沒有新貼文，停止抓取")
                    break
            else:
                no_new_posts_count = 0
                last_post_count = current_post_count
            
            # 如果已經找到第一篇非置頂貼文且已經有貼文，檢查是否已達到截止日期
            if found_first_non_pinned and current_post_count > 0:
                # 檢查最新的貼文日期是否已經接近截止日期
                if len(self.posts) > 0:
                    latest_post_date = datetime.strptime(self.posts[-1]["post_date"], "%Y-%m-%d %H:%M")
                    if latest_post_date < cutoff_date:
                        print(f"已達到截止日期範圍，停止抓取")
                        break
            
            print(f"已抓取 {current_post_count} 篇貼文，繼續下滑...")
            scroll_attempts += 1
        
        # 如果一直沒有找到非置頂貼文（可能所有貼文都是置頂或沒有貼文）
        if not found_first_non_pinned:
            print("未找到非置頂貼文，可能是所有貼文都是置頂貼文或該用戶沒有貼文")
        
        print(f"完成抓取，共取得 {len(self.posts)} 篇14天內的貼文")
        self.driver.quit()
        return {
            "username": self.username,
            "display_name": self.display_name,
            "bio": self.bio,
            "posts": self.posts
        }

# 使用範例
if __name__ == "__main__":
    crawler = ThreadsCrawler("username", headless=False)  # 替換為實際用戶名
    # crawler.auto_login()  # 如果需要登入則取消註解
    user_data = crawler.crawl_user()
    print(f"用戶: {user_data['display_name']}")
    print(f"簡介: {user_data['bio']}")
    print(f"貼文數量: {len(user_data['posts'])}")
    for i, post in enumerate(user_data['posts'][:5], 1):  # 顯示前5篇貼文
        print(f"\n貼文 {i} ({post['post_date']}):")
        print(post['content'][:100] + "..." if len(post['content']) > 100 else post['content'])