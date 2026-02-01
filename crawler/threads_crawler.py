from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import re

def safe_parse_datetime(date_text):
    """安全解析日期時間 - 支援多種格式"""
    if not date_text:
        return None
    
    try:
        parts = date_text.split(" ")
        if len(parts) >= 3:
            date_raw = parts[0]
            time_raw = parts[2]
            date_combined = f"{date_raw} {time_raw}".replace("上午", "AM").replace("下午", "PM")
            dt = datetime.strptime(date_combined, "%Y年%m月%d日 %p%I:%M")
            return dt
    except:
        pass
    
    try:
        dt = datetime.fromisoformat(date_text.replace('Z', '+00:00'))
        return dt
    except:
        pass
    
    return None
def contains_invalid_fraction(text):
    """
    text 已為單行字串
    - 若結尾為 分子 ≠ 1 的分數（x/y）→ False
    - 其他情況（含 1/n 或非分數結尾）→ True
    """
    match = re.search(r'(\d+)/(\d+)$', text)

    if match and match.group(1) != '1':
        return True

    return False



def extract_post_content(post_container, username):
    """
    改進版內容提取 - 使用更精確的策略避免重複
    """
    try:
        # 策略1: 尋找主要內容區域的特徵 class
        content_candidates = []
        
        # 找到所有可能包含內容的 div
        main_content_divs = post_container.find_all('div', class_=lambda x: x and any(
            marker in str(x) for marker in ['x1iorvi4', 'x78zum5', 'xdt5ytf']
        ))
        
        # 策略2: 找尋包含實際文字內容的 span
        for div in main_content_divs:
            spans = div.find_all('span', class_=lambda x: x and 'x1lliihq' in str(x))
            for span in spans:
                text = span.get_text(strip=True)
        
        # 策略3: 如果上述方法失敗,使用備用方案
        if not content_candidates:
            all_spans = post_container.find_all('span')
            for span in all_spans:
                text = span.get_text(strip=True)
                # 嚴格過濾條件
                if ('則留言' not in text and
                    '次分享' not in text and
                    username not in text):
                    content_candidates.append(text)
        
        # 去重並組合
        seen_content = set()
        unique_lines = []
        
        for text in content_candidates:
            # 正規化文字(移除多餘空白)
            normalized = ' '.join(text.split())
            
            # 檢查是否為子字串(避免包含關係的重複)
            is_duplicate = False
            to_remove = None
            
            # 先找出需要移除的項目，不要在迭代時修改集合
            for existing in seen_content:
                if normalized in existing or existing in normalized:
                    # 保留較長的版本
                    if len(normalized) > len(existing):
                        to_remove = existing
                        break
                    else:
                        is_duplicate = True
                        break
            
            # 在迭代完成後才修改集合
            if to_remove:
                seen_content.discard(to_remove)
                unique_lines = [line for line in unique_lines if line != to_remove]
            
            if not is_duplicate and normalized:
                seen_content.add(normalized)
                unique_lines.append(normalized)
        
        # 最終清理
        final_content = '\n'.join(unique_lines)
        
        # 移除明顯的介面元素
        ui_patterns = [
            r'\d+/\d+$',  # 分頁標記
            r'^\d+\s*(分鐘|小時|天|週|個月)前?$',  # 時間標記
            r'^已驗證$',
            r'^\d+\s*則留言$',
            r'^\d+\s*次分享$',
            r'^查看其他\d+\s*則回覆$',
            r'^查看更多$',  # 更多按鈕
            r'^\d{4}-\d{2}-\d{2}$',  # 時間2025-12-8
            r'^\d{4}-\d{2}-\d{1}$',
            r'^\d{4}-\d{1}-\d{2}$',
            r'^\d{4}-\d{1}-\d{1}$',
        ]
        
        lines = final_content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not any(re.match(pattern, line) for pattern in ui_patterns):
                cleaned_lines.append(line)
        
        final_content = '\n'.join(cleaned_lines)
        
        # 如果內容太短,嘗試直接從 article 提取
        if len(final_content) < 10:
            article = post_container if post_container.name == 'article' else post_container.find_parent('article')
            if article:
                # 找到文字密度最高的 div
                text_divs = article.find_all('div', recursive=True)
                max_text_div = max(text_divs, key=lambda d: len(d.get_text(strip=True)), default=None)
                if max_text_div:
                    final_content = max_text_div.get_text(separator='\n', strip=True)
                    # 再次清理
                    lines = [l.strip() for l in final_content.split('\n') if len(l.strip()) > 10]
                    final_content = '\n'.join(lines)
        if "已釘選" in final_content:
            return ""
        result = re.sub(r'(?m)^\s*\d+\s*\n?', '', final_content)
        return result.strip()

    except Exception as e:
        print(f"⚠️ 提取內容時發生錯誤: {e}")
        return ""


class ThreadsCrawler:
    def __init__(self, username, headless=True):
        self.username = username
        self.display_name = ""
        self.bio = ""
        self.all_posts_raw = []
        self.cutoff_date = None
        self.first_non_pinned_date = None
        
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 20)

    def scroll_and_wait(self, wait_time=3):
        """滾動並等待新內容載入"""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(wait_time)
        new_height = self.driver.execute_script("return document.body.scrollHeight")
        return new_height != last_height

    def auto_login(self, cookie_path="cookies.pkl"):
        """自動登入使用 cookies"""
        self.driver.get("https://www.threads.com/login")
        time.sleep(2)
        
        try:
            import pickle
            with open(cookie_path, "rb") as f:
                cookies = pickle.load(f)
            
            for cookie in cookies:
                try:
                    self.driver.add_cookie(cookie)
                except:
                    pass
            
            self.driver.refresh()
            time.sleep(5)
            print("✓ 自動登入成功\n")
        except FileNotFoundError:
            print(f"✗ 找不到 cookie 檔案: {cookie_path}\n")

    def extract_user_info(self):
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        if not self.display_name: 
            div = soup.find('div', class_='xcrlgei') 
            h1 = div.find('h1') if div else None 
            self.display_name = h1.get_text(strip=True) if h1 else ""
        
        if not self.display_name:
            self.display_name = self.username
        
        bio_div = soup.find('div', class_='xw7yly9')
        if bio_div:
            self.bio = bio_div.get_text(strip=True)

    def update_cutoff_date(self):
        """動態更新截止日期"""
        if self.cutoff_date is not None:
            return
        
        if len(self.all_posts_raw) < 2:
            return
        
        
        for i, post in enumerate(self.all_posts_raw):
            post_date = datetime.strptime(post["post_date"], "%Y-%m-%d %H:%M")
            self.first_non_pinned_date = post_date
            self.cutoff_date = post_date - timedelta(days=14)
            
            print(f"📌 第一篇非置頂貼文: {self.first_non_pinned_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"📅 抓取截止日期: {self.cutoff_date.strftime('%Y-%m-%d %H:%M')}")
            break

    def extract_posts_with_cutoff(self, seen_posts):
        """提取貼文並即時檢查時間限制"""
        new_posts_count = 0
        should_stop = False
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        articles = soup.find_all('article')
        
        if not articles:
            time_elements = soup.find_all('time')
        else:
            time_elements = []
            for article in articles:
                time_elem = article.find('time')
                if time_elem:
                    time_elements.append(time_elem)
        
        for time_elem in time_elements:
            try:
                date_text = time_elem.get('title', '') or time_elem.get('datetime', '')
                dt = safe_parse_datetime(date_text)
                
                if not dt:
                    continue
                
                post_container = time_elem.find_parent('article')
                
                if not post_container:
                    current = time_elem
                    for _ in range(15):
                        current = current.parent
                        if not current:
                            break
                        if current.name == 'div' and current.get('class'):
                            classes = ' '.join(current.get('class', []))
                            if 'x1n2onr6' in classes:
                                post_container = current
                                break
                
                if not post_container:
                    continue
                
                post_text = extract_post_content(post_container, self.username)
                
                if not post_text or len(post_text) < 5:
                    continue
                
                # 檢查是否包含無效的分數格式
                if contains_invalid_fraction(post_text):
                    print(f"⏭️  跳過包含無效分數格式的貼文 (發文時間: {dt.strftime('%Y-%m-%d %H:%M')})")
                    continue
                
                # 如果這是第一篇貼文且包含無效分數，跳過
                if len(self.all_posts_raw) == 0 and contains_invalid_fraction(post_text):
                    print(f"⏭️  第一篇貼文包含無效分數格式，跳過")
                    continue
                
                # 使用內容的 hash 來去重(更可靠)
                post_signature = hash(post_text[:200].strip())
                if post_signature in seen_posts:
                    continue
                
                seen_posts.add(post_signature)
                if post_text != "" and r'^(?!1/\d+$)\d+/\d+$' not in post_text:
                    self.all_posts_raw.append({
                        "content": post_text,
                        "post_date": dt.strftime("%Y-%m-%d %H:%M")
                    })
                    new_posts_count += 1
                
                if self.cutoff_date is None and len(self.all_posts_raw) >= 3:
                    self.update_cutoff_date()
                
                if self.cutoff_date and dt < self.cutoff_date:
                    print(f"⏱️  發現超過截止日期的貼文: {dt.strftime('%Y-%m-%d %H:%M')}")
                    should_stop = True
                    break
                
            except Exception as e:
                continue
        
        return new_posts_count, should_stop

    def crawl_user(self):
        """爬取用戶資料"""
        url = f"https://www.threads.com/@{self.username}"
        print(f"正在訪問: {url}")
        self.driver.get(url)
        time.sleep(5)
        
        print("正在提取用戶資訊...")
        self.extract_user_info()
        print(f"✓ 用戶名稱: {self.display_name}")
        print(f"✓ 帳號: @{self.username}")
        print(f"✓ 簡介: {self.bio[:50]}..." if len(self.bio) > 50 else f"✓ 簡介: {self.bio}")
        print()
        
        seen_posts = set()
        max_scroll_attempts = 50
        scroll_attempts = 0
        no_new_posts_count = 0
        
        print("開始抓取貼文...\n")
        
        while scroll_attempts < max_scroll_attempts:
            new_posts, should_stop = self.extract_posts_with_cutoff(seen_posts)
            
            if should_stop:
                print(f"✓ 已達截止日期，停止抓取\n")
                break
            
            if new_posts == 0:
                no_new_posts_count += 1
                if no_new_posts_count >= 3:
                    print(f"✓ 連續3次未發現新貼文，停止抓取\n")
                    break
            else:
                no_new_posts_count = 0
                print(f"📝 已抓取 {len(self.all_posts_raw)} 篇貼文...")
            
            self.scroll_and_wait(wait_time=3)
            scroll_attempts += 1
        
        
        filtered_posts = []
        for i, post in enumerate(self.all_posts_raw):
            if self.cutoff_date:
                post_date = datetime.strptime(post["post_date"], "%Y-%m-%d %H:%M")
                if post_date >= self.cutoff_date:
                    filtered_posts.append(post)
            else:
                filtered_posts.append(post)
        
        print(f"✓ 分析完成！共取得 {len(filtered_posts)} 篇14天內的貼文\n")
        self.driver.quit()
        
        return {
            "username": self.username,
            "display_name": self.display_name,
            "bio": self.bio,
            "posts": filtered_posts,
            "first_post_date": self.first_non_pinned_date.strftime("%Y-%m-%d %H:%M") if self.first_non_pinned_date else None,
            "cutoff_date": self.cutoff_date.strftime("%Y-%m-%d %H:%M") if self.cutoff_date else None,
            "total_posts_collected": len(self.all_posts_raw)
        }


if __name__ == "__main__":
    crawler = ThreadsCrawler("uda711", headless=False)
    crawler.auto_login(cookie_path="C:/Users/USER/Desktop/課程/專題/threads_depression_web/crawler/cookies.pkl")
    
    user_data = crawler.crawl_user()
    
    print("=" * 70)
    print("用戶資訊")
    print("=" * 70)
    print(f"用戶名稱: {user_data['display_name']}")
    print(f"帳號: @{user_data['username']}")
    print(f"簡介: {user_data['bio']}")
    print()
    
    print("=" * 70)
    print("抓取統計")
    print("=" * 70)
    print(f"總共收集貼文數: {user_data['total_posts_collected']}")
    print(f"抓取截止日期: {user_data['cutoff_date']}")
    print(f"兩週內貼文總數: {len(user_data['posts'])}")
    print()
    
    if len(user_data['posts']) > 0:
        print("=" * 70)
        print("貼文內容（依時間由新到舊）")
        print("=" * 70)
        
        for i, post in enumerate(user_data['posts'], 1):
            print(f"\n【貼文 {i}】")
            print(f"發文時間: {post['post_date']}")
            print(f"內容:\n{post['content']}")
            print("-" * 70)
    else:
        print("⚠️ 未抓取到任何貼文")
    
    import json
    output_file = f"{user_data['username']}_posts_clean.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 資料已儲存至: {output_file}")