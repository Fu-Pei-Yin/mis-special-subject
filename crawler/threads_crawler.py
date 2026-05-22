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

    # 優先用 ISO 格式（datetime 屬性），最穩定
    try:
        dt = datetime.fromisoformat(date_text.replace('Z', '+00:00'))
        # 轉成 naive datetime（去掉時區）
        if dt.tzinfo is not None:
            from datetime import timezone
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            # 轉成台灣時間（UTC+8）
            dt = dt + timedelta(hours=8)
        return dt
    except:
        pass

    # 備用：解析中文 title 格式
    # 可能格式：'2026年5月17日 上午11:32' 或 '2026年5月17日星期日 上午11:32'
    try:
        # 移除星期（若存在）：找「日星期」後的星期字樣
        cleaned = re.sub(r'星期[一二三四五六日]', '', date_text).strip()
        # cleaned = '2026年5月17日 上午11:32'
        parts = cleaned.split(' ')
        if len(parts) >= 2:
            date_raw = parts[0]   # '2026年5月17日'
            time_raw = parts[1]   # '上午11:32' 或 '下午8:34'
            date_combined = f"{date_raw} {time_raw}".replace("上午", "AM").replace("下午", "PM")
            dt = datetime.strptime(date_combined, "%Y年%m月%d日 %p%I:%M")
            return dt
    except:
        pass

    return None

def contains_invalid_fraction(text):
    match = re.search(r'(\d+)/(\d+)$', text)
    if match and match.group(1) != '1':
        return True
    return False


def extract_post_content(post_container, username):
    try:
        content_candidates = []

        # 策略一：從已知的內容 div 往下找 x1lliihq span（根據 HTML 圖片確認的結構）
        # 修正：原本只賦值給 text 變數，從未 append 進 content_candidates
        main_content_divs = post_container.find_all('div', class_=lambda x: x and any(
            marker in str(x) for marker in ['x1iorvi4', 'x78zum5', 'xdt5ytf']
        ))
        for div in main_content_divs:
            spans = div.find_all('span', class_=lambda x: x and 'x1lliihq' in str(x))
            for span in spans:
                text = span.get_text(strip=True)
                if text:  # ← 修正：補上 append，原本漏掉這行
                    content_candidates.append(text)

        # 策略二：直接找所有 x1lliihq span，取其內層子 span 的文字
        # 根據 HTML 圖片：x1lliihq span 內有 ::before、子 <span>（實際文字）、::after
        # 不能限制「葉節點」，要取子 span 的 get_text()
        if not content_candidates:
            target_spans = post_container.find_all('span', class_=lambda x: x and 'x1lliihq' in str(x))
            for span in target_spans:
                # 優先取直接子 span 的文字（避免重複抓外層）
                child_spans = span.find_all('span', recursive=False)
                if child_spans:
                    for child in child_spans:
                        text = child.get_text(strip=True)
                        if text:
                            content_candidates.append(text)
                else:
                    text = span.get_text(strip=True)
                    if text:
                        content_candidates.append(text)

        # 策略三：fallback — 所有 span 掃描，過濾 UI 文字
        if not content_candidates:
            all_spans = post_container.find_all('span')
            for span in all_spans:
                text = span.get_text(strip=True)
                if ('則留言' not in text and
                    '次分享' not in text and
                    username not in text):
                    content_candidates.append(text)
        
        seen_content = set()
        unique_lines = []
        
        for text in content_candidates:
            normalized = ' '.join(text.split())
            
            is_duplicate = False
            to_remove = None
            
            for existing in seen_content:
                if normalized in existing or existing in normalized:
                    if len(normalized) > len(existing):
                        to_remove = existing
                        break
                    else:
                        is_duplicate = True
                        break
            
            if to_remove:
                seen_content.discard(to_remove)
                unique_lines = [line for line in unique_lines if line != to_remove]
            
            if not is_duplicate and normalized:
                seen_content.add(normalized)
                unique_lines.append(normalized)
        
        final_content = '\n'.join(unique_lines)
        
        ui_patterns = [
            r'\d+/\d+$',
            r'^\d+\s*(分鐘|小時|天|週|個月)前?$',
            r'^已驗證$',
            r'^\d+\s*則留言$',
            r'^\d+\s*次分享$',
            r'^查看其他\d+\s*則回覆$',
            r'^查看更多$',
            r'^\d{4}-\d{2}-\d{2}$',
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
        
        if len(final_content) < 10:
            article = post_container if post_container.name == 'article' else post_container.find_parent('article')
            if article:
                text_divs = article.find_all('div', recursive=True)
                max_text_div = max(text_divs, key=lambda d: len(d.get_text(strip=True)), default=None)
                if max_text_div:
                    final_content = max_text_div.get_text(separator='\n', strip=True)
                    lines = [l.strip() for l in final_content.split('\n') if len(l.strip()) > 10]
                    final_content = '\n'.join(lines)
        if "已釘選" in final_content:
            return ""
        result = re.sub(r'(?m)^\s*\d+\s*\n?', '', final_content)
        return result.strip()

    except Exception as e:
        return ""

import hashlib

def get_post_signature(post_text, length=500):
    if not post_text:
        return None
    text = post_text[:length].strip()
    return hashlib.md5(text.encode("utf-8")).hexdigest()

class ThreadsCrawler:
    def __init__(self, username, headless=True, progress_callback=None):
        self.username = username
        self.display_name = ""
        self.bio = ""
        self.all_posts_raw = []
        self.cutoff_date = None
        self.first_non_pinned_date = None
        self.progress_callback = progress_callback or (lambda msg: print(msg))
        
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 20)

    def _log(self, msg):
        self.progress_callback(msg)

    def scroll_and_wait(self, wait_time=3):
        """
        修正 Bug 4：scroll 後加入明確等待，確保新內容有機會渲染。
        同時回傳是否頁面有增高（用於判斷是否已到底部）。
        """
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        self.driver.execute_script("window.scrollBy(0, window.innerHeight * 2);")
        time.sleep(wait_time)
        new_height = self.driver.execute_script("return document.body.scrollHeight")
        return new_height != last_height

    def auto_login(self, cookie_path="cookies.pkl"):
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
            self._log("✓ 自動登入成功\n")
        except FileNotFoundError:
            self._log(f"✗ 找不到 cookie 檔案: {cookie_path}\n")

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
        if self.cutoff_date is not None:
            return
        
        if len(self.all_posts_raw) < 5:
            return
        
        # 以今天（執行當下）往前推 14 天作為截止點
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.cutoff_date = today - timedelta(days=14)
        
        self._log(f"📅 今日日期: {today.strftime('%Y-%m-%d')}")
        self._log(f"📅 抓取截止日期（今天往前14天）: {self.cutoff_date.strftime('%Y-%m-%d %H:%M')}")

    def wait_for_posts_to_load(self, timeout=15):
        """
        修正 Bug 3：等待頁面中至少出現一個 <time> 元素，
        確認貼文已渲染後再開始抓取，避免在空頁面上爬取。
        """
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "time"))
            )
            return True
        except TimeoutException:
            self._log("⚠️ 等待貼文載入超時，頁面可能為空或載入失敗")
            return False

    def extract_posts_with_cutoff(self, seen_posts):
        new_posts_count = 0
        should_stop = False

        # 用 JavaScript 直接從已渲染的 DOM 抓取貼文，
        # 避免 page_source 的靜態 HTML 與實際 DOM class 不符的問題
        js_result = self.driver.execute_script("""
        const results = [];
        const timeElements = document.querySelectorAll('time[datetime]');

        timeElements.forEach(timeEl => {
            try {
                const datetime = timeEl.getAttribute('datetime');

                // 往上找貼文根節點：找到包含貼文內容的最近 div
                // 策略：從 time 往上，找到第一個 role=link 的 a 標籤的父層
                // 再繼續往上找有實質文字內容（>20字）的節點
                let node = timeEl;
                let postText = '';
                let found = false;

                // 先往上找到 time 所在的 a[role=link] 或直接的 link 容器
                for (let i = 0; i < 30; i++) {
                    node = node.parentElement;
                    if (!node) break;

                    // 找到貼文層級的容器：有 data-pressable-container 屬性
                    if (node.getAttribute('data-pressable-container') === 'true') {
                        // 從這個容器往上一層，找兄弟節點中的貼文文字
                        const parent = node.parentElement;
                        if (parent) {
                            // 找 parent 下所有直接子 div，取文字最長的
                            let maxLen = 0;
                            let bestEl = null;
                            parent.querySelectorAll('div > span, div > div > span').forEach(el => {
                                const txt = el.innerText || el.textContent || '';
                                if (txt.length > maxLen && txt.length > 10) {
                                    maxLen = txt.length;
                                    bestEl = el;
                                }
                            });
                            if (bestEl) {
                                postText = bestEl.innerText || bestEl.textContent || '';
                                found = true;
                            }
                        }
                        break;
                    }

                    // 備用：找到包含 data-interactive-id 的容器
                    if (node.getAttribute('data-interactive-id')) {
                        postText = node.innerText || node.textContent || '';
                        found = true;
                        break;
                    }
                }

                // 最後備用：找最近的大型文字區塊
                if (!found || postText.length < 5) {
                    node = timeEl;
                    for (let i = 0; i < 15; i++) {
                        node = node.parentElement;
                        if (!node) break;
                        const txt = node.innerText || '';
                        // 取第一個文字長度超過 20 字的祖先節點
                        if (txt.length > 20) {
                            postText = txt;
                            break;
                        }
                    }
                }

                if (postText && postText.length > 0) {
                    results.push({
                        datetime: datetime,
                        text: postText
                    });
                }
            } catch(e) {}
        });

        return results;
        """)

        self._log(f"  [debug] js_result count={len(js_result) if js_result else 0}")

        if not js_result:
            return 0, False

        for item in js_result:
            try:
                datetime_str = item.get('datetime', '')
                raw_text = item.get('text', '')

                dt = safe_parse_datetime(datetime_str)
                if not dt:
                    continue

                # 清理文字：移除 UI 雜訊行
                ui_patterns = [
                    r'^\d+\s*(分鐘|小時|天|週|個月)前?$',
                    r'^已驗證$',
                    r'^\d+\s*則留言$',
                    r'^\d+\s*次分享$',
                    r'^查看其他\d+\s*則回覆$',
                    r'^查看更多$',
                    r'^\d+/\d+$',
                    r'^已釘選$',
                ]
                lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
                lines = [l for l in lines if not any(re.match(p, l) for p in ui_patterns)]
                post_text = '\n'.join(lines).strip()

                if not post_text or len(post_text) < 5:
                    continue
                if "已釘選" in post_text:
                    continue
                if contains_invalid_fraction(post_text):
                    continue

                post_signature = get_post_signature(post_text)
                if post_signature in seen_posts:
                    continue
                seen_posts.add(post_signature)

                post_date_str = dt.strftime("%Y-%m-%d %H:%M")
                exists = any(
                    item2["content"] == post_text and item2["post_date"] == post_date_str
                    for item2 in self.all_posts_raw
                )
                if not exists:
                    self.all_posts_raw.append({
                        "content": post_text,
                        "post_date": post_date_str
                    })
                    new_posts_count += 1

                if self.cutoff_date is None and len(self.all_posts_raw) >= 5:
                    self.update_cutoff_date()

                if self.cutoff_date and dt < self.cutoff_date:
                    self._log(f"⏱️  發現超過截止日期的貼文: {dt.strftime('%Y-%m-%d %H:%M')}")
                    should_stop = True
                    break

            except Exception:
                continue

        return new_posts_count, should_stop

    def check_user_exists(self):
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        not_found_text = [
            "亂晃的人不一定是迷路，但這個頁面真的走丟了",
            "連結失效或頁面不存在。請返回以繼續探索。",
            "Page not found",
            "Sorry, this page isn't available"
        ]
        
        page_text = soup.get_text()
        for text in not_found_text:
            if text in page_text:
                return False
        
        if soup.find('div', string=re.compile("返回")):
            return False
            
        return True

    def crawl_user(self):
        url = f"https://www.threads.com/@{self.username}"
        self._log(f"正在訪問: {url}")
        self.driver.get(url)

        self._log("等待頁面貼文載入...")
        loaded = self.wait_for_posts_to_load(timeout=15)
        if not loaded:
            self._log("⚠️ 頁面載入異常，嘗試繼續...")
        else:
            time.sleep(2)

        if not self.check_user_exists():
            self._log(f"⚠ 錯誤：帳號 @{self.username} 不存在或已設為私人。")
            self.driver.quit()
            return {"error": "user_not_found"}

        self._log("正在提取用戶資訊...")
        self.extract_user_info()
        self._log(f"✓ 用戶名稱: {self.display_name}")
        self._log(f"✓ 帳號: @{self.username}")
        self._log(f"✓ 簡介: {self.bio[:50]}..." if len(self.bio) > 50 else f"✓ 簡介: {self.bio}")
        
        # seen_posts 在 crawl_user 層級統一維護，跨多次 scroll 共用
        seen_posts = set()
        max_scroll_attempts = 100
        scroll_attempts = 0
        no_new_posts_count = 0
        
        self._log("開始抓取貼文...\n")
        
        while scroll_attempts < max_scroll_attempts:
            new_posts, should_stop = self.extract_posts_with_cutoff(seen_posts)
            
            if should_stop:
                self._log(f"✓ 已達截止日期，停止抓取\n")
                break
            
            if new_posts == 0:
                no_new_posts_count += 1
                if no_new_posts_count >= 5:
                    self._log(f"✓ 連續5次未發現新貼文，停止抓取\n")
                    break
            else:
                no_new_posts_count = 0
                self._log(f"📝 已抓取 {len(self.all_posts_raw)} 篇貼文...")
            
            # 修正 Bug 4：scroll 後等待時間加長至 4 秒，確保新內容渲染
            self.scroll_and_wait(wait_time=4)
            scroll_attempts += 1
        
        self._log(f"\n🔄 正在處理貼文...")
        self._log(f"✓ 原始收集貼文數: {len(self.all_posts_raw)}")
        
        self.all_posts_raw.sort(key=lambda x: datetime.strptime(x["post_date"], "%Y-%m-%d %H:%M"), reverse=True)
        self._log(f"✓ 已按時間排序（由新到舊）")
        
        if len(self.all_posts_raw) > 0:
            latest_post_date = datetime.strptime(self.all_posts_raw[0]["post_date"], "%Y-%m-%d %H:%M")
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = today - timedelta(days=14)
            
            self._log(f"📅 最新貼文時間: {latest_post_date.strftime('%Y-%m-%d %H:%M')}")
            self._log(f"📅 今日日期: {today.strftime('%Y-%m-%d')}")
            self._log(f"📅 輸出截止日期（今天往前14天）: {cutoff_date.strftime('%Y-%m-%d %H:%M')}")
            
            filtered_posts = []
            for post in self.all_posts_raw:
                post_date = datetime.strptime(post["post_date"], "%Y-%m-%d %H:%M")
                if post_date >= cutoff_date:
                    filtered_posts.append(post)
            
            self._log(f"✓ 篩選後14天內貼文數: {len(filtered_posts)}")
            
            if len(filtered_posts) > 0:
                oldest_in_range = datetime.strptime(filtered_posts[-1]["post_date"], "%Y-%m-%d %H:%M")
                self._log(f"✓ 貼文時間範圍: {oldest_in_range.strftime('%Y-%m-%d %H:%M')} ~ {latest_post_date.strftime('%Y-%m-%d %H:%M')}\n")
        else:
            filtered_posts = []
            cutoff_date = None
            latest_post_date = None
            self._log("⚠️ 未抓取到任何貼文\n")
        
        self.driver.quit()
        
        return {
            "username": self.username,
            "display_name": self.display_name,
            "bio": self.bio,
            "posts": filtered_posts,
            "first_post_date": latest_post_date.strftime("%Y-%m-%d %H:%M") if latest_post_date else None,
            "cutoff_date": cutoff_date.strftime("%Y-%m-%d %H:%M") if cutoff_date else None,
            "total_posts_collected": len(self.all_posts_raw)
        }


if __name__ == "__main__":
    crawler = ThreadsCrawler("yu.12299", headless=False)
    crawler.auto_login(cookie_path="C:/Users/USER/Desktop/課程/專題/threads_depression_web/crawler/cookies.pkl")
    
    user_data = crawler.crawl_user()
    
    if user_data.get("error"):
        print(f"❌ 錯誤: {user_data['error']}")
    else:
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