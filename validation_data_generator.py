# ============================================================
# validation_data_generator.py
# 為憂鬱傾向偵測模型生成驗證集資料
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# ----------------------
# 配置參數
# ----------------------
OUTPUT_PATH = "./validation_data.xlsx"
NUM_USERS = 100  # 生成用戶數量
DEPRESSION_RATIO = 0.3  # 憂鬱症案例比例
MIN_POSTS = 5
MAX_POSTS = 30
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ----------------------
# 內容模板庫
# ----------------------

# 憂鬱傾向貼文模板
DEPRESSION_POSTS = [
    "今天又是痛苦的一天，感覺撐不住了",
    "生活好像沒什麼意義，每天都很累",
    "覺得很空虛，需要有人陪伴",
    "真的好想消失，不想活了",
    "感到非常孤獨，沒有人理解我",
    "每天都在掙扎，看不到希望",
    "凌晨三點還是睡不著，心裡很絕望",
    "為什麼我要這麼痛苦",
    "已經無法再假裝開心了",
    "感覺自己是個負擔",
    "對所有事情都提不起興趣",
    "又一個失眠的夜晚，腦袋很亂",
    "真的好累，想要休息很久很久",
    "沒有力氣再繼續了",
    "覺得自己很沒用，什麼都做不好",
    "想哭但哭不出來",
    "世界好像與我無關",
    "為什麼我要承受這些",
    "感覺被世界拋棄了",
    "每天醒來都希望一切結束",
]

DEPRESSION_POSTS_MILD = [
    "最近心情不太好",
    "有點累，想休息",
    "感覺壓力有點大",
    "今天狀態不太好",
    "有些煩躁，不知道為什麼",
    "睡眠品質不好，有點影響情緒",
    "對很多事情都感到疲憊",
    "想要一個人靜一靜",
    "心情有點低落",
    "最近總是提不起勁",
]

# 正常貼文模板
NORMAL_POSTS = [
    "今天天氣真好，出門散步心情很好",
    "和朋友聚餐，很開心",
    "分享一下今天的午餐，超好吃",
    "週末計畫去爬山，期待",
    "剛看完一部電影，推薦給大家",
    "工作終於告一段落，可以放鬆了",
    "今天學到了新東西，很有成就感",
    "感謝大家的支持和鼓勵",
    "生活平淡但很滿足",
    "早安，新的一天開始了",
    "晚安，祝大家有個好夢",
    "運動完感覺很舒服",
    "今天效率很高，完成了很多事",
    "喝杯咖啡，享受悠閒時光",
    "最近在學習新技能",
    "和家人視訊聊天，很溫馨",
    "今天買了喜歡的東西",
    "工作雖然忙碌但很充實",
    "期待即將到來的假期",
    "分享一下最近的生活照",
]

# Bio 模板
DEPRESSION_BIOS = [
    "經常失眠的人 | 抑鬱症患者",
    "在低潮中掙扎 | 需要陪伴",
    "憂鬱症康復中 | 一天一天來",
    "焦慮與抑鬱共存 | 努力活著",
    "心理健康倡導者 | 曾經的抑鬱症患者",
    "與憂鬱症共處 | 尋找生活的意義",
    "情緒不穩定 | 正在治療中",
    "夜晚的思考者 | 白天的偽裝者",
]

NORMAL_BIOS = [
    "熱愛生活 | 喜歡攝影",
    "美食愛好者 | 旅行達人",
    "程式設計師 | 咖啡成癮者",
    "學生 | 喜歡閱讀和音樂",
    "上班族 | 週末運動愛好者",
    "自由工作者 | 享受自由",
    "貓奴 | 宅在家也很快樂",
    "健身愛好者 | 追求健康生活",
    "電影迷 | 分享生活日常",
    "愛笑的人 | 相信明天會更好",
]

# ----------------------
# 資料生成函數
# ----------------------

def generate_username():
    """生成隨機用戶名"""
    prefixes = ["user", "happy", "sunny", "blue", "dark", "night", "dream", "soul", "sky", "star"]
    return f"{random.choice(prefixes)}{random.randint(1000, 9999)}"

def generate_display_name():
    """生成隨機顯示名稱"""
    names = ["小明", "小華", "阿杰", "小美", "阿強", "小芳", "志明", "春嬌", 
             "David", "Emily", "John", "Sarah", "Mike", "Lisa", "Tom", "Amy"]
    return random.choice(names)

def generate_posts(is_depression, num_posts):
    """根據憂鬱狀態生成貼文"""
    posts = []
    
    if is_depression:
        # 憂鬱用戶：70%負面，20%輕度負面，10%正常
        for _ in range(num_posts):
            rand = random.random()
            if rand < 0.7:
                posts.append(random.choice(DEPRESSION_POSTS))
            elif rand < 0.9:
                posts.append(random.choice(DEPRESSION_POSTS_MILD))
            else:
                posts.append(random.choice(NORMAL_POSTS))
    else:
        # 正常用戶：90%正常，10%輕度負面（每個人都有低潮時刻）
        for _ in range(num_posts):
            if random.random() < 0.9:
                posts.append(random.choice(NORMAL_POSTS))
            else:
                posts.append(random.choice(DEPRESSION_POSTS_MILD))
    
    return posts

def generate_post_dates(num_posts, base_date=None):
    """生成貼文日期（過去14天內）"""
    if base_date is None:
        base_date = datetime.now()
    
    dates = []
    for _ in range(num_posts):
        # 隨機選擇過去14天內的某一天
        days_ago = random.randint(0, 13)
        # 隨機選擇時間（憂鬱用戶更可能在深夜發文）
        if random.random() < 0.3:  # 30%機率在深夜
            hour = random.randint(0, 5)
        else:
            hour = random.randint(6, 23)
        
        minute = random.randint(0, 59)
        post_date = base_date - timedelta(days=days_ago, hours=(23-hour), minutes=(59-minute))
        dates.append(post_date)
    
    # 按時間排序（從舊到新）
    dates.sort()
    return dates

def generate_validation_data():
    """生成完整的驗證集資料"""
    print("開始生成驗證集資料...")
    
    all_data = []
    num_depression = int(NUM_USERS * DEPRESSION_RATIO)
    num_normal = NUM_USERS - num_depression
    
    print(f"生成 {num_depression} 個憂鬱傾向用戶")
    print(f"生成 {num_normal} 個正常用戶")
    
    # 生成憂鬱用戶資料
    for i in range(num_depression):
        username = generate_username()
        display_name = generate_display_name()
        bio = random.choice(DEPRESSION_BIOS)
        
        num_posts = random.randint(MIN_POSTS, MAX_POSTS)
        posts = generate_posts(is_depression=True, num_posts=num_posts)
        dates = generate_post_dates(num_posts)
        
        for post, date in zip(posts, dates):
            all_data.append({
                "username": username,
                "display_name": display_name,
                "bio": bio,
                "post_content": post,
                "post_date": date,
                "depression_or_not": 1
            })
    
    # 生成正常用戶資料
    for i in range(num_normal):
        username = generate_username()
        display_name = generate_display_name()
        bio = random.choice(NORMAL_BIOS)
        
        num_posts = random.randint(MIN_POSTS, MAX_POSTS)
        posts = generate_posts(is_depression=False, num_posts=num_posts)
        dates = generate_post_dates(num_posts)
        
        for post, date in zip(posts, dates):
            all_data.append({
                "username": username,
                "display_name": display_name,
                "bio": bio,
                "post_content": post,
                "post_date": date,
                "depression_or_not": 0
            })
    
    # 建立 DataFrame
    df = pd.DataFrame(all_data)
    
    # 打亂順序
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    return df

# ----------------------
# 主程式
# ----------------------

if __name__ == "__main__":
    print("="*60)
    print("憂鬱傾向偵測模型 - 驗證集資料生成器")
    print("="*60)
    
    # 生成資料
    df = generate_validation_data()
    
    # 儲存資料
    df.to_excel(OUTPUT_PATH, index=False)
    print(f"\n驗證集資料已儲存至: {OUTPUT_PATH}")
    
    # 顯示統計資訊
    print("\n" + "="*60)
    print("資料統計")
    print("="*60)
    print(f"總記錄數: {len(df)}")
    print(f"總用戶數: {df['username'].nunique()}")
    print(f"憂鬱用戶數: {df[df['depression_or_not']==1]['username'].nunique()}")
    print(f"正常用戶數: {df[df['depression_or_not']==0]['username'].nunique()}")
    
    # 顯示每個用戶的平均貼文數
    posts_per_user = df.groupby('username').size()
    print(f"\n每位用戶平均貼文數: {posts_per_user.mean():.2f}")
    print(f"最多貼文數: {posts_per_user.max()}")
    print(f"最少貼文數: {posts_per_user.min()}")
    
    # 顯示範例資料
    print("\n" + "="*60)
    print("範例資料預覽")
    print("="*60)
    print(df.head(10).to_string())
    
    print("\n✓ 驗證集資料生成完成！")
