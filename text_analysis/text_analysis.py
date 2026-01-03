# 03_text_analysis.py
import pandas as pd
import jieba
from collections import Counter

INPUT_PATH = "high_risk_posts.csv"
STOPWORDS_PATH = r"C:\Users\eva19\Desktop\graduate_project\程式碼\詞頻分析\stopwords_zh.txt"

# ==============================
# 讀取資料
# ==============================
df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

texts = (
    df["post_content"].fillna("").astype(str) + " " +
    df["bio"].fillna("").astype(str)
)

# ==============================
# 載入停用詞
# ==============================
with open(STOPWORDS_PATH, encoding="utf-8") as f:
    stopwords = set(w.strip() for w in f if w.strip())

# ==============================
# 斷詞 + 詞頻
# ==============================
counter = Counter()

for text in texts:
    for w in jieba.cut(text):
        w = w.strip()
        if len(w) < 2:
            continue
        if w in stopwords:
            continue
        counter[w] += 1

# ==============================
# 輸出 top 100 詞頻為單行 query_list
# ==============================
top_words = counter.most_common(100)
query_list = [w for w, c in top_words]

# 單行輸出
print("query_list =", query_list)