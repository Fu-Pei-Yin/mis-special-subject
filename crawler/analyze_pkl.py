import pickle
from pathlib import Path
# 讀取 pkl 檔案
BASE_DIR = Path(__file__).resolve().parent
COOKIE_PATH = BASE_DIR / "cookies.pkl"
with open(COOKIE_PATH, 'rb') as f:
    
    data = pickle.load(f)

print(data) # 顯示載入的內容
