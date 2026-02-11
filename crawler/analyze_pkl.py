import pickle

# 讀取 pkl 檔案
with open('C:/Users/USER/Desktop/課程/專題/threads_depression_web/crawler/cookies.pkl', 'rb') as f:
    
    data = pickle.load(f)

print(data) # 顯示載入的內容
