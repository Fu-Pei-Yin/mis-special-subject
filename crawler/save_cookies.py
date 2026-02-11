from selenium import webdriver
import pickle
import time
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)
driver.get("https://www.threads.com/login")
print("請手動登入 Threads 網站...")
input("登入完成後按 Enter 繼續...")
cookies = driver.get_cookies()
with open("C:/Users/USER/Desktop/課程/專題/threads_depression_web/crawler/cookies.pkl", "wb") as f:
    pickle.dump(cookies, f)
print("Cookies 已保存到 cookies.pkl")
driver.quit()
