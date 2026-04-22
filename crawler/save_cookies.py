from selenium import webdriver
import pickle
import time
from pathlib import Path
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)
driver.get("https://www.threads.com/login")
print("請手動登入 Threads 網站...")
input("登入完成後按 Enter 繼續...")
ASE_DIR = Path(__file__).resolve().parent
COOKIE_PATH = BASE_DIR / "cookies.pkl"
cookies = driver.get_cookies()
with open(COOKIE_PATH, "wb") as f:
    pickle.dump(cookies, f)
print("Cookies 已保存到 cookies.pkl")
driver.quit()
