from crawler.threads_crawler import ThreadsCrawler
from inference.predict import predict_username

if __name__ == "__main__":
    username = input("請輸入 Threads 使用者名稱：")

    crawler = ThreadsCrawler(username)
    crawler.auto_login()
    #crawler.login()
    user_data = crawler.crawl_user()

    prob = predict_username(user_data)
    prob=prob*100
    print(f"{username} 憂鬱傾向機率：{prob:.4f}%")
