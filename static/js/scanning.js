document.addEventListener('DOMContentLoaded', () => {
    const statusText = document.querySelector('.scanning p');
    const messages = [
        "正在擷取近 14 天貼文...",
        "正在分析行為模式與語意特徵...",
        "正在計算情緒波動指標...",
        "AI 臨床模型交叉比對中...",
        "即將完成評估報告..."
    ];
    let currentIndex = 0;
    const interval = setInterval(() => {
        if (currentIndex < messages.length - 1) {
            currentIndex++;
            statusText.innerText = messages[currentIndex];
        } else {
            clearInterval(interval);
        }
    }, 500);
});