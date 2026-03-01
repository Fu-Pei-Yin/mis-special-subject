document.addEventListener('DOMContentLoaded', () => {
    const statusText = document.querySelector('.scanning p');

    // 從 HTML 取得 username（由 Django template 注入）
    const username = document.querySelector('[data-username]')?.dataset.username
                  || document.querySelector('input[name="username"]')?.value
                  || '';

    if (!username) return;

    const progressUrl = `/progress/${encodeURIComponent(username)}/`;  // 對應 urls.py

    let pollInterval;

    function poll() {
        fetch(progressUrl)
            .then(res => res.json())
            .then(data => {
                // 顯示最新一則訊息（過濾空白行）
                const newMsgs = (data.messages || []).filter(m => m.trim());
                if (newMsgs.length > 0) {
                    statusText.innerText = newMsgs[newMsgs.length - 1];
                }

                if (data.done) {
                    clearInterval(pollInterval);
                    statusText.innerText = '✓ 資料擷取完成，正在產生報告...';
                    // 提交表單跳轉至 result 頁
                    document.getElementById('forwardForm').submit();
                }
            })
            .catch(() => {
                // 網路錯誤靜默處理，繼續輪詢
            });
    }

    // 每 1.5 秒輪詢一次
    pollInterval = setInterval(poll, 1500);
    poll(); // 立即執行一次
});