document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('form');
    const submitBtn = form.querySelector('button');
    const input = form.querySelector('input');

    form.addEventListener('submit', (e) => {
        // 檢查是否輸入內容
        if (input.value.trim() === "") {
            e.preventDefault();
            alert("請輸入使用者名稱");
            return;
        }

        // 按鈕點擊後的視覺反饋
        submitBtn.innerHTML = "正在分析中...";
        submitBtn.style.opacity = "0.7";
        submitBtn.disabled = true;

        // 這裡可以加入動畫效果，例如淡出頁面
        console.log(`正在提交分析請求：${input.value}`);
    });

    // 增加一個簡單的打字機效果或滑鼠追蹤光暈 (選配)
    document.addEventListener('mousemove', (e) => {
        const x = e.clientX / window.innerWidth;
        const y = e.clientY / window.innerHeight;
        // 可以在背景移動時微調漸層位置
    });
});