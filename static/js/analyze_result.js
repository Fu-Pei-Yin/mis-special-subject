document.addEventListener("DOMContentLoaded", () => {

  // --- 1. 柱狀圖（系統總體風險評估）動畫 ---
  document.querySelectorAll(".bar").forEach(bar => {
    const value = bar.dataset.height;
    if (value !== undefined) {
      setTimeout(() => {
        bar.style.height = `${value}%`;
      }, 150);
    }
  });

  // --- 2. 橫向進度條（核心行為洞察摘要）動畫 ---
  document.querySelectorAll(".fill").forEach(fill => {
    const value = fill.dataset.width;
    if (value !== undefined) {
      setTimeout(() => {
        fill.style.width = `${value}%`;
      }, 150);
    }
  });

});