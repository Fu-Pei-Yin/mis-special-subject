/* ═══════════════════════════════════════════════════════
   suggest.js  ─  MINDGUARD AI 臨床診斷報告互動邏輯
   包含：頁面導航、白箱可解釋性分析動態效果
   ═══════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {

  /* ── 1. 頁面導航按鈕 ─────────────────────────── */
  const closeBtn = document.querySelector(".close-btn");
  const backBtn  = document.querySelector(".back-btn");
  if (closeBtn) closeBtn.onclick = () => window.history.back();
  if (backBtn)  backBtn.onclick  = () => window.history.back();


  /* ── 2. 白箱指標：計數器動畫 ─────────────────── */
  /**
   * 讓 .wb-cell-value 的數字從 0 往上滾動至最終值。
   * 支援整數與帶 % / + 前綴格式。
   */
  function animateCounter(el) {
    const rawText  = el.textContent.trim();
    const hasPlus  = rawText.startsWith("+");
    const hasPct   = rawText.endsWith("%");
    const numStr   = rawText.replace(/[+%]/g, "");
    const target   = parseFloat(numStr);

    if (isNaN(target)) return;

    const duration = 900;   // ms
    const fps      = 60;
    const steps    = Math.round(duration / (1000 / fps));
    let   current  = 0;

    const timer = setInterval(() => {
      current++;
      const progress = current / steps;
      // Ease-out cubic
      const eased    = 1 - Math.pow(1 - progress, 3);
      const value    = Math.round(eased * target);
      const prefix   = hasPlus && value > 0 ? "+" : "";
      const suffix   = hasPct ? "%" : "";
      el.textContent = `${prefix}${value}${suffix}`;

      if (current >= steps) {
        clearInterval(timer);
        el.textContent = rawText; // 確保最終值精確
      }
    }, 1000 / fps);
  }

  // 使用 IntersectionObserver 讓指標進入視窗才開始動畫
  const wbCells = document.querySelectorAll(".wb-cell-value");
  if (wbCells.length > 0 && "IntersectionObserver" in window) {
    const counterObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          counterObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.4 });

    wbCells.forEach(cell => counterObserver.observe(cell));
  }


  /* ── 3. 指標格 Tooltip ────────────────────────── */
  const CELL_TIPS = {
    "負向詞次數": "過去分析期間，貼文中出現的負向情緒詞彙總次數。\n閾值：≥2 警告，≥5 危險。",
    "正向詞次數": "貼文中出現的正向情緒詞彙總次數。\n越高代表正向表達越豐富。",
    "情緒分數（負－正）": "負向詞次數減去正向詞次數的差值。\n正值越大表示負面情緒傾向越強。",
    "負向詞比例": "負向詞佔所有情緒詞的百分比。\n閾值：≥25% 警告，≥50% 危險。",
    "深夜發文比例": "00:00–04:59 之間發文佔全部貼文的百分比。\n閾值：≥25% 警告，≥50% 危險。",
    "貼文總數": "分析期間內的有效貼文總數，作為樣本量參考。",
  };

  // 建立 Tooltip DOM
  const tooltip = document.createElement("div");
  tooltip.className = "wb-tooltip";
  document.body.appendChild(tooltip);

  let tooltipTimer = null;

  function showTooltip(e, text) {
    clearTimeout(tooltipTimer);
    tooltip.textContent = text;
    positionTooltip(e);
    tooltip.classList.add("visible");
  }

  function hideTooltip() {
    tooltipTimer = setTimeout(() => tooltip.classList.remove("visible"), 120);
  }

  function positionTooltip(e) {
    const pad = 12;
    let x = e.clientX + pad;
    let y = e.clientY - tooltip.offsetHeight - pad;

    if (x + tooltip.offsetWidth > window.innerWidth - 10)
      x = e.clientX - tooltip.offsetWidth - pad;
    if (y < 8)
      y = e.clientY + pad;

    tooltip.style.left = x + "px";
    tooltip.style.top  = y + "px";
  }

  document.querySelectorAll(".wb-cell").forEach(cell => {
    const labelEl = cell.querySelector(".wb-cell-label");
    if (!labelEl) return;
    const key = labelEl.textContent.trim();
    const tip = CELL_TIPS[key];
    if (!tip) return;

    cell.style.cursor = "help";
    cell.addEventListener("mouseenter", e => showTooltip(e, tip));
    cell.addEventListener("mousemove",  e => positionTooltip(e));
    cell.addEventListener("mouseleave", hideTooltip);
  });


  /* ── 4. 貼文關鍵詞：文字內高亮 ──────────────── */
  /**
   * 讀取 data-neg / data-pos 屬性，
   * 在 .wb-post-content 的文字中標記對應詞彙。
   */
  document.querySelectorAll(".wb-post-content").forEach(contentEl => {
    const negList = (contentEl.dataset.neg || "").split(",").map(w => w.trim()).filter(Boolean);
    const posList = (contentEl.dataset.pos || "").split(",").map(w => w.trim()).filter(Boolean);

    if (!negList.length && !posList.length) return;

    let html = contentEl.textContent;

    // 先處理負向（優先於正向）
    negList.forEach(word => {
      html = html.replace(
        new RegExp(escapeRegExp(word), "g"),
        `<mark class="highlight-neg">${word}</mark>`
      );
    });

    posList.forEach(word => {
      html = html.replace(
        new RegExp(escapeRegExp(word), "g"),
        `<mark class="highlight-pos">${word}</mark>`
      );
    });

    contentEl.innerHTML = html;
  });

  function escapeRegExp(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }


  /* ── 5. 貼文列表：展開 / 收合 ────────────────── */
  const postList = document.querySelector(".wb-post-list");

  if (postList) {
    const COLLAPSED_HEIGHT = 420; // px，與 CSS max-height 對齊
    const allPosts = postList.querySelectorAll(".wb-post-item");

    // 若貼文數量不多，不顯示切換按鈕
    if (allPosts.length > 4) {
      const toggleBtn = document.createElement("button");
      toggleBtn.className = "wb-toggle-btn";
      toggleBtn.innerHTML = `<span class="toggle-icon">▼</span> 展開全部 ${allPosts.length} 則貼文`;
      postList.after(toggleBtn);

      let expanded = false;

      toggleBtn.addEventListener("click", () => {
        expanded = !expanded;
        if (expanded) {
          postList.style.maxHeight = postList.scrollHeight + "px";
          toggleBtn.innerHTML = `<span class="toggle-icon">▼</span> 收合貼文`;
          toggleBtn.classList.add("expanded");
        } else {
          postList.style.maxHeight = COLLAPSED_HEIGHT + "px";
          toggleBtn.innerHTML = `<span class="toggle-icon">▼</span> 展開全部 ${allPosts.length} 則貼文`;
          toggleBtn.classList.remove("expanded");
          // 滾動回貼文區塊頂部
          postList.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      });
    }
  }


  /* ── 6. 貼文卡片：Staggered 淡入 ────────────── */
  const postItems = document.querySelectorAll(".wb-post-item");
  postItems.forEach((item, i) => {
    item.style.animationDelay = `${i * 60}ms`;
  });


  /* ── 7. 風險詞標籤：Hover 震動效果 ──────────── */
  document.querySelectorAll(".risk-word").forEach(rw => {
    rw.addEventListener("mouseenter", () => {
      rw.style.animation = "none";
      // 強制 reflow
      void rw.offsetWidth;
      rw.style.animation = "shakeRisk 0.35s ease";
    });
  });

  // 注入 shakeRisk keyframe（避免污染 CSS 檔案）
  if (!document.getElementById("wb-extra-styles")) {
    const style = document.createElement("style");
    style.id = "wb-extra-styles";
    style.textContent = `
      @keyframes shakeRisk {
        0%  { transform: translateX(0); }
        25% { transform: translateX(-3px); }
        50% { transform: translateX(3px); }
        75% { transform: translateX(-2px); }
        100%{ transform: translateX(0); }
      }
    `;
    document.head.appendChild(style);
  }


  /* ── 8. 白箱區塊：整體 Scroll 淡入 ─────────── */
  const wbSection = document.querySelector(".whitebox-section");
  if (wbSection && "IntersectionObserver" in window) {
    wbSection.style.opacity = "0";
    wbSection.style.transform = "translateY(20px)";
    wbSection.style.transition = "opacity 0.6s ease, transform 0.6s ease";

    const sectionObserver = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        wbSection.style.opacity   = "1";
        wbSection.style.transform = "translateY(0)";
        sectionObserver.disconnect();
      }
    }, { threshold: 0.1 });

    sectionObserver.observe(wbSection);
  }

}); // End DOMContentLoaded