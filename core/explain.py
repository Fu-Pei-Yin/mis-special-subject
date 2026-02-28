"""
explain.py — 模型可說明性模組
將 UserLogicClassifier 從黑箱升級為灰箱/白箱模式

支援三種說明層次：
  1. 邏輯特徵貢獻（白箱：直接可解讀的規則）
  2. 貼文注意力權重（灰箱：Attention-based 重要性）
  3. Integrated Gradients（灰箱：梯度歸因，特徵級別）
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ──────────────────────────────────────────────
# 資料結構
# ──────────────────────────────────────────────

@dataclass
class PostExplanation:
    """單則貼文的說明"""
    index: int
    content: str
    attention_weight: float          # 模型給這則貼文的注意力比重
    neg_words_found: List[str]       # 命中的負向詞
    pos_words_found: List[str]       # 命中的正向詞
    risk_contribution: str           # "high" / "medium" / "low"


@dataclass
class FeatureExplanation:
    """數值與邏輯特徵的說明"""
    # 數值特徵
    post_count: int
    avg_post_len: float
    bio_len: int
    night_ratio: float
    max_post_len: float
    min_post_len: float
    # 邏輯特徵
    neg_count: int
    pos_count: int
    logic_score: int                 # neg - pos，越高風險越高
    bio_flag: int                    # 個人簡介是否含憂鬱關鍵詞
    neg_ratio: float
    # 梯度歸因（Integrated Gradients）
    ig_numeric: Optional[np.ndarray] = None   # shape (6,)
    ig_logic: Optional[np.ndarray] = None     # shape (5,)


@dataclass
class ExplainResult:
    """完整說明結果"""
    username: str
    risk_prob: float
    risk_level: str                          # "高風險" / "中等風險" / "低風險"
    top_posts: List[PostExplanation]
    feature_exp: FeatureExplanation
    summary: str                             # 自然語言摘要
    attention_weights: np.ndarray            # 所有貼文的注意力權重


# ──────────────────────────────────────────────
# 注意力鉤子（Attention Hook）
# ──────────────────────────────────────────────

class AttentionExtractor:
    """
    在 forward pass 期間攔截 Attention 權重。
    假設 UserLogicClassifier 的 Transformer encoder 名稱為 self.encoder
    或任何包含 MultiheadAttention 的子模組。
    """
    def __init__(self):
        self.attention_weights: List[torch.Tensor] = []
        self._hooks = []

    def register(self, model: torch.nn.Module):
        """為所有 MultiheadAttention 層掛上 hook"""
        self.attention_weights.clear()

        def hook_fn(module, input, output):
            # output 通常是 (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                self.attention_weights.append(output[1].detach().cpu())

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                h = module.register_forward_hook(hook_fn)
                self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_post_attention(self, num_posts: int) -> Optional[np.ndarray]:
        """
        整合所有 layer 的 attention，回傳 shape (num_posts,) 的重要性分數。
        取各 head、各 layer 的平均，再對 [CLS] token 的 attention 做 softmax。
        """
        if not self.attention_weights:
            return None
        # 取第一層對 sequence 維度的平均
        # attn shape: (batch, heads, seq_len, seq_len)
        attn = self.attention_weights[0]  # (1, heads, seq, seq)
        attn = attn.mean(dim=1)           # (1, seq, seq)
        attn = attn[0]                    # (seq, seq)
        # 用第 0 個 token（CLS 或第一個 post）對所有 posts 的 attention
        post_attn = attn[0, :num_posts].numpy()
        post_attn = post_attn / (post_attn.sum() + 1e-9)
        return post_attn


# ──────────────────────────────────────────────
# Integrated Gradients
# ──────────────────────────────────────────────

def integrated_gradients(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    target_input_keys: List[str],
    n_steps: int = 50,
) -> Dict[str, np.ndarray]:
    """
    對指定的輸入特徵計算 Integrated Gradients。
    回傳各 key 對應的歸因分數（與原始輸入同形狀）。
    """
    model.eval()
    attributions = {}

    for key in target_input_keys:
        original = batch[key].to(device).float()
        baseline = torch.zeros_like(original)  # baseline = 全零

        integrated_grad = torch.zeros_like(original)

        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            interpolated = (baseline + alpha * (original - baseline)).requires_grad_(True)

            # 建立完整 batch，只替換目標 key
            temp_batch = {k: v.to(device).float() for k, v in batch.items()}
            temp_batch[key] = interpolated

            logits = model(
                temp_batch["posts"],
                temp_batch["bio"],
                temp_batch["num"],
                temp_batch["logic"],
            )
            prob = torch.sigmoid(logits)
            prob.sum().backward()

            integrated_grad += interpolated.grad.detach()

        # IG = (original - baseline) * mean_gradient
        ig = (original - baseline) * integrated_grad / n_steps
        attributions[key] = ig.squeeze(0).cpu().numpy()

    return attributions


# ──────────────────────────────────────────────
# 邏輯特徵白箱說明
# ──────────────────────────────────────────────

NEG_WORDS = ["痛苦", "沒意義", "撐不住", "累", "空虛",
             "需要陪伴", "消失", "不想活", "孤獨", "絕望", "負面"]
POS_WORDS = ["開心", "期待", "感謝", "喜歡", "平靜", "快樂", "滿足"]
BIO_RISK_WORDS = ["抑鬱", "憂鬱", "低潮", "焦慮"]

NUM_FEAT_NAMES = ["貼文數", "平均貼文長度", "個人簡介長度",
                  "深夜發文比例", "最長貼文", "最短貼文"]
LOGIC_FEAT_NAMES = ["負向詞次數", "正向詞次數", "情緒分數(負-正)",
                    "簡介含憂鬱詞", "負向詞比例"]


def _risk_level(prob: float) -> str:
    if prob >= 0.65:
        return "高風險"
    elif prob >= 0.40:
        return "中等風險"
    return "低風險"


def _find_words_in_text(text: str, wordlist: List[str]) -> List[str]:
    return [w for w in wordlist if w in text]


def _generate_summary(
    username: str,
    prob: float,
    feature_exp: FeatureExplanation,
    top_posts: List[PostExplanation],
    ig_numeric: Optional[np.ndarray],
    ig_logic: Optional[np.ndarray],
) -> str:
    level = _risk_level(prob)
    lines = [
        f"【使用者：{username}】憂鬱風險評估摘要",
        f"預測機率：{prob:.1%}　風險等級：{level}",
        "",
        "─── 白箱說明（邏輯規則層）───",
        f"• 共 {feature_exp.post_count} 則貼文，負向詞出現 {feature_exp.neg_count} 次，"
        f"正向詞 {feature_exp.pos_count} 次，情緒分數 {feature_exp.logic_score:+d}",
        f"• 深夜（00-05時）發文比例：{feature_exp.night_ratio:.1%}",
    ]
    if feature_exp.bio_flag:
        lines.append("• ⚠️  個人簡介含憂鬱相關關鍵詞（抑鬱/憂鬱/低潮/焦慮）")

    if top_posts:
        lines += ["", "─── 灰箱說明（Attention 層）最具影響力的貼文 ───"]
        for i, p in enumerate(top_posts[:3], 1):
            neg_str = "、".join(p.neg_words_found) or "無"
            lines.append(
                f"  {i}. 注意力權重 {p.attention_weight:.2%}｜"
                f"負向詞：{neg_str}｜"
                f"內容節錄：「{str(p.content)[:40]}…」"
            )

    if ig_numeric is not None:
        top_idx = np.argsort(np.abs(ig_numeric))[::-1][:3]
        lines += ["", "─── 灰箱說明（Integrated Gradients，數值特徵）───"]
        for idx in top_idx:
            direction = "↑ 升高風險" if ig_numeric[idx] > 0 else "↓ 降低風險"
            lines.append(
                f"  • {NUM_FEAT_NAMES[idx]}：歸因分數 {ig_numeric[idx]:+.4f}  {direction}"
            )

    if ig_logic is not None:
        top_idx = np.argsort(np.abs(ig_logic))[::-1][:3]
        lines += ["", "─── 灰箱說明（Integrated Gradients，邏輯特徵）───"]
        for idx in top_idx:
            direction = "↑ 升高風險" if ig_logic[idx] > 0 else "↓ 降低風險"
            lines.append(
                f"  • {LOGIC_FEAT_NAMES[idx]}：歸因分數 {ig_logic[idx]:+.4f}  {direction}"
            )

    return "\n".join(lines)


# ──────────────────────────────────────────────
# 主要 Explainer 類別
# ──────────────────────────────────────────────

class DepressionModelExplainer:
    """
    包裝 UserLogicClassifier，提供多層次可說明性。

    使用方式：
        explainer = DepressionModelExplainer(model, embed_model, device)
        result = explainer.explain(user_data)
        print(result.summary)
        explainer.plot_attention(result)    # 貼文注意力長條圖
        explainer.plot_features(result)     # 特徵歸因長條圖
    """

    def __init__(
        self,
        model: torch.nn.Module,
        embed_model,
        device: torch.device,
        use_integrated_gradients: bool = True,
        ig_steps: int = 50,
    ):
        self.model = model
        self.embed_model = embed_model
        self.device = device
        self.use_ig = use_integrated_gradients
        self.ig_steps = ig_steps
        self._attn_extractor = AttentionExtractor()

    # ── 核心方法 ──────────────────────────────

    def explain(self, user_data: dict) -> ExplainResult:
        """
        對單一使用者產生完整說明。

        Args:
            user_data: 與 predict_username 相同格式的字典

        Returns:
            ExplainResult 物件，含機率、風險等級、貼文說明、特徵說明、摘要
        """
        from core.dataset import (
            ThreadsInferenceDataset,
            compute_user_numeric_features,
            extract_logic_features,
            MAX_POSTS,
        )

        username = user_data.get("username", "unknown")
        bio = user_data.get("bio", "")
        posts_raw = user_data.get("posts", [])[:MAX_POSTS]
        post_texts = [p["content"] for p in posts_raw]
        post_dates = [p["post_date"] for p in posts_raw]

        # 1. 建立 Dataset & DataLoader
        dataset = ThreadsInferenceDataset(
            users_data=[user_data], embed_model=self.embed_model
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        batch = next(iter(loader))

        # 2. Attention hook
        self._attn_extractor.register(self.model)
        self.model.eval()
        with torch.no_grad():
            posts_t = batch["posts"].to(self.device)
            bio_t   = batch["bio"].to(self.device)
            num_t   = batch["num"].to(self.device)
            logic_t = batch["logic"].to(self.device)
            logits  = self.model(posts_t, bio_t, num_t, logic_t)
            prob    = torch.sigmoid(logits).item()
        self._attn_extractor.remove()

        # 3. 取得注意力權重
        attn_weights = self._attn_extractor.get_post_attention(len(post_texts))
        if attn_weights is None:
            # fallback：用均勻分布
            n = max(len(post_texts), 1)
            attn_weights = np.ones(n) / n

        # 4. 貼文說明（PostExplanation）
        post_explanations = []
        for i, text in enumerate(post_texts):
            w = float(attn_weights[i]) if i < len(attn_weights) else 0.0
            neg_found = _find_words_in_text(str(text), NEG_WORDS)
            pos_found = _find_words_in_text(str(text), POS_WORDS)
            if w >= 0.10 or neg_found:
                risk = "high"
            elif w >= 0.05:
                risk = "medium"
            else:
                risk = "low"
            post_explanations.append(PostExplanation(
                index=i,
                content=str(text),
                attention_weight=w,
                neg_words_found=neg_found,
                pos_words_found=pos_found,
                risk_contribution=risk,
            ))
        top_posts = sorted(post_explanations, key=lambda p: p.attention_weight, reverse=True)

        # 5. 特徵說明（FeatureExplanation）
        num_raw = compute_user_numeric_features(post_texts, bio, post_dates)
        logic_raw = extract_logic_features(post_texts, bio)

        # 6. Integrated Gradients
        ig_numeric = ig_logic = None
        if self.use_ig:
            try:
                attributions = integrated_gradients(
                    model=self.model,
                    batch={k: v.clone() for k, v in batch.items()},
                    device=self.device,
                    target_input_keys=["num", "logic"],
                    n_steps=self.ig_steps,
                )
                ig_numeric = attributions.get("num")
                ig_logic   = attributions.get("logic")
            except Exception as e:
                print(f"[Explainer] Integrated Gradients 計算失敗，跳過：{e}")

        feature_exp = FeatureExplanation(
            post_count=num_raw[0],
            avg_post_len=num_raw[1],
            bio_len=num_raw[2],
            night_ratio=num_raw[3],
            max_post_len=num_raw[4],
            min_post_len=num_raw[5],
            neg_count=logic_raw[0],
            pos_count=logic_raw[1],
            logic_score=logic_raw[2],
            bio_flag=logic_raw[3],
            neg_ratio=logic_raw[4],
            ig_numeric=ig_numeric,
            ig_logic=ig_logic,
        )

        # 7. 自然語言摘要
        summary = _generate_summary(
            username, prob, feature_exp, top_posts, ig_numeric, ig_logic
        )

        return ExplainResult(
            username=username,
            risk_prob=prob,
            risk_level=_risk_level(prob),
            top_posts=top_posts,
            feature_exp=feature_exp,
            summary=summary,
            attention_weights=attn_weights,
        )

    # ── 視覺化方法 ────────────────────────────

    def plot_attention(self, result: ExplainResult, max_posts: int = 15):
        """貼文注意力長條圖（需要 matplotlib）"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
        except ImportError:
            print("[Explainer] 請安裝 matplotlib 以使用視覺化功能")
            return

        posts = result.top_posts[:max_posts]
        labels = [f"貼文{p.index+1}" for p in posts]
        weights = [p.attention_weight for p in posts]
        colors = ["#e74c3c" if p.risk_contribution == "high"
                  else "#f39c12" if p.risk_contribution == "medium"
                  else "#2ecc71" for p in posts]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(labels, weights, color=colors)
        ax.set_xlabel("注意力權重")
        ax.set_title(
            f"[{result.username}] 各貼文注意力權重\n"
            f"風險機率：{result.risk_prob:.1%}（{result.risk_level}）"
        )
        ax.invert_yaxis()

        # 在條形右側標出命中的負向詞
        for bar, post in zip(bars, posts):
            if post.neg_words_found:
                ax.text(
                    bar.get_width() + 0.002,
                    bar.get_y() + bar.get_height() / 2,
                    "⚠ " + "、".join(post.neg_words_found[:3]),
                    va="center", fontsize=8, color="#c0392b"
                )

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#e74c3c", label="高貢獻"),
            Patch(facecolor="#f39c12", label="中貢獻"),
            Patch(facecolor="#2ecc71", label="低貢獻"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")
        plt.tight_layout()
        plt.show()

    def plot_features(self, result: ExplainResult):
        """特徵 Integrated Gradients 歸因長條圖"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
        except ImportError:
            print("[Explainer] 請安裝 matplotlib 以使用視覺化功能")
            return

        fe = result.feature_exp
        if fe.ig_numeric is None and fe.ig_logic is None:
            print("[Explainer] Integrated Gradients 資料不可用，無法繪圖")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"[{result.username}] 特徵歸因（Integrated Gradients）\n"
            f"風險機率：{result.risk_prob:.1%}（{result.risk_level}）",
            fontsize=13
        )

        def _bar_plot(ax, names, values, title):
            colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]
            ax.barh(names, values, color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(title)
            ax.set_xlabel("歸因分數（正 = 升高風險，負 = 降低風險）")
            ax.invert_yaxis()

        if fe.ig_numeric is not None:
            _bar_plot(axes[0], NUM_FEAT_NAMES, fe.ig_numeric.tolist(), "數值特徵")
        else:
            axes[0].set_visible(False)

        if fe.ig_logic is not None:
            _bar_plot(axes[1], LOGIC_FEAT_NAMES, fe.ig_logic.tolist(), "邏輯特徵")
        else:
            axes[1].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_logic_radar(self, result: ExplainResult):
        """雷達圖：白箱邏輯指標總覽"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            import math
            matplotlib.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
        except ImportError:
            print("[Explainer] 請安裝 matplotlib 以使用視覺化功能")
            return

        fe = result.feature_exp
        categories = ["負向詞\n次數", "深夜\n發文比", "負向詞\n比例",
                      "簡介\n含風險詞", "情緒\n分數"]
        # 正規化到 0–1（越高越危險）
        max_neg = max(fe.neg_count, 1)
        vals = [
            min(fe.neg_count / 20, 1.0),
            fe.night_ratio,
            fe.neg_ratio,
            float(fe.bio_flag),
            min(max(fe.logic_score / 20, 0.0), 1.0),
        ]
        N = len(categories)
        angles = [n / N * 2 * math.pi for n in range(N)]
        angles += angles[:1]
        vals += vals[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, vals, "o-", linewidth=2, color="#e74c3c")
        ax.fill(angles, vals, alpha=0.25, color="#e74c3c")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(
            f"[{result.username}] 白箱邏輯指標雷達圖\n"
            f"風險機率：{result.risk_prob:.1%}（{result.risk_level}）",
            pad=20
        )
        plt.tight_layout()
        plt.show()


# ──────────────────────────────────────────────
# 便利函式：直接取代原有 predict_username
# ──────────────────────────────────────────────

def predict_with_explanation(user_data: dict, verbose: bool = True) -> ExplainResult:
    """
    取代原有 predict_username，同時回傳預測結果與完整說明。

    Args:
        user_data:  與原有格式相同的使用者資料字典
        verbose:    是否直接印出摘要

    Returns:
        ExplainResult（含 .risk_prob, .risk_level, .summary 等欄位）

    範例：
        result = predict_with_explanation(user_data)
        print(result.risk_prob)      # 0.78
        print(result.summary)        # 自然語言說明
        explainer.plot_attention(result)
        explainer.plot_features(result)
        explainer.plot_logic_radar(result)
    """
    import torch
    from sentence_transformers import SentenceTransformer
    from model.model import UserLogicClassifier

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "C:/Users/USER/Desktop/課程/專題/threads_depression_web/model/fold_1_best.pt"

    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    model = UserLogicClassifier(
        embed_dim=384,
        num_num_feats=6,
        num_logic_feats=5,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    explainer = DepressionModelExplainer(
        model=model,
        embed_model=embed_model,
        device=DEVICE,
        use_integrated_gradients=True,
        ig_steps=50,
    )
    result = explainer.explain(user_data)

    if verbose:
        print(result.summary)

    return result