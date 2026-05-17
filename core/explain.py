"""
explain.py — BiLSTM 可說明性模組（已適配 GeneralBiLSTM）

說明層次：
  1. 數值特徵貢獻（白箱）
  2. 貼文重要性（使用 embedding norm 近似）
  3. Integrated Gradients（數值特徵）
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


# ──────────────────────────────────────────────
# 資料結構
# ──────────────────────────────────────────────

@dataclass
class PostExplanation:
    index: int
    content: str
    importance: float
    risk_contribution: str


@dataclass
class FeatureExplanation:
    post_count: int
    avg_post_len: float
    bio_len: int
    night_ratio: float
    max_post_len: float
    min_post_len: float
    ig_numeric: Optional[np.ndarray] = None


@dataclass
class ExplainResult:
    username: str
    risk_prob: float
    risk_level: str
    top_posts: List[PostExplanation]
    feature_exp: FeatureExplanation
    summary: str
    post_importance: np.ndarray


# ──────────────────────────────────────────────
# Integrated Gradients（只對 numeric）
# ──────────────────────────────────────────────

def integrated_gradients(
    model,
    batch,
    device,
    n_steps=30
):
    model.eval()
    original = batch["num"].to(device).float()
    baseline = torch.zeros_like(original)

    integrated_grad = torch.zeros_like(original)

    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        interpolated = (baseline + alpha * (original - baseline)).requires_grad_(True)

        logits = model(
            batch["posts"].to(device),
            batch["bio"].to(device),
            interpolated
        )

        prob = torch.sigmoid(logits)
        prob.backward(torch.ones_like(prob))

        integrated_grad += interpolated.grad.detach()

    ig = (original - baseline) * integrated_grad / n_steps
    return ig.squeeze(0).cpu().numpy()


# ──────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────

def _risk_level(prob: float) -> str:
    if prob >= 0.5:
        return "高風險"
    elif prob >= 0.25:
        return "需觀察"
    return "低風險"


NUM_FEAT_NAMES = [
    "貼文數", "平均貼文長度", "個人簡介長度",
    "深夜發文比例", "最長貼文", "最短貼文"
]


def _generate_summary(username, prob, feature_exp, top_posts, ig_numeric):
    level = _risk_level(prob)

    lines = [
        f"【使用者：{username}】憂鬱風險評估",
        f"預測機率：{prob:.1%}（{level}）",
        "",
        "─── 行為特徵（白箱）───",
        f"• 貼文數：{feature_exp.post_count}",
        f"• 平均貼文長度：{feature_exp.avg_post_len:.1f}",
        f"• 深夜發文比例：{feature_exp.night_ratio:.1%}",
    ]

    if top_posts:
        lines.append("\n─── 重要貼文（LSTM 近似）───")
        for i, p in enumerate(top_posts[:3], 1):
            lines.append(
                f"{i}. 重要度 {p.importance:.2%}｜"
                f"{p.content[:40]}..."
            )

    if ig_numeric is not None:
        idxs = np.argsort(np.abs(ig_numeric))[::-1][:3]
        lines.append("\n─── 關鍵特徵（IG）───")
        for i in idxs:
            direction = "↑風險" if ig_numeric[i] > 0 else "↓風險"
            lines.append(
                f"{NUM_FEAT_NAMES[i]}：{ig_numeric[i]:+.4f} {direction}"
            )

    return "\n".join(lines)


# ──────────────────────────────────────────────
# 主 Explainer
# ──────────────────────────────────────────────

class DepressionModelExplainer:

    def __init__(self, model, embed_model, device):
        self.model = model
        self.embed_model = embed_model
        self.device = device

    def explain(self, user_data: dict) -> ExplainResult:

        from core.dataset import (
            ThreadsInferenceDataset,
            compute_user_numeric_features,
            MAX_POSTS
        )

        username = user_data.get("username", "unknown")
        bio = user_data.get("bio", "")
        posts_raw = user_data.get("posts", [])[:MAX_POSTS]

        post_texts = [p["content"] for p in posts_raw]
        post_dates = [p["post_date"] for p in posts_raw]

        dataset = ThreadsInferenceDataset(
            users_data=[user_data],
            embed_model=self.embed_model
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        batch = next(iter(loader))

        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                batch["posts"].to(self.device),
                batch["bio"].to(self.device),
                batch["num"].to(self.device)
            )
            prob = torch.sigmoid(logits).item()

        # ───── 貼文重要性（embedding norm）─────
        post_embs = batch["posts"][0].cpu().numpy()
        norms = np.linalg.norm(post_embs, axis=1)
        if norms.sum() > 0:
            norms = norms / norms.sum()
        else:
            norms = np.ones_like(norms) / len(norms)

        post_exps = []
        for i, text in enumerate(post_texts):
            w = float(norms[i])
            if w > 0.1:
                level = "high"
            elif w > 0.05:
                level = "medium"
            else:
                level = "low"

            post_exps.append(PostExplanation(
                index=i,
                content=str(text),
                importance=w,
                risk_contribution=level
            ))

        top_posts = sorted(post_exps, key=lambda x: x.importance, reverse=True)

        # ───── 數值特徵 ─────
        num_raw = compute_user_numeric_features(post_texts, bio, post_dates)

        # ───── IG ─────
        try:
            ig_numeric = integrated_gradients(
                self.model,
                {k: v.clone() for k, v in batch.items()},
                self.device
            )
        except:
            ig_numeric = None

        feature_exp = FeatureExplanation(
            post_count=num_raw[0],
            avg_post_len=num_raw[1],
            bio_len=num_raw[2],
            night_ratio=num_raw[3],
            max_post_len=num_raw[4],
            min_post_len=num_raw[5],
            ig_numeric=ig_numeric
        )

        summary = _generate_summary(
            username, prob, feature_exp, top_posts, ig_numeric
        )

        return ExplainResult(
            username=username,
            risk_prob=prob,
            risk_level=_risk_level(prob),
            top_posts=top_posts,
            feature_exp=feature_exp,
            summary=summary,
            post_importance=norms
        )