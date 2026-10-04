import numpy as np
import pandas as pd
from typing import List

def compute_user_numeric_features(post_texts, bio, post_dates):
    post_count = len(post_texts)
    avg_post_len = np.mean([len(str(t)) for t in post_texts]) if post_count > 0 else 0.0
    bio_len = len(str(bio))
    night_count = sum([1 for d in post_dates if 0 <= pd.to_datetime(d).hour <= 5])
    night_ratio = night_count / post_count if post_count > 0 else 0.0
    max_post_len = max([len(str(t)) for t in post_texts]) if post_count > 0 else 0.0
    min_post_len = min([len(str(t)) for t in post_texts]) if post_count > 0 else 0.0
    return [post_count, avg_post_len, bio_len, night_ratio, max_post_len, min_post_len]


def extract_logic_features(post_list: List[str], bio: str):
    neg_words = ["痛苦", "沒意義", "撐不住", "累", "空虛", "需要陪伴", "消失", "不想活", "孤獨", "絕望"]
    pos_words = ["開心", "期待", "感謝", "喜歡", "平靜", "快樂", "滿足"]

    neg_count = sum([sum([w in p for w in neg_words]) for p in post_list])
    pos_count = sum([sum([w in p for w in pos_words]) for p in post_list])
    logic_score = neg_count - pos_count
    bio_flag = 1 if any(w in bio for w in ["抑鬱", "憂鬱", "低潮", "焦慮"]) else 0
    total_count = neg_count + pos_count
    neg_ratio = neg_count / total_count if total_count > 0 else 0
    return [neg_count, pos_count, logic_score, bio_flag, neg_ratio]
