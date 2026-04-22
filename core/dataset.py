import torch
import numpy as np
import joblib
from torch.utils.data import Dataset
from pathlib import Path
EMBED_DIM = 384
MAX_POSTS = 32
BASE_DIR = Path(__file__).resolve().parent
NUM_SCALER_PATH   = BASE_DIR / "../model/numerical_scaler.pkl"

numerical_scaler = joblib.load(NUM_SCALER_PATH)

# ══════════════════════════════════════════════════════════════
#  詞庫：模組層級常數，供 dataset.py 內部與 views.py 共同使用
#  如需新增或修改詞彙，只需改這裡即可，訓練與顯示端同步生效
# ══════════════════════════════════════════════════════════════
NEG_WORDS = [
    "悲傷", "憂鬱", "哭泣", "孤獨", "自殺", "罪惡感", "焦慮", "恐慌", "壓力",
    "空虛", "絕望", "害怕", "困惑", "心煩意亂", "易怒", "不安全感", "厭惡",
    "妄想", "耐受性",

    "失眠", "睡眠問題", "疲累", "嗜睡", "噩夢", "疼痛", "頭痛", "偏頭痛",
    "噁心", "嘔吐", "食慾問題", "體重波動", "昏沉", "頭暈", "抽搐",

    "自我懷疑", "意識到", "自責", "恨", "討厭", "無價值感",
    "失敗", "懷疑", "羞愧",

    "死亡", "創傷", "虐待", "侵犯", "騷擾", "離婚", "意外", "酷刑",
    "痛苦", "戰爭",

    "社交退縮", "離開", "分手", "忽視", "煩躁", "難以信任", "憤怒",
    "爭吵", "孤立",

    "混蛋", "爛透了", "垃圾", "廢話", "幹", "靠北", "媽的", "婊子", "該死",

    "心理治療", "抗憂鬱劑", "精神科醫生", "藥物", "副作用", "住院", "血清素"
]
POS_WORDS = [
    "開心", "太棒了", "美麗的", "享受", "樂趣", "喜愛", "完美", "興奮", "笑容",

    "運動", "健身房", "動力", "康復", "療癒", "瑜珈", "希望", "支持",
    "放鬆", "因應",

    "朋友", "遊戲", "音樂", "電影", "派對", "寵物", "貓", "狗",
    "度假", "閱讀", "藝術", "社交", "教會",

    "家庭", "家", "目標", "成功", "未來", "參與", "分享"
]
BIO_RISK_WORDS = [
    "悲傷", "憂鬱", "哭泣", "孤獨", "自殺", "罪惡感", "焦慮", "恐慌", "壓力",
    "空虛", "絕望", "害怕", "困惑", "心煩意亂", "易怒", "不安全感", "厭惡",
    "妄想", "耐受性",

    "失眠", "睡眠問題", "疲累", "嗜睡", "噩夢", "疼痛", "頭痛", "偏頭痛",
    "噁心", "嘔吐", "食慾問題", "體重波動", "昏沉", "頭暈", "抽搐",

    "自我懷疑", "意識到", "自責", "恨", "討厭", "無價值感",
    "失敗", "懷疑", "羞愧",

    "死亡", "創傷", "虐待", "侵犯", "騷擾", "離婚", "意外", "酷刑",
    "痛苦", "戰爭",

    "社交退縮", "離開", "分手", "忽視", "煩躁", "難以信任", "憤怒",
    "爭吵", "孤立",

    "混蛋", "爛透了", "垃圾", "廢話", "幹", "靠北", "媽的", "婊子", "該死",

    "心理治療", "抗憂鬱劑", "精神科醫生", "藥物", "副作用", "住院", "血清素"
]

def extract_logic_features(post_list, bio): 
    """ 回傳 5 維邏輯特徵： [neg_count, pos_count, logic_score, bio_flag(0/1), neg_ratio(0~1)] 詞庫統一引用模組層級常數 NEG_WORDS / POS_WORDS / BIO_RISK_WORDS """ 
    neg_count = sum(sum(w in p for w in NEG_WORDS) for p in post_list) 
    pos_count = sum(sum(w in p for w in POS_WORDS) for p in post_list) 
    logic_score = neg_count - pos_count 
    bio_flag = 1 if any(w in str(bio) for w in BIO_RISK_WORDS) else 0 
    total = neg_count + pos_count 
    neg_ratio = neg_count / total if total > 0 else 0.0 
    return [neg_count, pos_count, logic_score, bio_flag, neg_ratio]
def compute_user_numeric_features(post_texts, bio, post_dates):
    """
    回傳 6 維數值特徵：
      [post_count, avg_post_len, bio_len, night_ratio(0~1), max_post_len, min_post_len]

    post_dates 需為字串格式 "YYYY-MM-DD HH:MM"（views.py 的 _fmt_date 已確保）
    """
    post_count = len(post_texts)
    avg_post_len = np.mean([len(str(t)) for t in post_texts]) if post_count > 0 else 0.0
    bio_len      = len(str(bio))

    night_count = 0
    for d in post_dates:
        try:
            hour = int(str(d).split(" ")[1].split(":")[0])
            if 0 <= hour <= 5:
                night_count += 1
        except (IndexError, ValueError):
            pass  # 格式異常時跳過，不影響其他計算

    night_ratio  = night_count / post_count if post_count > 0 else 0.0
    max_post_len = max([len(str(t)) for t in post_texts]) if post_count > 0 else 0.0
    min_post_len = min([len(str(t)) for t in post_texts]) if post_count > 0 else 0.0

    return [post_count, avg_post_len, bio_len, night_ratio, max_post_len, min_post_len]




class ThreadsInferenceDataset(Dataset):
    def __init__(self, users_data, embed_model):
        self.post_embs, self.bio_embs, self.numeric_feats= [], [], []

        for user in users_data:
            if not isinstance(user, dict):
                user = {
                    "username":     getattr(user, "username", ""),
                    "display_name": getattr(user, "display_name", ""),
                    "bio":          getattr(user, "bio", ""),
                    "posts":        getattr(user, "posts", []),
                }
            bio       = str(user.get("bio", ""))
            posts_raw = user.get("posts", [])[:MAX_POSTS]
            posts      = [str(p["content"]) for p in posts_raw]
            post_dates = [str(p.get("post_date", "")) for p in posts_raw]

            # post embedding
            if len(posts) == 0:
                post_emb = np.zeros((MAX_POSTS, EMBED_DIM), dtype=np.float32)
            else:
                emb = np.array(embed_model.encode(posts), dtype=np.float32)
                if emb.shape[0] < MAX_POSTS:
                    pad  = np.zeros((MAX_POSTS - emb.shape[0], EMBED_DIM), dtype=np.float32)
                    emb  = np.vstack([emb, pad])
                else:
                    emb = emb[:MAX_POSTS]
                post_emb = emb

            bio_emb    = np.array(embed_model.encode(bio), dtype=np.float32)
            num_feat   = compute_user_numeric_features(posts, bio, post_dates)

            self.post_embs.append(post_emb)
            self.bio_embs.append(bio_emb)
            self.numeric_feats.append(num_feat)

        self.post_embs     = torch.tensor(np.array(self.post_embs), dtype=torch.float32)
        self.bio_embs      = torch.tensor(np.array(self.bio_embs),  dtype=torch.float32)
        self.numeric_feats = torch.tensor(
            numerical_scaler.transform(self.numeric_feats), dtype=torch.float32
        )

    def __len__(self):
        return len(self.post_embs)

    def __getitem__(self, idx):
        return {
            "posts": self.post_embs[idx],
            "bio":   self.bio_embs[idx],
            "num":   self.numeric_feats[idx],
        }