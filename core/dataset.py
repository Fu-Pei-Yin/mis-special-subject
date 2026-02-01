import torch
import numpy as np
import joblib
from torch.utils.data import Dataset

EMBED_DIM = 384
MAX_POSTS = 32

NUM_SCALER_PATH = "C:/Users/USER/Desktop/課程/專題/threads_depression_web/model/numerical_scaler.pkl"
LOGIC_SCALER_PATH = "C:/Users/USER/Desktop/課程/專題/threads_depression_web/model/logic_scaler.pkl"

numerical_scaler = joblib.load(NUM_SCALER_PATH)
logic_scaler = joblib.load(LOGIC_SCALER_PATH)

def compute_user_numeric_features(post_texts, bio, post_dates):
    post_count = len(post_texts)
    avg_post_len = np.mean([len(str(t)) for t in post_texts]) if post_count > 0 else 0.0
    bio_len = len(str(bio))
    night_count = sum(1 for d in post_dates if 0 <= int(d.split(" ")[1].split(":")[0]) <= 5)
    night_ratio = night_count / post_count if post_count > 0 else 0.0
    max_post_len = max([len(str(t)) for t in post_texts]) if post_count > 0 else 0.0
    min_post_len = min([len(str(t)) for t in post_texts]) if post_count > 0 else 0.0
    return [post_count, avg_post_len, bio_len, night_ratio, max_post_len, min_post_len]

def extract_logic_features(post_list, bio):
    neg_words = ["痛苦","沒意義","撐不住","累","空虛","需要陪伴","消失","不想活","孤獨","絕望","負面"]
    pos_words = ["開心","期待","感謝","喜歡","平靜","快樂","滿足"]
    neg_count = sum(sum(w in p for w in neg_words) for p in post_list)
    pos_count = sum(sum(w in p for w in pos_words) for p in post_list)
    logic_score = neg_count - pos_count
    bio_flag = 1 if any(w in bio for w in ["抑鬱","憂鬱","低潮","焦慮"]) else 0
    total = neg_count + pos_count
    neg_ratio = neg_count / total if total > 0 else 0.0
    return [neg_count, pos_count, logic_score, bio_flag, neg_ratio]

class ThreadsInferenceDataset(Dataset):
    def __init__(self, users_data, embed_model):
        self.post_embs, self.bio_embs, self.numeric_feats, self.logic_feats = [], [], [], []

        for user in users_data:
            if not isinstance(user, dict):
                user = {
                    "username": getattr(user, "username", ""),
                    "display_name": getattr(user, "display_name", ""),
                    "bio": getattr(user, "bio", ""),
                    "posts": getattr(user, "posts", [])
                }
            bio = user.get("bio", "")
            posts_raw = user.get("posts", [])[:MAX_POSTS]
            posts = [p["content"] for p in posts_raw]
            post_dates = [p["post_date"] for p in posts_raw]

            # embedding
            if len(posts) == 0:
                post_emb = np.zeros((MAX_POSTS, EMBED_DIM), dtype=np.float32)
            else:
                emb = np.array(embed_model.encode(posts), dtype=np.float32)
                if emb.shape[0] < MAX_POSTS:
                    pad = np.zeros((MAX_POSTS - emb.shape[0], EMBED_DIM), dtype=np.float32)
                    emb = np.vstack([emb, pad])
                else:
                    emb = emb[:MAX_POSTS]
                post_emb = emb

            bio_emb = np.array(embed_model.encode(bio), dtype=np.float32)
            num_feat = compute_user_numeric_features(posts, bio, post_dates)
            logic_feat = extract_logic_features(posts, bio)

            self.post_embs.append(post_emb)
            self.bio_embs.append(bio_emb)
            self.numeric_feats.append(num_feat)
            self.logic_feats.append(logic_feat)

        self.post_embs = torch.tensor(np.array(self.post_embs), dtype=torch.float32)
        self.bio_embs = torch.tensor(np.array(self.bio_embs), dtype=torch.float32)
        self.numeric_feats = torch.tensor(numerical_scaler.transform(self.numeric_feats), dtype=torch.float32)
        self.logic_feats = torch.tensor(logic_scaler.transform(self.logic_feats), dtype=torch.float32)

    def __len__(self):
        return len(self.post_embs)

    def __getitem__(self, idx):
        return {
            "posts": self.post_embs[idx],
            "bio": self.bio_embs[idx],
            "num": self.numeric_feats[idx],
            "logic": self.logic_feats[idx]
        }
