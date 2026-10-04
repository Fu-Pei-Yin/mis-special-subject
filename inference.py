import torch
import joblib
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils.feature_utils import (
    compute_user_numeric_features,
    extract_logic_features
)
from utils.dataset_utils import UserLogicClassifier

# -----------------------
# Config
# -----------------------
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384
MAX_POSTS = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load artifacts
# -----------------------
num_scaler = joblib.load(f"C:/Users/USER/Desktop/課程/專題/模型訓練/saved_user_model/numerical_scaler.pkl")
logic_scaler = joblib.load(f"C:/Users/USER/Desktop/課程/專題/模型訓練/saved_user_model/logic_scaler.pkl")

with open(f"C:/Users/USER/Desktop/課程/專題/模型訓練/saved_user_model/meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

# 使用「最佳 fold」或指定 fold
MODEL_PATH = f"C:/Users/USER/Desktop/課程/專題/模型訓練/saved_user_model/fold_1_best.pt"

model = UserLogicClassifier(
    EMBED_DIM,
    len(meta["numeric_features"]),
    len(meta["logic_features"])
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -----------------------
# Load embedding model
# -----------------------
st_model = SentenceTransformer(EMBED_MODEL_NAME)

# -----------------------
# Predict function
# -----------------------
def predict_user(posts, bio, post_dates):
    # numeric
    num_feat = compute_user_numeric_features(posts, bio, post_dates)
    num_feat = num_scaler.transform([num_feat])

    # logic
    logic_feat = extract_logic_features(posts, bio)
    logic_feat = logic_scaler.transform([logic_feat])

    # embeddings
    post_texts = posts[:MAX_POSTS]
    post_emb = st_model.encode(post_texts)
    if len(post_emb) < MAX_POSTS:
        pad = np.zeros((MAX_POSTS - len(post_emb), EMBED_DIM))
        post_emb = np.vstack([post_emb, pad])
    post_emb = torch.tensor(post_emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    bio_emb = torch.tensor(
        st_model.encode([bio]), dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        logit = model(
            post_emb,
            bio_emb,
            torch.tensor(num_feat, dtype=torch.float32).to(DEVICE),
            torch.tensor(logic_feat, dtype=torch.float32).to(DEVICE)
        )
        prob = torch.sigmoid(logit).item()

    return prob, int(prob >= 0.5)
