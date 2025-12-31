import torch
from sentence_transformers import SentenceTransformer
from inference.dataset import ThreadsInferenceDataset
from model.model import UserLogicClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_username(user_data):
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    dataset = ThreadsInferenceDataset(users_data=[user_data], embed_model=embed_model)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model = UserLogicClassifier(
        embed_dim=384,          # 訓練時的 embedding_dim
        num_num_feats=6,
        num_logic_feats=5
    ).to(DEVICE)

    model.load_state_dict(torch.load("C:/Users/USER/Desktop/課程/專題/threads_depression_detector/model/fold_1_best.pt", map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        for batch in loader:
            posts = batch["posts"].to(DEVICE)
            bio = batch["bio"].to(DEVICE)
            num = batch["num"].to(DEVICE)
            logic = batch["logic"].to(DEVICE)
            logits = model(posts, bio, num, logic)
            prob = torch.sigmoid(logits).item()
            return prob
