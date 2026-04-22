"""
predict.py（更新版）
整合說明性模組，predict_username 現在同時回傳機率與 ExplainResult。
"""
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from core.dataset import ThreadsInferenceDataset
from model.model import UserLogicClassifier
from core.explain import DepressionModelExplainer, ExplainResult

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH   = BASE_DIR / "../model/fold_1_best.pt"


def _load_model_and_embed():
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    model = UserLogicClassifier(
        embed_dim=384,
        num_num_feats=6,
        num_logic_feats=5,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, embed_model


def predict_username(user_data: dict) -> float:
    """
    原始介面（向下相容）：只回傳風險機率（float）。
    """
    model, embed_model = _load_model_and_embed()
    dataset = ThreadsInferenceDataset(users_data=[user_data], embed_model=embed_model)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["posts"].to(DEVICE),
                batch["bio"].to(DEVICE),
                batch["num"].to(DEVICE),
                batch["logic"].to(DEVICE),
            )
            return torch.sigmoid(logits).item()


def predict_with_explanation(
    user_data: dict,
    verbose: bool = True,
    plot: bool = False,
) -> ExplainResult:
    """
    新版介面：回傳完整 ExplainResult，包含：
      - result.risk_prob     : 風險機率 (float)
      - result.risk_level    : "高風險" / "中等風險" / "低風險"
      - result.summary       : 自然語言說明摘要
      - result.top_posts     : 各貼文注意力與關鍵詞分析
      - result.feature_exp   : 數值與邏輯特徵（含 IG 歸因）
      - result.attention_weights : 所有貼文的注意力分布

    Args:
        user_data : 使用者資料（與原本格式相同）
        verbose   : 是否印出摘要
        plot      : 是否顯示視覺化圖表（需要 matplotlib）

    範例::

        result = predict_with_explanation(user_data, plot=True)
        print(f"風險機率：{result.risk_prob:.1%}")
    """
    model, embed_model = _load_model_and_embed()

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

    if plot:
        explainer.plot_attention(result)
        explainer.plot_features(result)
        explainer.plot_logic_radar(result)

    return result