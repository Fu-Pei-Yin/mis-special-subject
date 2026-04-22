# model/model.py

import torch
import torch.nn as nn


# ============================================================
# Posts Encoder (BiLSTM)
# ============================================================

class PostsLSTMEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.output_dim = hidden_dim * 2  # BiLSTM

    def forward(self, x):
        """
        x: (B, MAX_POSTS, EMBED_DIM)
        """
        _, (hidden, _) = self.lstm(x)

        # 取最後一層 forward + backward
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)

        return out


# ============================================================
# Main Classifier (General Model - No Logic Features)
# ============================================================

class GeneralBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_num_feats: int,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        hidden: int = 128,
        dropout: float = 0.4
    ):
        super().__init__()

        # ----- Posts encoder -----
        self.posts_encoder = PostsLSTMEncoder(
            embed_dim=embed_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout
        )

        # ----- Bio projection -----
        self.bio_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ----- Feature dimension -----
        total_dim = self.posts_encoder.output_dim + embed_dim + num_num_feats

        # ----- MLP classifier -----
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden // 2, 1)
        )

    def forward(self, posts, bio, num_feats):
        """
        posts: (B, MAX_POSTS, EMBED_DIM)
        bio:   (B, EMBED_DIM)
        num_feats: (B, num_num_feats)
        """

        posts_vec = self.posts_encoder(posts)
        bio_vec = self.bio_proj(bio)

        x = torch.cat(
            [posts_vec, bio_vec, num_feats],
            dim=1
        )

        logits = self.mlp(x).squeeze(-1)

        return logits