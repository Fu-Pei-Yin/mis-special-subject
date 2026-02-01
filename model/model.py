# model/model.py

import torch
import torch.nn as nn


# ============================================================
# Posts Encoder (Transformer)
# ============================================================

class PostsEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        nhead: int = 4,
        nhid: int = 256,
        nlayers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=nlayers
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, MAX_POSTS, EMBED_DIM)
        """
        out = self.transformer(x)
        out = self.layer_norm(out)
        pooled = out.mean(dim=1)
        return pooled


# ============================================================
# Main Classifier
# ============================================================

class UserLogicClassifier(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_num_feats: int,
        num_logic_feats: int,
        hidden: int = 128,
        dropout: float = 0.4
    ):
        super().__init__()

        # ----- Text posts encoder -----
        self.posts_encoder = PostsEncoder(
            embed_dim=embed_dim,
            dropout=dropout
        )

        # ----- Bio projection -----
        self.bio_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        total_dim = embed_dim * 2 + num_num_feats + num_logic_feats

        # ----- Classifier MLP -----
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden // 2, 1)
        )

    def forward(self, posts, bio, num_feats, logic_feats):
        """
        posts: (B, MAX_POSTS, EMBED_DIM)
        bio:   (B, EMBED_DIM)
        num_feats:   (B, num_num_feats)
        logic_feats: (B, num_logic_feats)
        """
        posts_vec = self.posts_encoder(posts)
        bio_vec = self.bio_proj(bio)

        x = torch.cat(
            [posts_vec, bio_vec, num_feats, logic_feats],
            dim=1
        )

        logits = self.mlp(x).squeeze(-1)
        return logits
