import torch.nn as nn
from torch.utils.data import Dataset
import torch
import random
AUGMENT_PROB = 0.3
NOISE_STD = 0.05
class UserPostsDataset(Dataset):
    def __init__(self, post_embs, bio_embs, numeric_feats, logic_feats, labels, augment=False):
        self.post_embs = torch.tensor(post_embs)
        self.bio_embs = torch.tensor(bio_embs)
        self.num_feats = torch.tensor(numeric_feats, dtype=torch.float32)
        self.logic_feats = torch.tensor(logic_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.augment = augment
    
    def __len__(self):
        return self.post_embs.shape[0]
    
    def __getitem__(self, idx):
        posts = self.post_embs[idx]
        bio = self.bio_embs[idx]
        
        # Data augmentation: add Gaussian noise
        if self.augment and random.random() < AUGMENT_PROB:
            posts = posts + torch.randn_like(posts) * NOISE_STD
            bio = bio + torch.randn_like(bio) * NOISE_STD
        
        return {
            "posts": posts,
            "bio": bio,
            "num": self.num_feats[idx],
            "logic": self.logic_feats[idx],
            "label": self.labels[idx]
        }

# ============================================================
# 6. Enhanced Model
# ============================================================
class PostsEncoder(nn.Module):
    def __init__(self, embed_dim, nhead=4, nhid=256, nlayers=2, dropout=0.3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        out = self.transformer(x)
        out = self.layer_norm(out)
        pooled = out.mean(dim=1)
        return pooled

class UserLogicClassifier(nn.Module):
    def __init__(self, embed_dim, num_num_feats, num_logic_feats, hidden=128, dropout=0.4):
        super().__init__()
        self.posts_encoder = PostsEncoder(embed_dim, dropout=dropout)
        self.bio_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        total_dim = embed_dim * 2 + num_num_feats + num_logic_feats
        
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
        posts_vec = self.posts_encoder(posts)
        bio_vec = self.bio_proj(bio)
        x = torch.cat([posts_vec, bio_vec, num_feats, logic_feats], dim=1)
        logits = self.mlp(x).squeeze(-1)
        return logits