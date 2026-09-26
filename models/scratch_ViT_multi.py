import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============== Swish 激活 ===============
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# =============== Patch Embedding ===============
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=15, in_chans=3, embed_dim=225):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # [B, 3, 224, 224] -> [B, embed_dim, H/P, W/P] -> [B, N, D]
        x = self.proj(x)  # [B, 225, 14, 14]  (if 224/15=14)
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


# =============== Efficient Attention (linearized) ===============
class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)  # [B,N,H,D]
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)  # [B,H,N,D]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Linearized attention: softmax(Q) * (softmax(K)^T * V)
        q = F.softmax(q, dim=-1)  # [B,H,N,D]
        k = F.softmax(k, dim=-2)  # [B,H,N,D]

        kv = torch.einsum('bhnd,bhne->bhde', k, v)  # [B,H,D,D]
        kv = kv / math.sqrt(self.head_dim)
        out = torch.einsum('bhnd,bhde->bhne', q, kv)  # [B,H,N,D]

        out = out.permute(0, 2, 1, 3).reshape(B, N, C)  # [B,N,C]
        return self.proj(out)


# =============== Transformer Encoder Block ===============
class TransformerBlock(nn.Module):
    def __init__(self, dim=225, num_heads=3, mlp_ratio=512/225, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# =============== Scratch MiniViT (Multi-User) ===============
class ScratchMiniViT_MultiLabel(nn.Module):
    def __init__(self, num_users, img_size=224, patch_size=5,
                 in_chans=3, embed_dim=225, depth=3, num_heads=3, mlp_ratio=512/225):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.num_users = num_users

        # multi head
        self.user_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 1024),
                Swish(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                Swish(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                Swish(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                Swish(),
                nn.Dropout(0.05),
                nn.Linear(128 , 1)
            )
            for _ in range(num_users)
        ])

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # MAX-AGGR-2 pooling: 取每个 feature 维度的最大值
        x = x.max(dim=1)[0]  # [B, D]
        # Average pooling
        #x = x.mean(dim=1)  # [B, D]

        # Multi-head outputs
        logits = [head(x) for head in self.user_heads]
        logits = torch.cat(logits, dim=1)  # [B, num_users]

        return logits

