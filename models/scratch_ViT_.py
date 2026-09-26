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
    def __init__(self, img_size=224, patch_size=15, in_chans=1, embed_dim=225):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # [B, C, H, W] -> [B, embed_dim, H/P, W/P] -> [B, N, D]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# =============== Efficient Attention ===============
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

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-2)

        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        kv = kv / math.sqrt(self.head_dim)

        out = torch.einsum('bhnd,bhde->bhne', q, kv)

        out = out.permute(0, 2, 1, 3).reshape(B, N, C)

        return self.proj(out)


# =============== Transformer Block ===============
class TransformerBlock(nn.Module):
    def __init__(self, dim=225, num_heads=3, mlp_ratio=512/225):
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


# =============== Binary ViT ===============
class BinaryViT(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=15,
                 in_chans=3,
                 embed_dim=225,
                 depth=3,
                 num_heads=3,
                 mlp_ratio=512/225):

        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # 🔥 Binary head（只改这里）
        self.head = nn.Sequential(
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

            nn.Linear(128, 1)
        )

    def forward(self, x):

        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]

        # Transformer
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # MAX pooling
        x = x.max(dim=1)[0]  # [B, D]

        # Binary output
        logits = self.head(x)  # [B, 1]

        return logits