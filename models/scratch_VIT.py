import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Linear Patch Embedding
# ============================================================

class PatchEmbed(nn.Module):

    def __init__(self, img_size=300, patch_size=15, in_chans=1, embed_dim=225):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        patch_dim = patch_size * patch_size * in_chans

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, x):

        # x: [B,C,H,W]

        patches = self.unfold(x)
        patches = patches.transpose(1,2)

        embeddings = self.proj(patches)

        return embeddings



# ============================================================
# Efficient Attention 
# ============================================================

class EfficientAttention(nn.Module):

    def __init__(self, dim, num_heads=3, qkv_bias = True):

        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias = qkv_bias)
        self.k = nn.Linear(dim, dim, bias = qkv_bias)
        self.v = nn.Linear(dim, dim, bias = qkv_bias)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):

        B,N,C = x.shape

        q = self.q(x).reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        k = self.k(x).reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        v = self.v(x).reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        # -----------------------------
        # Efficient Attention (paper)
        # -----------------------------
        q = F.softmax(q, dim=-1)   # over feature
        k = F.softmax(k, dim=-2)   # over tokens

        # Explicit K^T V and Q(K^T V) with matmul.
        # k: [B, h, N, d], v: [B, h, N, d]
        k_t = k.transpose(-2, -1)                    # [B, h, d, N]
        kv = torch.matmul(k_t, v)                    # [B, h, d, d]
        kv = kv / math.sqrt(self.head_dim)           # scale before multiplying Q
        out = torch.matmul(q, kv)                    # [B, h, N, d]

        out = out.permute(0,2,1,3).reshape(B, N, C)

        return self.proj(out)

'''
class EfficientAttention(nn.Module):

    def __init__(self, dim, num_heads=3, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Split into heads first, then apply per-head Q/K/V linear layers.
        self.q_heads = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim, bias=qkv_bias)
            for _ in range(num_heads)
        ])
        self.k_heads = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim, bias=qkv_bias)
            for _ in range(num_heads)
        ])
        self.v_heads = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim, bias=qkv_bias)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        # [B, N, C] -> [B, h, N, d]
        x_heads = x.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q_list = []
        k_list = []
        v_list = []
        for i in range(self.num_heads):
            head_x = x_heads[:, i, :, :]  # [B, N, d]
            q_list.append(self.q_heads[i](head_x))
            k_list.append(self.k_heads[i](head_x))
            v_list.append(self.v_heads[i](head_x))

        q = torch.stack(q_list, dim=1)  # [B, h, N, d]
        k = torch.stack(k_list, dim=1)  # [B, h, N, d]
        v = torch.stack(v_list, dim=1)  # [B, h, N, d]

        # Efficient Attention (paper)
        q = F.softmax(q, dim=-1)  # over feature
        k = F.softmax(k, dim=-2)  # over tokens

        v = v / math.sqrt(self.head_dim)
        kv = torch.einsum("bhnd,bhne->bhde", k, v)   # [B, h, d, d]
        out = torch.einsum("bhnd,bhde->bhne", q, kv)  # [B, h, N, d]

        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)
'''
# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self, dim=225, num_heads=3, mlp_ratio=512/225, dropout=0.0):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim,num_heads = num_heads)
        self.drop_attn = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.drop_mlp = nn.Dropout(dropout)

    def forward(self, x):

        x = x + self.drop_attn(self.attn(self.norm1(x)))
        x = x + self.drop_mlp(self.mlp(self.norm2(x)))

        return x



# ============================================================
# Binary ViT
# ============================================================

class BinaryViT(nn.Module):

    def __init__(self,
                 img_size=300,
                 patch_size=15,
                 in_chans=1,
                 embed_dim=225,
                 depth=3,
                 num_heads=3,
                 mlp_ratio=512/225,
                 dropout=0.0):
        """
        dropout: applied on attention output and inside MLP (after GELU). Use e.g. 0.1 to reduce overfitting.
        Default 0.0 matches a no-dropout setup (paper does not state dropout).
        """

        super().__init__()

        self.patch_embed = PatchEmbed(img_size,patch_size,in_chans,embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Decision Layer
        self.head = nn.Linear(embed_dim,1)


    def forward(self,x):

        x = self.patch_embed(x)

        for blk in self.blocks:
            x = blk(x)

        #x = self.norm(x)

        # MAX-AGGR-2
        x = x.max(dim=1)[0]
    

        logits = self.head(x)

        return logits