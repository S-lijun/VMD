import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

'''
# Swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PretrainedViT_B16_Multilabel_NoCLS_NoPos(nn.Module):
    """
    - 去掉 CLS token
    - 去掉 positional embeddings
    - 使用所有 patch features 做 max aggregation (论文风格)
    - 不修改你的 multi-label user heads
    """
    def __init__(self, num_users):
        super().__init__()

        # Load pretrained ViT-B/16 backbone
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.base = vit_b_16(weights=weights)

        # unfreeze backbone
        for param in self.base.parameters():
            param.requires_grad = True

        # Remove classification head
        in_features = self.base.heads.head.in_features
        self.base.heads = nn.Identity()

        self.num_users = num_users

        # === Multi-user independent heads (unchanged) ===
        self.user_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 512),
                Swish(),
                nn.Dropout(0.5),

                nn.Linear(512, 128),
                Swish(),
                nn.Dropout(0.3),

                nn.Linear(128, 1)  # binary output per user
            )
            for _ in range(num_users)
        ])

    # -------------------------------------------------------------------------
    #  🔥 前向传播 = 论文风格：
    #    - 无 CLS token
    #    - 无 positional embedding
    #    - transformer patches → max pooling → heads
    # -------------------------------------------------------------------------
    def forward(self, x):

        # x: [B, C, H, W]  (your images)
        # Patch embedding, but WITHOUT adding CLS token
        x = self.base._process_input(x)  # [B, num_patches, hidden_dim]

        B, N, D = x.shape

        # ====== 🔥 不加 class token ======
        # ====== 🔥 不加 positional embedding ======
        # 所以直接送进 encoder

        # the ViT encoder expects shape [B, N, D]
        # but torchvision uses: encoder(x) where x is [B, N, D]

        # Go through transformer layers without CLS token
        x = self.base.encoder.dropout(x)  # keep consistency
        x = self.base.encoder.layers(x)   # transformer blocks
        x = self.base.encoder.ln(x)

        # ===== 🔥 MAX-AGGR-2 (论文推荐) =====
        # 取每个 feature 维度在所有 patches 的最大值
        # x: [B, N, D] → pooled: [B, D]
        pooled = torch.max(x, dim=1)[0]

        # ===== Multi-head classifier (your unchanged part) =====
        logits = [head(pooled) for head in self.user_heads]  # list of [B,1]
        logits = torch.cat(logits, dim=1)                    # [B, num_users]

        return logits
'''

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PretrainedViT_B16_Multilabel_NoCLS_NoPos(nn.Module):
    def __init__(self, num_users, image_size=224):
        super().__init__()

        # 1. 创建 448 尺寸的模型骨架 (不带 weights 避免报错)
        self.base = vit_b_16(weights=None, image_size=image_size)

        # 2. 获取官方 224 尺寸的预训练权重
        print(f"[INFO] Loading ImageNet-1K pretrained weights...")
        pretrained_state_dict = ViT_B_16_Weights.IMAGENET1K_V1.get_state_dict(check_hash=True)
        
        # 3. 过滤并加载权重
        model_dict = self.base.state_dict()
        # 只要名字一样且形状一样 (比如 Transformer 层)，就加载进去
        matched_dict = {
            k: v for k, v in pretrained_state_dict.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
        
        missing_keys, unexpected_keys = self.base.load_state_dict(matched_dict, strict=False)
        print(f"[INFO] Loaded {len(matched_dict)} weight tensors.")
        print(f"[INFO] Missing keys (usually position/cls/heads): {len(missing_keys)}")

        # 4. 冻结/解冻设置
        for param in self.base.parameters():
            param.requires_grad = True

        # 移除原有的分类头
        in_features = self.base.heads.head.in_features
        self.base.heads = nn.Identity()

        self.num_users = num_users
        self.user_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 512),
                Swish(),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                Swish(),
                nn.Dropout(0.3),
                nn.Linear(128, 1)
            ) for _ in range(num_users)
        ])

    def forward(self, x):
        # 此时 self.base.image_size 是 448，校验通过
        x = self.base._process_input(x)  # [B, num_patches, hidden_dim]
        
        # 按照你的论文风格：跳过 CLS 和 Position Embedding
        x = self.base.encoder.dropout(x)
        x = self.base.encoder.layers(x)
        x = self.base.encoder.ln(x)

        # Max Pooling 聚合
        pooled = torch.max(x, dim=1)[0]

        logits = [head(pooled) for head in self.user_heads]
        return torch.cat(logits, dim=1)
