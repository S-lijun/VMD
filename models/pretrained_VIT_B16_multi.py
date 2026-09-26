import timm
import torch
import torch.nn as nn

'''
# --- ViT-B/32 (timm), kept for quick rollback ---
class PretrainedViT_B32_Multilabel(nn.Module):

    def __init__(self, num_users, img_size=448):

        super().__init__()

        self.backbone = timm.create_model(
            "vit_base_patch32_224",
            pretrained=True,
            img_size=img_size,
            num_classes=0
        )

        feature_dim = self.backbone.num_features

        self.user_heads = nn.ModuleList([
            nn.Linear(feature_dim, 1)
            for _ in range(num_users)
        ])

    def forward(self, x):

        features = self.backbone(x)

        logits = [
            head(features)
            for head in self.user_heads
        ]

        return torch.cat(logits, dim=1)
'''


class PretrainedViT_B16_Multilabel(nn.Module):

    def __init__(self, num_users, img_size=448):

        super().__init__()

        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            img_size=img_size,
            num_classes=0
        )

        feature_dim = self.backbone.num_features

        self.user_heads = nn.ModuleList([
            nn.Linear(feature_dim, 1)
            for _ in range(num_users)
        ])

    def forward(self, x):

        features = self.backbone(x)

        logits = [
            head(features)
            for head in self.user_heads
        ]

        return torch.cat(logits, dim=1)
