import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# Basic Conv Block
# =====================================================
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# =====================================================
# Scratch CNN Backbone (Modified for flexible image size)
# =====================================================
class ScratchCNNBackbone(nn.Module):
    """
    Input : [B, 3, H, W]  (H, W can be any size, e.g., 448)
    Output: [B, 1024]
    """
    def __init__(self, image_size=224):
        super().__init__()
        self.image_size = image_size

        # Stage 1-4 
        self.stage1 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2)
        )

        self.stage2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2)
        )

        self.stage3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=2)
        )

        self.stage4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(kernel_size=2)
        )

        # Stage 5 
        self.stage5 = nn.Sequential(
            ConvBlock(512, 1024),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        
        # if x.shape[-1] != self.image_size:
        #     x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear')

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = torch.flatten(x, 1)  # [B, 1024]
        return x


# =====================================================
# Scratch Multi-Label CNN (Main Model)
# =====================================================
class ScratchMultiCNN(nn.Module):
    """
    - Pure scratch CNN
    - Multi-label (one logit per user)
    - Output shape: [B, num_users]
    """
    def __init__(self, num_users, image_size=224):
        super().__init__()

        self.num_users = num_users
        #  backbone
        self.backbone = ScratchCNNBackbone(image_size=image_size)

        # Independent head per user
        self.user_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.4),

                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(128, 1)
            )
            for _ in range(num_users)
        ])

    def forward(self, x):
        """
        x: [B, 3, H, W]
        return: logits [B, num_users]
        """
        features = self.backbone(x)   # [B, 1024]

        logits = [head(features) for head in self.user_heads]  # list of [B,1]
        logits = torch.cat(logits, dim=1)                      # [B, num_users]

        return logits