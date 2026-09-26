import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights
# Model: Every branches have independnet FC layers

import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights

'''
class PretrainedGoogLeNet_Multilabel(nn.Module):
    def __init__(self, num_users):
        super().__init__()

        # Load pretrained GoogLeNet backbone
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        self.base = googlenet(weights=weights)
        self.base.aux1 = None
        self.base.aux2 = None
        self.base.fc = nn.Identity()

        # Unfreeze backbone
        for param in self.base.parameters():
            param.requires_grad = True

        self.num_users = num_users

        # Independent FC head
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
        # Backbone feature extraction
        x = self.base.conv1(x)
        x = self.base.maxpool1(x)
        x = self.base.conv2(x)
        x = self.base.conv3(x)
        x = self.base.maxpool2(x)
        x = self.base.inception3a(x)
        x = self.base.inception3b(x)
        x = self.base.maxpool3(x)
        x = self.base.inception4a(x)
        x = self.base.inception4b(x)
        x = self.base.inception4c(x)
        x = self.base.inception4d(x)
        x = self.base.inception4e(x)
        x = self.base.maxpool4(x)
        x = self.base.inception5a(x)
        x = self.base.inception5b(x)
        x = self.base.avgpool(x)     # [B, 1024, 1, 1]
        x = torch.flatten(x, 1)      # [B, 1024]

        # 每个用户独立head计算logit
        logits = [head(x) for head in self.user_heads]   # list of [B,1]
        logits = torch.cat(logits, dim=1)                # [B, num_users]

        return logits
'''


class PretrainedGoogLeNet_Multilabel(nn.Module):
    def __init__(self, num_users):
        super().__init__()

        # Load pretrained GoogLeNet backbone
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        self.base = googlenet(weights=weights)
        self.base.aux1 = None
        self.base.aux2 = None
        self.base.fc = nn.Identity()

        # Unfreeze backbone
        for param in self.base.parameters():
            param.requires_grad = True

        self.num_users = num_users

        # Independent FC head
        self.user_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 1)
            )
            for _ in range(num_users)
        ])

    def forward(self, x):
        # Backbone feature extraction
        x = self.base.conv1(x)
        x = self.base.maxpool1(x)
        x = self.base.conv2(x)
        x = self.base.conv3(x)
        x = self.base.maxpool2(x)
        x = self.base.inception3a(x)
        x = self.base.inception3b(x)
        x = self.base.maxpool3(x)
        x = self.base.inception4a(x)
        x = self.base.inception4b(x)
        x = self.base.inception4c(x)
        x = self.base.inception4d(x)
        x = self.base.inception4e(x)
        x = self.base.maxpool4(x)
        x = self.base.inception5a(x)
        x = self.base.inception5b(x)
        x = self.base.avgpool(x)     # [B, 1024, 1, 1]
        x = torch.flatten(x, 1)      # [B, 1024]
        x = self.base.dropout(x)

        # 每个用户独立head计算logit
        logits = [head(x) for head in self.user_heads]   # list of [B,1]
        logits = torch.cat(logits, dim=1)                # [B, num_users]

        return logits


