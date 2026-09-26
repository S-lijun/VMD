import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights
'''

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            Swish(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.se(x)         # [B, 1024]
        return x * weights           # [B, 1024]


class PretrainedGoogLeNet(nn.Module):
    def __init__(self):
        super(PretrainedGoogLeNet, self).__init__()

       
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        base = googlenet(weights=weights)

   
        self.feature_extractor = nn.Sequential(
            base.conv1,
            base.maxpool1,
            base.conv2,
            base.conv3,
            base.maxpool2,
            base.inception3a,
            base.inception3b,
            base.maxpool3,
            base.inception4a,
            base.inception4b,
            base.inception4c,
            base.inception4d,
            base.inception4e,
            base.maxpool4,
            base.inception5a,
            base.inception5b,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        
        self.se = SEBlock(1024)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            Swish(),
            nn.Dropout(0.2),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # GoogLeNet 
        x = self.se(x)                 # SE 
        x = self.classifier(x)         # FC 
        return x                       '''


# Original Model

import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights

# Swish 
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Backbone
class PretrainedGoogLeNet(nn.Module):
    def __init__(self):
        super(PretrainedGoogLeNet, self).__init__()

        # Load pretraining GoogLeNet
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        self.base = googlenet(weights=weights)

        # Freeze every GoogLeNet parameters
        for param in self.base.parameters():
            param.requires_grad = False

        # customize 3 extra FC layers
        self.extra_fc = nn.Sequential(
            nn.Linear(1000, 512),
            Swish(), # Swich
            nn.Dropout(0.5), #0.2

            nn.Linear(512, 128),
            Swish(),
            nn.Dropout(0.3),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.base(x)         # GoogLeNet output: [B, 1000]
        x = self.extra_fc(x)     # output: [B, 1] logit
        return x



# This is FT model:

'''
import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights

class PretrainedGoogLeNet(nn.Module):
    def __init__(self, fine_tune=True):
        super(PretrainedGoogLeNet, self).__init__()
        
        # Load pretrained GoogLeNet
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        self.base = googlenet(weights=weights)
        
        # fine-tune backbone
        if not fine_tune:
            for param in self.base.parameters():
                param.requires_grad = False
        else:
            # unfreeze
            for name, param in self.base.named_parameters():
                if "inception4" in name or "inception5" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        # head
        self.head = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1)  # binary output
        )
    
    def forward(self, x):
        x = self.base(x)  # [B, 1000]
        x = self.head(x)
        return x
'''
# Weak Learner
'''
import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights

# Swish 
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Backbone
class PretrainedGoogLeNet(nn.Module):
    def __init__(self):
        super(PretrainedGoogLeNet, self).__init__()

        # Load pretraining GoogLeNet
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        self.base = googlenet(weights=weights)

        # Freeze every GoogLeNet parameters
        for param in self.base.parameters():
            param.requires_grad = False

        # customize 3 extra FC layers
        self.extra_fc = nn.Sequential(
            nn.Linear(1000, 1)
        )

    def forward(self, x):
        x = self.base(x)         # GoogLeNet output: [B, 1000]
        x = self.extra_fc(x)     # output: [B, 1] logit
        return x

'''
'''
import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights

# Swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Backbone
class PretrainedGoogLeNet(nn.Module):
    def __init__(self):
        super(PretrainedGoogLeNet, self).__init__()

        # Load pretrained GoogLeNet
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        self.base = googlenet(weights=weights)

        # 去掉原始的 ImageNet 分类层 (fc: 1024→1000)，直接输出 1024 维特征
        self.base.fc = nn.Identity()

        # Freeze every GoogLeNet parameter
        for param in self.base.parameters():
            param.requires_grad = False

        # customize 3 extra FC layers (输入从 1000 改成 1024)
        self.extra_fc = nn.Sequential(
            nn.Linear(1024, 512),
            Swish(),
            nn.Dropout(0.5),

            nn.Linear(512, 64),
            Swish(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.base(x)       # GoogLeNet backbone output: [B, 1024]
        x = self.extra_fc(x)   # Custom head: [B, 1] logit
        return x
'''