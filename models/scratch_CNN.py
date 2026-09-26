import torch
import torch.nn as nn
import torch.nn.functional as F

'''Bad Performance'''

class insiderThreatCNN(nn.Module):
    def __init__(self, input_size=224):
        super(insiderThreatCNN, self).__init__()
        self.input_size = input_size

        # ----- Convolutional layers with BatchNorm -----
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(128) #-> [128, 28, 28]

        self.pool = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # -> [128,]

        # ----- Fixed flattened size after global_avg_pool -----
        self.flattened_size = 128
############################################################
        # ----- Fully Connected layers -----
        self.fc5 = nn.Linear(self.flattened_size, 1024)
        self.dropout1 = nn.Dropout(0.3)

        self.fc6 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)

        self.fc7 = nn.Linear(512, 1)

        # ----- Initialize Weights -----
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01))
        #print("[FORWARD] after conv1:", x.mean().item(), x.std().item())

        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01))
        #print("[FORWARD] after conv2:", x.mean().item(), x.std().item())

        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        #print("[FORWARD] after conv3:", x.mean().item(), x.std().item())

        x = self.global_avg_pool(x)  # (B, 128, 1, 1)
        x = torch.flatten(x, 1)      # (B, 128)

        x = F.leaky_relu(self.fc5(x), negative_slope=0.01)
        #print("[FORWARD] after fc5:", x.mean().item(), x.std().item())
        x = self.dropout1(x)

        x = F.leaky_relu(self.fc6(x), negative_slope=0.01)
        #print("[FORWARD] after fc6:", x.mean().item(), x.std().item())
        x = self.dropout2(x)

        x = self.fc7(x)
        #print("[FORWARD] final output:", x.mean().item(), x.std().item())

        return x  # raw logits for BCEWithLogitsLoss
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class insiderThreatCNN(nn.Module):
    def __init__(self):
        super(insiderThreatCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
    
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
    
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
    
        return x'''

