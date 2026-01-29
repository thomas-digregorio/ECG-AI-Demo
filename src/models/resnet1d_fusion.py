import torch
import torch.nn as nn
from src.models.resnet1d import ResNet1d, ResNetBottleneck

class ContextNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, embedding_dim=16):
        super(ContextNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            # No activation at the end, this is an embedding feature
        )
        
    def forward(self, x):
        return self.net(x)

class ResNet1dFusion(nn.Module):
    def __init__(self, num_classes=5, input_channels=12, context_input_dim=11):
        super(ResNet1dFusion, self).__init__()
        
        # Branch A: The Eye (ECG)
        # We reuse the ResNet backbone logic but chop off the classification head
        self.backbone = ResNet1d(ResNetBottleneck, [3, 4, 6, 3], num_classes=num_classes, input_channels=input_channels)
        
        # Replace the final FC of backbone to output a feature vector (e.g. 128)
        # The ResNet1d output before FC is (512 * expansion) = 2048
        self.backbone_dim = 512 * 4
        self.ecg_feature_dim = 128
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone_dim, self.ecg_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Branch B: The Chart Reader (Metadata)
        self.context_embedding_dim = 16
        self.context_net = ContextNet(input_dim=context_input_dim, embedding_dim=self.context_embedding_dim)
        
        # Fusion Layer
        total_dim = self.ecg_feature_dim + self.context_embedding_dim
        self.classifier = nn.Linear(total_dim, num_classes)
        
    def forward(self, x_img, x_ctx):
        # 1. Process ECG
        img_feats = self.backbone(x_img) # (Batch, 128)
        
        # 2. Process Metadata
        ctx_feats = self.context_net(x_ctx) # (Batch, 16)
        
        # 3. Fuse
        combined = torch.cat((img_feats, ctx_feats), dim=1) # (Batch, 144)
        
        # 4. Classify
        logits = self.classifier(combined)
        return logits

def resnet1d_fusion(num_classes=5):
    return ResNet1dFusion(num_classes=num_classes)
