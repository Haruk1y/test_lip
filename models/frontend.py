import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        
        # 3D Convolution front-end
        self.front_end = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 128, kernel_size=(5, 5, 5), stride=(1, 1, 1), padding=(2, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # チャンネル数を3に変換する層を追加
        self.channel_conv = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )
        
        # ResNet34 backend
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backend = nn.Sequential(*list(resnet.children())[:-2])
        
        # Projection layer
        self.proj = nn.Sequential(
            nn.Conv2d(512, output_dim, kernel_size=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        # x shape: (batch, channel=1, time, height, width)
        b, c, t, h, w = x.size()
        
        # Front-end 3D CNN
        x = self.front_end(x)  # (batch, 256, time, h/8, w/8)
        
        # Reshape for 2D processing
        x = x.transpose(1, 2).contiguous()  # (batch, time, 256, h/8, w/8)
        x = x.view(-1, 256, x.size(3), x.size(4))  # (batch*time, 256, h/8, w/8)
        
        # Convert to 3 channels for ResNet
        x = self.channel_conv(x)  # (batch*time, 3, h/8, w/8)
        
        # Backend ResNet
        x = self.backend(x)  # (batch*time, 512, h/32, w/32)
        
        # Project to desired dimension
        x = self.proj(x)  # (batch*time, output_dim, h/32, w/32)
        
        # Global average pooling
        x = x.mean([-2, -1])  # (batch*time, output_dim)
        
        # Reshape back to sequence
        x = x.view(b, t, -1)  # (batch, time, output_dim)
        
        return x