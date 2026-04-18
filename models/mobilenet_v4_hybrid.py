import torch
import torch.nn as nn

class ConvBN(nn.Sequential):
    def __init__(self, in_c, out_c, k, s=1, g=1):
        padding = (k - 1) // 2
        super(ConvBN, self).__init__(
            nn.Conv2d(in_c, out_c, k, s, padding, groups=g, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

class MobileNetV4HybridLarge(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV4HybridLarge, self).__init__()
        
        # Matches mobilenetv4_hybrid_large structure
        self.stem = ConvBN(3, 32, 3, s=2)
        
        # Large models have many stages. 
        # Note: If your .pt has 'stages', we name this stages.
        self.stages = nn.Sequential(
            ConvBN(32, 48, 3, s=2),
            ConvBN(48, 96, 3, s=2),
            ConvBN(96, 192, 3, s=2),
            ConvBN(192, 384, 3, s=2),
            ConvBN(384, 512, 3, s=1),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # timm Hybrid models often nest the classifier inside a 'head'
        self.head = nn.Sequential(
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x