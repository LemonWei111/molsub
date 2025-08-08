__author__ = "Lemon Wei"
__email__ = "Lemon2922436985@gmail.com"
__version__ = "1.1.0"

import torch.nn as nn
import torch.nn.functional as F

from timm.models.swin_transformer import swin_base_patch4_window7_224

class SpatialInteractionBlock(nn.Module):
    def __init__(self, dim):
        super(SpatialInteractionBlock, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        return x
    
class FeatureCompressionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureCompressionBlock, self).__init__()
        self.compress = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.compress(x)
    
class RelationalAggregationBlock(nn.Module):
    def __init__(self, channels):
        super(RelationalAggregationBlock, self).__init__()
        self.deform_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        x = self.deform_conv(x)
        x = self.ca(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x
    
class BTS_ST(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(BTS_ST, self).__init__()
        self.swin = swin_base_patch4_window7_224(pretrained=pretrained)
        self.sib = SpatialInteractionBlock(dim=1024)
        self.fcb = FeatureCompressionBlock(in_channels=1024, out_channels=64)
        self.rab = RelationalAggregationBlock(channels=1024)

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(32, num_classes, kernel_size=1)
        # )

        # 添加全局平均池化层和全连接层进行分类
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        features = self.swin.forward_features(x)  # 获取 Swin Transformer 特征
        features = features.permute(0, 3, 1, 2)
        # print(features.shape)
        features = self.sib(features)  # 应用 SIB
        # print(features.shape)
        features = self.rab(features)  # 应用 RAB
        # print(features.shape)
        features = self.fcb(features)  # 应用 FCB 压缩特征
        # print(features.shape)

        # output = self.decoder(features)  # 解码得到最终输出
        # print(features.shape)

        # 全局平均池化
        pooled_features = self.global_pool(features).flatten(1)  # [batch_size, 512]
        # print(pooled_features.shape)
        # 分类器
        output = self.classifier(pooled_features)  # [batch_size, num_classes]
        return output