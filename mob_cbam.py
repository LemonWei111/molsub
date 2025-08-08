__author__ = "Lemon Wei"
__email__ = "Lemon2922436985@gmail.com"
__version__ = "1.1.0"

import torch
import torch.nn as nn
from torchvision import models, ops

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

def _init(module):
    if isinstance(module, nn.Conv2d):
        # 使用Kaiming正态分布初始化卷积层权重
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            # 如果存在偏置，则将其初始化为0
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        # 使用Xavier正态分布初始化全连接层权重
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            # 如果存在偏置，则将其初始化为0
            nn.init.constant_(module.bias, 0)

# 递归函数，用于查找所有的 SqueezeExcitation 层 # CBAM 替换SE（diff from MOB-CBAM）
def find_squeeze_excitation_layers(model, se_layers=None):
    if se_layers is None:
        se_layers = []

    for module in model.features[4:16]:
        sel = module.block[2]
        if isinstance(sel, ops.misc.SqueezeExcitation):
            module.block[2] = CBAM(sel.fc1.in_channels)

    return se_layers, model
    
# CBAM 加载每个块的卷积输出之后
def add_cbam(model):
    model.features[2].block[2] = nn.Sequential(
        model.features[2].block[2],
        CBAM(model.features[3].block[0][0].in_channels)
        )
    model.features[3].block[2] = nn.Sequential(
        model.features[3].block[2],
        CBAM(model.features[4].block[0][0].in_channels)
        )
    model.features[2].block[2][1].apply(_init)
    model.features[3].block[2][1].apply(_init)

    '''
    # Large
    model.features[7].block[2] = nn.Sequential(
        model.features[7].block[2],
        CBAM(model.features[8].block[0][0].in_channels)
        )
    model.features[8].block[2] = nn.Sequential(
        model.features[8].block[2],
        CBAM(model.features[9].block[0][0].in_channels)
        )
    model.features[9].block[2] = nn.Sequential(
        model.features[9].block[2],
        CBAM(model.features[10].block[0][0].in_channels)
        )
    model.features[10].block[2] = nn.Sequential(
        model.features[10].block[2],
        CBAM(model.features[11].block[0][0].in_channels)
        )
    model.features[7].block[2][1].apply(_init)
    model.features[8].block[2][1].apply(_init)
    model.features[9].block[2][1].apply(_init)
    model.features[10].block[2][1].apply(_init)
    '''
    return model

def add_cbam_for_densenet(model):
    model.features.denseblock1 = nn.Sequential(
        model.features.denseblock1,
        CBAM(model.features.transition1.conv.in_channels)
        )
    model.features.denseblock2 = nn.Sequential(
        model.features.denseblock2,
        CBAM(model.features.transition2.conv.in_channels)
        )
    model.features.denseblock3 = nn.Sequential(
        model.features.denseblock3,
        CBAM(model.features.transition3.conv.in_channels)
        )
    '''
    model.features.transition3 = nn.Sequential(
        model.features.transition3,
        CBAM(model.features.denseblock4.denselayer1.conv1.in_channels)
        )
    '''
    model.features.transition1[1].apply(_init)
    model.features.transition2[1].apply(_init)
    model.features.transition3[1].apply(_init)

    return model


if __name__ == '__main__':
    # 加载预训练的 MobileNetV3-Large 模型
    # model = models.mobilenet_v3_large(pretrained=True)
    model = models.mobilenet_v3_small(pretrained=True)
    print(model)

    input('add')
    model = add_cbam(model)
    print(model)

    # 查找所有的 SqueezeExcitation 层
    # se_layers, model = find_squeeze_excitation_layers(model) # large
    # print(model)

