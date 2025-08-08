__author__ = "Lemon Wei"
__email__ = "Lemon2922436985@gmail.com"
__version__ = "1.1.0"

DEBUG = False

import os
import math
import time
import torch
import random
import numpy as np
import torch.nn as nn
import ml_insights as mli
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from torchvision import models
from vit_pytorch.mobile_vit import MobileViT
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, roc_curve

from betacal import BetaCalibration
from sklearn.linear_model import LogisticRegression

from bts_st import BTS_ST
from MACTFusion import ShallowClassifier, DeepClassifier
from mob_cbam import find_squeeze_excitation_layers, add_cbam, add_cbam_for_densenet
from config import cfg
from refers_utils import load_weights
from refers.modeling import VisionTransformer, CONFIGS

from data_loader import get_filtered_loader, CombinedDataset
from mxp_utils.get_loaders import get_combo_loader

from loss import MWNLoss
from torch_losses.soft_f1 import SoftF1LossMulti
from torch_losses.soft_mcc import SoftMCCLossMulti
from torch_losses.combined import WeightedCombinedLosses

from utils import plot_loss_and_accuracy, save_tensors_as_single_image, to_categorical, see_dict_images, swap_first_two_dims, draw_roc, convert_three_to_one_channel, save_confusion_matrix_as_image, create_weight_matrix, get_image_hash, draw_reliability_graph, matrix_metrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 在开发新功能或排查问题时使用
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
# 实验性的 PyTorch 环境变量，
# 启用了 CUDA Direct Stream Access (DSA) 功能。
# 允许 PyTorch 更高效地管理流和事件，特别是在多流或多 GPU 的环境中。
# 可以帮助减少 CPU 和 GPU 之间的同步开销，从而提升性能。
os.environ['TORCH_USE_CUDA_DSA'] = '1'

scaler = GradScaler()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: 模型的输出，形状为 (N, C)，其中 N 是批量大小，C 是类别数
        :param targets: 真实标签，形状为 (N,)，其中 N 是批量大小
        :return: 焦点损失
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# TODO
class WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, n_samples_per_class=None, *args, **kwargs):
        """
        自定义的加权交叉熵损失函数
        """
        if DEBUG:
            print(n_samples_per_class, type(n_samples_per_class))
        N = np.sum(n_samples_per_class)

        # 计算每个类别的权重
        weights = torch.tensor([0.5 * N / n for n in n_samples_per_class], dtype=torch.float)

        # 调用父类的构造函数，将计算好的权重传入
        super(WeightedCrossEntropyLoss, self).__init__(weight=weights, *args, **kwargs)

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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[1, 2, 4]):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        N, C, H, W = x.size()
        for i in range(len(self.pool_sizes)):
            level = self.pool_sizes[i]
            kernel_size = (H // level, W // level)
            stride = (H // level, W // level)
            padding = (H % level // 2, W % level // 2)

            # Apply pooling operation
            tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
            if i == 0:
                spp = tensor.view(N, -1)
            else:
                spp = torch.cat((spp, tensor.view(N, -1)), 1)
        return spp

class AttentionFusion(nn.Module):
    def __init__(self, cnn_dim, feature_dim=1000, hidden_dim=512):
        super(AttentionFusion, self).__init__()
        self.fc_cnn = nn.Linear(cnn_dim, hidden_dim)
        self.fc_feature = nn.Linear(feature_dim, hidden_dim)
        
        # 注意力权重应为一个向量，而不是标量
        self.attn_weight = nn.Parameter(torch.Tensor(hidden_dim, 1))  # 注意：这里改为 (hidden_dim, 1)
        nn.init.xavier_uniform_(self.attn_weight)
        
    def forward(self, cnn_features, features):
        """
        参数:
        - cnn_features: 形状为 [batch_size, cnn_dim] 的张量
        - features: 形状为 [batch_size, feature_dim] 的张量
        
        返回:
        - fused_features: 形状为 [batch_size, hidden_dim] 的融合特征
        """

        transformed_cnn = self.fc_cnn(cnn_features)  # [batch_size, hidden_dim]
        transformed_features = self.fc_feature(features)  # [batch_size, hidden_dim]
        
        # 计算注意力权重
        combined = transformed_cnn + transformed_features  # [batch_size, hidden_dim]
        attn_weights = torch.matmul(combined, self.attn_weight) # .squeeze(-1)  # [batch_size]
        attn_weights = F.softmax(attn_weights, dim=0).unsqueeze(-1)
        # 融合特征
        fused_features = (transformed_cnn * attn_weights + transformed_features * (1 - attn_weights)).sum(dim=0)
        
        return fused_features
'''    
class AttentionFusion(nn.Module):
    def __init__(self, cnn_dim=32, feature_dim=1000, hidden_dim=512):
        super(AttentionFusion, self).__init__()
        self.cnn_dim = cnn_dim
        self.fc_cnn = nn.Linear(cnn_dim, hidden_dim)
        self.fc_feature = nn.Linear(feature_dim, hidden_dim)
        
        # 注意力权重应为一个向量，而不是标量
        self.attn_layer = CrossAttention()
        
    def forward(self, cnn_features, features):
        transformed_cnn = self.fc_cnn(cnn_features.view(features.size(0), -1, self.cnn_dim))  # [batch_size, hidden_dim]
        transformed_features = self.fc_feature(features.unsqueeze(1))  # [batch_size, hidden_dim]
        
        fused_features = self.attn_layer(transformed_cnn, transformed_features, transformed_features)
        
        return fused_features
'''
class CrossAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        # 确保输入张量的形状符合 MultiheadAttention 的要求
        query = query.permute(1, 0, 2)  # 原数据(N, L, E)
        key = key.permute(1, 0, 2)      # 原数据(N, S, E)
        value = value.permute(1, 0, 2)  # 原数据(N, S, E)
        # print(query.shape, key.shape)
        # 应用多头注意力机制
        attn_output, _ = self.multihead_attn(query=query, key=key, value=value)

        # 将输出重新排列回原始形状
        attn_output = attn_output.mean(dim=0) # 如果输入是三维张量，则对时间步长维度取平均
        # print('mean', attn_output.shape)
        return attn_output

class CombinedModelWithAttention(nn.Module):
    def __init__(self, cnn_model, feature_dim=1000, num_classes=10):
        super(CombinedModelWithAttention, self).__init__()
        self.cnn = cnn_model
        self.attention_fusion = AttentionFusion(feature_dim, hidden_dim=512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, images, features):
        cnn_features = self.cnn(images)
        fused_features = self.attention_fusion(cnn_features, features)
        output = self.fc(fused_features)
        return output
    
# 多尺度特征融合  
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
       
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
   
    def forward(self, features):
        assert len(features) == len(self.lateral_convs)
       
        # 侧向连接
        laterals = [lateral_conv(feature) for lateral_conv, feature in zip(self.lateral_convs, features)]
       
        # 自顶向下路径
        used_backbone_levels = len(features)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += nn.functional.interpolate(laterals[i], scale_factor=2, mode='nearest')
       
        # FPN卷积
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        '''
        # 示例
        in_channels_list = [256, 512, 1024]
        out_channels = 256
        fpn = FPN(in_channels_list, out_channels)

        # 假设features是来自不同层次的特征图
        features = [torch.randn(1, c, 64, 64) for c in in_channels_list]
        outs = fpn(features)
        '''
        return outs
    
class LowRankAdaptation(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super(LowRankAdaptation, self).__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.empty(out_features, rank))
        self.B = nn.Parameter(torch.empty(rank, in_features))
        # print(in_features, self.rank, out_features)
        # 使用Kaiming初始化
        nn.init.kaiming_uniform_(self.A)
        nn.init.kaiming_uniform_(self.B)

    def forward(self, x):
        # print(x.shape, self.A.shape, self.B.shape, torch.mm(self.A, self.B).shape)
        return F.linear(x, torch.mm(self.A, self.B))
'''
class SimpleCNN(nn.Module):
    def __init__(self, input_size=224, lora_rank=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # 输入通道变换
        
        # self.lora1 = LowRankAdaptation(16 * (input_size // 2) * (input_size // 2), 16 * (input_size // 2) * (input_size // 2), rank=lora_rank)
        
        # SimpleCNN
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # 不要随便加
        # self.spp1 = SpatialPyramidPooling(pool_sizes=[1, 2]) # , 4])
        # self.fc1 = nn.Linear(32 * (1**2 + 2**2), 128)

        self.fc1 = nn.Linear(32 * (input_size // 4) * (input_size // 4), 128)
        
        self.fc = nn.Linear(128, 2)
        self.input_size = input_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        
        # x = x.view(-1, 16 * (self.input_size // 2) * (self.input_size // 2))  # 展平特征图
        # x = F.relu(self.lora1(x))  # 应用 lora1 层
        # x = x.view(-1, 16, self.input_size // 2, self.input_size // 2)  # 重塑为 (N, 16, H//2, W//2)
        
        x = F.relu(self.conv2(x))

        # x = self.spp1(x) # 应用SPP

        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * (self.input_size // 4) * (self.input_size // 4)) # 展平特征图
        
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        return x
'''
class SimpleCNN(nn.Module):
    def __init__(self, input_size=224, lora_rank=4):
        super(SimpleCNN, self).__init__()
        # 使用nn.Sequential定义卷积层、激活函数和池化层
        self.features = nn.Sequential(
            # 第一层：卷积 + ReLU + 最大池化
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二层：卷积 + ReLU + 最大池化
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # SimpleCNN
            # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2, dilation=2), # 使用空洞卷积来扩大感受野
            # nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 增大步长
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 全连接层，用于分类
        self.classifier = nn.Sequential(
            nn.Linear(32 * (input_size // 4) * (input_size // 4), 128), # SimpleCNN/使用空洞卷积来扩大感受野
            # nn.Linear(32 * (input_size // 8) * (input_size // 8), 128), # 增大步长
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平特征图以输入到全连接层
        x = self.classifier(x)
        return x

'''
# 简单拼接
class SimpleCNN_with_SIFT(nn.Module):
    def __init__(self, input_size=224, feature_dim=1000, num_classes=2):
        super(SimpleCNN_with_SIFT, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # 输入通道变换
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * (input_size // 4) * (input_size // 4) + feature_dim, 128)
        
        self.fc = nn.Linear(128, num_classes)
        self.input_size = input_size

    def forward(self, x, features):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * (self.input_size // 4) * (self.input_size // 4)) # 展平特征图
        # print(x.dtype, features.dtype) # float16, float64
        combined_features = torch.cat((x, features), dim=1).float()
        output1 = F.relu(self.fc1(combined_features))
        output = self.fc(output1)
        return output
'''
class SimpleCNN_with_SIFT(nn.Module):
    def __init__(self, input_channel=1, input_size=224, feature_dim=1000, num_classes=2, pretrained=False):
        super(SimpleCNN_with_SIFT, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1) # 输入通道变换
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.attention_fusion = AttentionFusion(32 * (self.input_size // 4) * (self.input_size // 4), feature_dim=feature_dim, hidden_dim=512) # Attention 
        # self.attention_fusion = AttentionFusion(feature_dim=feature_dim, hidden_dim=512) # Cross Attention 

        self.fc1 = nn.Linear(512, 128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, features):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        cnn_features = x.view(-1, 32 * (self.input_size // 4) * (self.input_size // 4)) # Attention 展平特征图
        # print(cnn_features.shape, features.shape)
        fused_features = self.attention_fusion(cnn_features.float(), features.float())
        # print(fused_features.shape)
        output1 = F.relu(self.fc1(fused_features))
        # print(output1.shape)
        output = self.fc(output1)
        # print(output.shape)
        return output

class MOBCBAM_with_SIFT(nn.Module):
    def __init__(self, input_channel=1, input_size=224, feature_dim=1000, num_classes=2, pretrained=True):
        super(MOBCBAM_with_SIFT, self).__init__()
        self.input_size = input_size
        self.conv = MOB_CBAM(input_channel=input_channel, num_classes=num_classes)
        self.attention_fusion = AttentionFusion(576 * 7 * 7, feature_dim=feature_dim, hidden_dim=512) # Attention 
        # self.attention_fusion = AttentionFusion(feature_dim=feature_dim, hidden_dim=512) # Cross Attention 

        self.fc1 = nn.Linear(512, 128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, features):
        x = self.conv.model.features(x)
        # print(x.shape)
        cnn_features = x.view(-1, 576 * 7 * 7) # Attention 展平特征图
        # print(cnn_features.shape, features.shape)
        fused_features = self.attention_fusion(cnn_features.float(), features.float())
        # print(fused_features.shape)
        output1 = F.relu(self.fc1(fused_features))
        # print(output1.shape)
        output = self.fc(output1)
        # print(output.shape)
        return output

class DENSE_with_SIFT(nn.Module):
    def __init__(self, input_channel=1, input_size=224, feature_dim=1000, num_classes=2, pretrained=True):
        super(DENSE_with_SIFT, self).__init__()
        self.input_size = input_size
        model = models.densenet121(pretrained=pretrained)
        if input_channel == 1:
            model = convert_three_to_one_channel(model) # 平均权重
        # self.model.features.conv0 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # 随机权重
        model = add_cbam_for_densenet(model) # dense_cbam
        self.conv = model.features
        
        self.attention_fusion = AttentionFusion(1024 * 7 * 7, feature_dim=feature_dim, hidden_dim=512) # Attention 
        # self.attention_fusion = AttentionFusion(feature_dim=feature_dim, hidden_dim=512) # Cross Attention 

        self.fc1 = nn.Linear(512, 128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, features):
        x = self.conv(x)
        # print(x.shape)
        cnn_features = x.view(-1, 1024 * 7 * 7) # Attention 展平特征图
        # print(cnn_features.shape, features.shape)
        fused_features = self.attention_fusion(cnn_features.float(), features.float())
        # print(fused_features.shape)
        output1 = F.relu(self.fc1(fused_features))
        # print(output1.shape)
        output = self.fc(output1)
        # print(output.shape)
        return output

'''
# new
# 注意力
class SimpleCNN_with_SIFT(nn.Module):
    def __init__(self, input_size=224, feature_dim=1000, num_classes=2, pretrained=False, model_path='model/molsub_cnn_HER2_7188.pth'):
        super(SimpleCNN_with_SIFT, self).__init__()
        self.input_size = input_size

        if pretrained:
            base_model = torch.load(model_path)
            self.features = base_model.features
        else:
            self.features = nn.Sequential(
                # 第一层：卷积 + ReLU + 最大池化
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                # nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # 第二层：卷积 + ReLU + 最大池化
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # SimpleCNN
                # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2, dilation=2), # 使用空洞卷积来扩大感受野
                # nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 增大步长
                nn.ReLU(inplace=True),
                # nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.attention_fusion = AttentionFusion(32 * (self.input_size // 4) * (self.input_size // 4), feature_dim=feature_dim, hidden_dim=512)
        
        # 全连接层，用于分类
        self.classifier = nn.Sequential(
            nn.Linear(512, 128), # SimpleCNN/使用空洞卷积来扩大感受野
            # nn.Linear(32 * (input_size // 8) * (input_size // 8), 128), # 增大步长
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, features):
        x = self.features(x)
        cnn_features = x.view(x.size(0), -1)  # 展平特征图
        fused_features = self.attention_fusion(cnn_features.float(), features.float())
        output = self.classifier(fused_features)
        return output
'''
class SimpleCNN_with_SPP(nn.Module):
    def __init__(self, input_size=224, lora_rank=4):
        super(SimpleCNN_with_SPP, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # 输入通道变换
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        self.spp1 = SpatialPyramidPooling(pool_sizes=[1, 2, 4]) # default
        self.fc1 = nn.Linear(32 * (1**2 + 2**2 + 4**2), 128) # default
        # self.spp1 = SpatialPyramidPooling(pool_sizes=[1, 2])
        # self.fc1 = nn.Linear(32 * (1**2 + 2**2), 128)
        
        self.fc = nn.Linear(128, 2)
        self.input_size = input_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = self.spp1(x) # 应用SPP
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        return x

class SimpleCNN_with_SE(nn.Module):
    def __init__(self, input_size=224, lora_rank=4):
        super(SimpleCNN_with_SE, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # 输入通道变换
        self.se = SELayer(channel=16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * (input_size // 4) * (input_size // 4), 128)
        self.fc = nn.Linear(128, 2)
        self.input_size = input_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.se(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * (self.input_size // 4) * (self.input_size // 4)) # 展平特征图
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        return x

class SimpleCNN_with_CBAM(nn.Module):
    def __init__(self, input_size=224, lora_rank=4):
        super(SimpleCNN_with_CBAM, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # 输入通道变换
        self.cbam = CBAM(in_planes=16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * (input_size // 4) * (input_size // 4), 128)
        self.fc = nn.Linear(128, 2)
        self.input_size = input_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.cbam(x) # 应用CBAM
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * (self.input_size // 4) * (self.input_size // 4)) # 展平特征图
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        return x

class ResNetWithLoRA(nn.Module):
    def __init__(self, base_model, lora_rank=16):
        super(ResNetWithLoRA, self).__init__()
        self.base_model = base_model
        num_ftrs = self.base_model.fc.out_features
        # print(base_model, num_ftrs)
        self.lora1 = LowRankAdaptation(num_ftrs, 1024, rank=lora_rank)  # 假设最后一个卷积层的输出通道数为512
        self.fc = nn.Linear(1024, 10)  # 假设分类任务有10个类别

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = F.relu(self.lora1(x))
        x = self.fc(x)
        return x

class DenseNetWithLoRA(nn.Module):
    def __init__(self, base_model, lora_rank=16):
        super(DenseNetWithLoRA, self).__init__()
        self.base_model = base_model
        num_ftrs = self.base_model.classifier.out_features
        # print(base_model, num_ftrs)
        self.lora1 = LowRankAdaptation(num_ftrs, 1024, rank=lora_rank)  # 假设最后一个卷积层的输出通道数为512
        self.fc = nn.Linear(1024, 10)  # 假设分类任务有10个类别

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = F.relu(self.lora1(x))
        x = self.fc(x)
        return x
    
class AttentionMILModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionMILModel, self).__init__()
        # Pre-trained feature extractor (e.g., ResNet)
        self.feature_extractor = SimpleCNN(input_size=224)# 64)# models.densenet121(pretrained=True)
        # self.feature_extractor.features.conv0 = nn.Conv2d(input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # self.feature_extractor.fc = nn.Identity()  # Remove the last fully connected layer
        self.feature_extractor.classifier[2] = nn.Identity()

        print(self.feature_extractor)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, bag):
        features = [self.feature_extractor(instance.unsqueeze(0)) for instance in bag]  # Extract features from each instance
        features = torch.stack(features).squeeze(1)  # Stack and remove unnecessary dimensions
        
        attentions = self.attention(features)
        attention_weights = torch.softmax(attentions, dim=0)
        
        # Weighted sum of the instance representations
        weighted_features = features * attention_weights.view(-1, 1)
        bag_representation = torch.sum(weighted_features, dim=0)
        
        return self.output_layer(bag_representation)

class DenseNet121(nn.Module):
    def __init__(self, input_channel=1):
        super(DenseNet121, self).__init__()
        self.model = models.densenet121(pretrained=True)
        # self.model = add_cbam_for_densenet(self.model) # dense_cbam
        self.model.features.conv0 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model = self.model.features

    def forward(self, x):
        return self.model(x)

class Classifier(nn.Module):
    def __init__(self, backbone, num_class):
        super().__init__()
        self.backbone = backbone
        self.drop_out = nn.Dropout()
        self.classifier = nn.Linear(1024, num_class, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.classifier(x)
        #x = torch.softmax(x, dim=-1)
        return x
    
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.densenet121(pretrained=False)  # 使用densenet121模型
        # DenseNet-121的最后一层是classifier，通常是一个线性层。
        # 如果你想获取特征提取器部分，可以去掉classifier层。
        # 注意，与ResNet不同，DenseNet的结构使得直接切分稍微复杂一些。
        # 下面我们假设你想要去掉最后的分类器层：
        feature_layers = list(base_model.children())[:-1]  # 去掉classifier层
        self.backbone = nn.Sequential(*feature_layers, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
                        
    def forward(self, x):
        return self.backbone(x)

def load_rad_dense121(num_class, input_channel):
    # 实例化backbone和classifier
    backbone = Backbone()
    # 加载预训练的DenseNet-121模型权重
    backbone.load_state_dict(torch.load("checkpoint/DenseNet121.pt", map_location=torch.device('cpu')))
    backbone.backbone[0].conv0 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) if input_channel != 3 else backbone.features.conv0 # 随机权重

    classifier = Classifier(backbone, num_class=num_class)  # 确保num_class已定义
    return classifier

class MOB_CBAM(nn.Module):
    def __init__(self, input_channel=1, num_classes=2):
        super(MOB_CBAM, self).__init__()
        from torchvision.models import MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT# IMAGENET1K_V1  # 或者使用 .DEFAULT 获取最新默认权重
        self.model = add_cbam(models.mobilenet_v3_small(weights=weights)) # pretrained=pretrained))

        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)
        self.model.features[0][0] = nn.Conv2d(input_channel, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        return self.model(x)

class VGG16(nn.Module):
    def __init__(self, input_channel=1, num_classes=2):
        super(VGG16, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.features[0] = nn.Conv2d(input_channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        num_ftrs = self.model.classifier[6].in_features # resnet, SimpleCNN
        self.model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.model(x)
    
class GatingNetwork(nn.Module):
    def __init__(self):
        super(GatingNetwork, self).__init__()
        # 可以包含一些卷积层或其他适合于你任务的层
        self.feature_extractor = DenseNet121()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(64, 3) # sc
        self.fc = nn.Linear(1024, 3) # densenet121
    
    def forward(self, x):
        # 计算每个专家的重要性权重
        x = self.feature_extractor(x)
        # print(x.shape)
        x = self.global_pool(x)  # 输出形状: (batch_size, channels, 1, 1)
        x = x.view(x.size(0), -1)  # 展平: (batch_size, channels*1*1)
        weights = self.fc(x)
        weights = F.softmax(weights, dim=1)
        return weights
    
class MixtureOfExperts(nn.Module):
    def __init__(self):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([VGG16(), SimpleCNN_with_SIFT(), MOB_CBAM()])
        self.gating_network = GatingNetwork()
        
    def forward(self, x, features):
        gating_outputs = self.gating_network(x)
        outputs = None
        for i, expert in enumerate(self.experts):
            try:
                expert_output = expert(x)
            except:
                expert_output = expert(x, features)
            if outputs is None:
                outputs = gating_outputs[:, i].unsqueeze(-1) * expert_output
            else:
                outputs += gating_outputs[:, i].unsqueeze(-1) * expert_output
        return outputs
    
# cited from refers   
class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed) # added on 1205
        torch.cuda.manual_seed_all(args.seed)

class MolSub(nn.Module):
    def __init__(self, train_loader, val_loader, args, num_classes=5, device=device):
        super(MolSub, self).__init__()
        learning_rate = args.lr
        weight_decay = args.decay
        momentum = args.momentum
        pretrained = args.pretrain
        dropout = args.dropout
        self.model_type = args.model_type
        input_channel = args.input_channel
        self.loss_type = args.loss_type

        self.device = device
        self.num_classes = num_classes
        if self.model_type == 'refers':
            config = CONFIGS['ViT-B_16']
            self.model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
            self.model = load_weights(self.model, args.pretrained_dir)
            self.model.transformer.embeddings.patch_embeddings = nn.Conv2d(input_channel, 768, kernel_size=(16, 16), stride=(16, 16))
            # for param in self.model.parameters():
            #     param.requires_grad = False
            # for param in self.model.head.parameters():
            #     param.requires_grad = True
            
        elif self.model_type == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
            # print(self.model)

            # 修改输入层的通道数
            self.model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            # 修改输出层的通道数
            num_ftrs = self.model.fc.in_features # resnet, SimpleCNN
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        elif self.model_type == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            num_ftrs = self.model.fc.in_features # resnet, SimpleCNN
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
            # print(self.model)
            # for param in self.model.parameters():
            #     param.requires_grad = False
            # for param in self.model.fc.parameters():
            #     param.requires_grad = True

        elif self.model_type == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
            self.model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            num_ftrs = self.model.fc.in_features # resnet, SimpleCNN
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        elif self.model_type == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            self.model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            num_ftrs = self.model.fc.in_features # resnet, SimpleCNN
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        elif self.model_type == 'cnn':
            pretrained = 0
            self.model = SimpleCNN()
            '''
            num_ftrs = self.model.fc.in_features # resnet, SimpleCNN
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
            self.model.conv1 = nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1) # 输入通道变换
            '''
            num_ftrs = self.model.classifier[2].in_features # resnet, SimpleCNN
            self.model.classifier[2] = nn.Linear(num_ftrs, self.num_classes, bias=True)
            self.model.features[0] = nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1) # 输入通道变换

        elif self.model_type == 'densenet121':
            self.model = models.densenet121(pretrained=pretrained)
            if input_channel == 1:
                self.model = convert_three_to_one_channel(self.model) # 平均权重

            num_ftrs = self.model.classifier.in_features # densenet
            self.model.classifier = nn.Linear(num_ftrs, self.num_classes, bias=True)

        elif self.model_type == 'densenet121-cbam':
            self.model = models.densenet121(pretrained=pretrained)
            # print(self.model)
            # self.model.denseblock4 = nn.Sequential()
            # self.model.norm5 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            
            if input_channel == 1:
                self.model = convert_three_to_one_channel(self.model) # 平均权重
                # self.model.features.conv0 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # 随机权重
            
            self.model = add_cbam_for_densenet(self.model) # dense_cbam

            num_ftrs = self.model.classifier.in_features # densenet
            self.model.classifier = nn.Linear(num_ftrs, self.num_classes, bias=True)
            
            # for param in self.model.features.parameters():
            #     param.requires_grad = False

        elif self.model_type == 'rad_dense':
            self.model = load_rad_dense121(self.num_classes, input_channel)

        elif self.model_type == 'densenet169':
            self.model = models.densenet169(pretrained=pretrained)
            num_ftrs = self.model.classifier.in_features # densenet
            self.model.classifier = nn.Linear(num_ftrs, self.num_classes)
            self.model.features.conv0 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        elif self.model_type == 'lora':
            # self.model = ResNetWithLoRA(models.resnet101(pretrained=pretrained))
            self.model = DenseNetWithLoRA(models.densenet121(pretrained=pretrained))
            # 冻结预训练模型的参数
            for param in self.model.base_model.parameters():
                param.requires_grad = False
                
            print(self.model)
            num_ftrs = 1024 # ResNetWithLoRA
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
            
            # self.model.base_model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # for param in self.model.base_model.conv1.parameters():
            #     param.requires_grad = True
            self.model.base_model.features.conv0 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            for param in self.model.base_model.features.conv0.parameters():
                param.requires_grad = True

        elif self.model_type == 'mobilevit':
            pretrained = 0
            self.model = MobileViT(
                image_size=(256, 256),  # 图像尺寸
                dims=[96, 120, 144],    # 模型维度
                channels=[16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],  # 通道数
                num_classes=num_classes           # 分类数目
            )
            self.model.conv1[0] = nn.Conv2d(input_channel, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        elif self.model_type == 'mob-cbam':
            # MOB-CBAM
            # _, self.model = find_squeeze_excitation_layers(models.mobilenet_v3_large(pretrained=pretrained))
            # self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, self.num_classes)
            # self.model.features[0][0] = nn.Conv2d(input_channel, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            from torchvision.models import MobileNet_V3_Small_Weights
            weights = MobileNet_V3_Small_Weights.DEFAULT# IMAGENET1K_V1  # 或者使用 .DEFAULT 获取最新默认权重
            self.model = add_cbam(models.mobilenet_v3_small(weights=weights)) # pretrained=pretrained))
            # self.model = add_cbam(models.mobilenet_v3_large(pretrained=pretrained))
            '''
            print(self.model)
            for i in range(5, 12):
                self.model.features[i] = nn.Sequential()
            self.model.features[12][0] = nn.Conv2d(40, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
            '''
            self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, self.num_classes)
            
            if input_channel == 1:
                # self.model = convert_three_to_one_channel(self.model, self.model_type)# 平均权重
                self.model.features[0][0] = nn.Conv2d(input_channel, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

            # for param in self.model.features.parameters():
            #     param.requires_grad = False

        elif self.model_type == 'mob_v2':
            # 加载预训练的 MobileNetV2
            self.model = models.mobilenet_v2(pretrained=True)
            self.model.features[0][0] = nn.Conv2d(input_channel, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            num_ftrs = self.model.classifier[1].in_features # resnet, SimpleCNN
            self.model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)
    
        elif self.model_type == 'shuffle_v2':
            # 加载预训练的 ShuffleNetV2 (0.5x 版本，更小)
            self.model = models.shufflenet_v2_x0_5(pretrained=True)
            # print(self.model)
            # self.model.stage4 = nn.Sequential()
            # self.model.conv5[0] = nn.Conv2d(96, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

            num_ftrs = self.model.fc.in_features # resnet, SimpleCNN
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
            self.model.conv1[0] = nn.Conv2d(input_channel, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        elif self.model_type == 'snet':
            self.model = models.squeezenet1_0(pretrained=True)
            self.model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
            self.model.features[0] = nn.Conv2d(input_channel, 96, kernel_size=(7, 7), stride=(2, 2))

        elif self.model_type == 'bts_st':
            self.model = BTS_ST(num_classes=self.num_classes, pretrained=True) if pretrained else BTS_ST(num_classes=self.num_classes, pretrained=False)
            self.model.swin.patch_embed.proj = nn.Conv2d(input_channel, 128, kernel_size=(4, 4), stride=(4, 4))

        elif self.model_type == 'vgg16':
            self.model = models.vgg16(pretrained=True) if pretrained else models.vgg16(pretrained=False)
            self.model.features[0] = nn.Conv2d(input_channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            num_ftrs = self.model.classifier[6].in_features # resnet, SimpleCNN
            self.model.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=self.num_classes, bias=True)

        elif self.model_type == 'mil':
            self.model = AttentionMILModel(input_dim=1, hidden_dim=128, output_dim=2) # 未全局初始化

        elif self.model_type == 'sc':
            pretrained = 0
            self.model = ShallowClassifier(in_channels=input_channel, growth_rate=32, num_classes=num_classes)

        elif self.model_type == 'dc':
            pretrained = 0
            self.model = DeepClassifier(in_channels=input_channel, num_classes=num_classes)

        elif self.model_type == 'cnn+sift':
            pretrained = 0
            self.model = SimpleCNN_with_SIFT(input_channel=input_channel, num_classes=self.num_classes, pretrained=pretrained)
            '''
            # new
            if not pretrained:
                self.model.features[0] = nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1) # 输入通道变换
            '''
            # num_ftrs = self.model.fc.in_features # resnet, SimpleCNN
            # self.model.fc = nn.Linear(num_ftrs, self.num_classes)
            # self.model.conv1 = nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1) # 输入通道变换

        elif self.model_type == 'mob+sift':
            pretrained = 0
            self.model = MOBCBAM_with_SIFT(input_channel=input_channel, num_classes=self.num_classes, pretrained=pretrained)

        elif self.model_type == 'dense+sift':
            pretrained = 0
            self.model = DENSE_with_SIFT(input_channel=input_channel, num_classes=self.num_classes, pretrained=pretrained)

        elif self.model_type == 'cnn+spp':
            pretrained = 0
            self.model = SimpleCNN_with_SPP()
            num_ftrs = self.model.fc.in_features # resnet, SimpleCNN
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
            self.model.conv1 = nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1) # 输入通道变换

        elif self.model_type == 'cnn+se':
            pretrained = 0
            self.model = SimpleCNN_with_SE()
            num_ftrs = self.model.fc.in_features # resnet, SimpleCNN
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
            self.model.conv1 = nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1) # 输入通道变换

        elif self.model_type == 'cnn+cbam':
            pretrained = 0
            self.model = SimpleCNN_with_CBAM()
            num_ftrs = self.model.fc.in_features # resnet, SimpleCNN
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
            self.model.conv1 = nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1) # 输入通道变换

        elif self.model_type == 'moe':
            pretrained = 0
            self.model = MixtureOfExperts() # input_channel 1, no drop
        elif self.model_type == 'mamba':
            # TODO.
            from MedMamba import VSSM as medmamba
            self.model = medmamba(num_classes=self.num_classes)

        if DEBUG:
            print(self.model)

        if not pretrained:
            # Kaiming 初始化
            def kaiming_init(module):
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            # 应用初始化方法
            self.model.apply(kaiming_init)
            
        # print(self.model)
        # input('check')

        # ATTENTION!别忘了！在第一个卷积层之后添加 dropout 层
        if dropout:
            if 'resnet' in self.model_type:
                # resnet
                self.model.layer1[0].conv1 = nn.Sequential(
                    self.model.layer1[0].conv1,
                    nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                )
            elif 'densenet' in self.model_type:
                # densenet
                self.model.features.conv0 = nn.Sequential(
                    self.model.features.conv0,
                    nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                )
            elif self.model_type == 'rad_dense':
                self.model.backbone.backbone[0].conv0 = nn.Sequential(
                    self.model.backbone.backbone[0].conv0,
                    nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                )
            elif self.model_type in ['cnn']:# , 'cnn+sift'] and (not pretrained): # new
                self.model.features[0] = nn.Sequential(
                    self.model.features[0],
                    nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                )
            elif 'cnn' in self.model_type:
                # SimpleCNN
                self.model.conv1 = nn.Sequential(
                    self.model.conv1,
                    nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                )

            elif self.model_type == 'lora':
                # ResNetWithLoRA
                # self.model.base_model.layer1[0].conv1 = nn.Sequential(
                #     self.model.base_model.layer1[0].conv1,
                #     nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                # )
                # DenseNetWithLoRA
                self.model.base_model.features.conv0 = nn.Sequential(
                    self.model.base_model.features.conv0,
                    nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                )
            
            elif self.model_type == 'refers':
                self.model.transformer.embeddings.patch_embeddings.dropout = nn.Dropout2d(p=dropout)  # 更换为 2D dropout 层

            elif self.model_type == 'mobilevit':
                self.model.conv1[0] = nn.Sequential(
                    self.model.conv1[0],
                    nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                )

            elif self.model_type == 'mob+sift':
                self.model.conv.model.features[0][0] = nn.Sequential(
                    self.model.conv.model.features[0][0],
                    nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                )

            elif self.model_type == 'dense+sift':
                self.model.conv.conv0 = nn.Sequential(
                    self.model.conv.conv0,
                    nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                )

            elif 'mob' in self.model_type:
                # MOB-CBAM
                self.model.features[0][0] = nn.Sequential(
                    self.model.features[0][0],
                    nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                )
            elif self.model_type == 'shuffle_v2':
                self.model.conv1[0] = nn.Sequential(
                    self.model.conv1[0],
                    nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                )
            elif self.model_type == 'snet':
                self.model.features[0] = nn.Sequential(
                    self.model.features[0],
                    nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                )
            elif self.model_type == 'bts_st':
                self.model.swin.patch_embed.proj = nn.Sequential(
                    self.model.swin.patch_embed.proj,
                    nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                )
            elif self.model_type == 'vgg16':
                self.model.features[0] = nn.Sequential(
                    self.model.features[0],
                    nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                )
            elif self.model_type == 'mil': # feature_extractor: SimpleCNN
                self.model.feature_extractor.features[0] = nn.Sequential(
                    self.model.feature_extractor.features[0],
                    nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                )
            elif self.model_type == 'sc':
                self.model.features.denseblock1.conv1 = nn.Sequential(
                    self.model.features.denseblock1.conv1,
                    nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                )
            elif self.model_type == 'dc':
                self.model.features.sd_conv.conv1 = nn.Sequential(
                    self.model.features.sd_conv.conv1,
                    nn.Dropout2d(p=dropout)  # 添加 2D dropout 层
                )

        self.model = self.model.to(self.device)
        
        if DEBUG:
            print(self.model)

        if self.loss_type == 'bce':
            self.criterion = torch.nn.BCEWithLogitsLoss() # one-hot label
        elif self.loss_type == 'wce':
            self.criterion = WeightedCrossEntropyLoss(train_loader.dataset.class_counts)
        elif 'ce' in self.loss_type:
            self.criterion = nn.CrossEntropyLoss()
            if 'sf1' in self.loss_type:
                self.criterion1 = SoftF1LossMulti(self.num_classes) # one-hot label
        elif self.loss_type == 'focal':
            self.criterion = FocalLoss()
        elif self.loss_type == 'mwnl':
            para_dict = {
                "num_class_list": [1, 1],
                "device": device,
                "cfg": cfg
                }
            self.criterion = MWNLoss(para_dict)
        elif self.loss_type == 'sf1':
            self.criterion = SoftF1LossMulti(self.num_classes) # one-hot label
        elif self.loss_type == 'mcc':
            self.criterion = SoftMCCLossMulti() # one-hot label
        elif self.loss_type == 'wcl':
            self.criterion = WeightedCombinedLosses()

        self.criterion = self.criterion.to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 初始化早停和学习率调度器
        self.early_stopping_patience = args.early_stopping_patience
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=20, eta_min=1e-05) # 20, eta_min=1e-5)
        # self.scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps=150, t_total=args.num_epochs)
        '''{
            "lr": 0.00479,
            "momentum": 0.84,
            "weight_decay": 0.005,
            "early_stopping_patience": 9.0,
            "t_max": 118.56225,
            "eta_min": 0.00046
        }
        best hyps {'lr': 0.01, 'momentum': 0.7, 'weight_decay': 0.01, 'early_stopping_patience': 6, 't_max': 119, 'eta_min': 1e-06, 
        'dropout': 0, 'batch_size': 51, 'oversample_ratio': 1, 'downsample_ratio': 0}
        
        huigu l:
            "lr": 0.00014,
            "momentum": 0.97556,
            "weight_decay": 0.00432,
            "early_stopping_patience": 15,
            "t_max": 52,
            "eta_min": 1e-05,
            "dropout": 0.30705,
            "batch_size": 9,
            "oversample_ratio": 1.41645,
            "downsample_ratio": 0.0
        '''

    def merge_adjacent_labels(self, outputs, labels):
        """
        # TODO. wrong, 训练数据是打乱的
        视图组合方案2：合并相邻的相同标签，并对相应的输出进行加和和归一化。
        
        :param outputs: 模型的输出，形状为 (batch_size, num_classes)
        :param labels: 标签，形状为 (batch_size,)
        :return: 新的输出和标签
        """
        new_outputs = []
        new_labels = []
        i = 0
        while i < len(labels):
            j = i + 1
            if j < len(labels) and labels[j] == labels[i]:
                j += 1
            # 提取需要合并的输出
            to_merge = outputs[i:j]
            # 合并输出
            merged_output = torch.sum(to_merge, dim=0)
            # 归一化
            normalized_output = merged_output / torch.sum(merged_output)
            new_outputs.append(normalized_output)
            new_labels.append(labels[i])
            i = j
        return torch.stack(new_outputs).to(self.device), torch.tensor(new_labels).to(self.device)

    def train_model(self, args, model_path='model/mosub.pth', filename = 'logging.png'):
        '''
        # For Tuning
        try:
            scaler = GradScaler()
            self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.decay)
            # self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
            
            # 初始化早停和学习率调度器
            self.early_stopping_patience = args.early_stopping_patience
            self.best_val_loss = float('inf')
            self.early_stopping_counter = 0
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.t_max, eta_min=args.eta_min)
            # self.scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps=150, t_total=args.num_epochs)

            if self.model_type == 'mob-cbam':
                # MOB-CBAM
                # _, self.model = find_squeeze_excitation_layers(models.mobilenet_v3_large(pretrained=pretrained))
                # self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, self.num_classes)
                # self.model.features[0][0] = nn.Conv2d(input_channel, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                from torchvision.models import MobileNet_V3_Small_Weights
                weights = MobileNet_V3_Small_Weights.DEFAULT# IMAGENET1K_V1  # 或者使用 .DEFAULT 获取最新默认权重
                self.model = add_cbam(models.mobilenet_v3_small(weights=weights)) # pretrained=pretrained))
                # self.model = add_cbam(models.mobilenet_v3_large(pretrained=pretrained))

                self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, self.num_classes)
                self.model.features[0][0] = nn.Conv2d(args.input_channel, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

                self.model.features[0][0] = nn.Sequential(
                    self.model.features[0][0],
                    nn.Dropout2d(p=args.dropout)  # 添加 2D dropout 层
                )
                self.model = self.model.to(self.device)

        except:
            print('no new args')
        '''
        torch.cuda.empty_cache()

        num_epochs = args.num_epochs
        save_epoch = args.save_epoch

        log_train_loss = []
        log_val_loss = []
        log_train_acc = []
        log_val_acc = []
        last_save_epoch = 0
        debug = False
        filtered_inputs = []
        filtered_labels = []
        for epoch in range(num_epochs):
            if self.model_type in ['densenet121', 'mob-cbam'] and epoch > num_epochs // 3.5:
                for param in self.model.features.parameters():
                    param.requires_grad = True

            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            if self.loss_type == 'mwnl':
                self.criterion.reset_epoch(epoch) # MWNLoss
            
            # 每个epoch都有训练和验证阶段
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # 设置模型为训练模式
                    dataloader = self.train_loader
                else:
                    self.model.eval()   # 设置模型为评估模式
                    dataloader = self.val_loader
                
                running_loss = 0.0
                running_corrects = 0
                
                # 迭代数据
                for batch in dataloader:
                    if self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                        inputs, features, labels = batch[0], batch[1], batch[2]
                        features = features.to(self.device)
                    else:
                        inputs, labels = batch[0], batch[1]

                    # 清零参数梯度
                    self.optimizer.zero_grad()

                    if (DEBUG or debug) and phase == 'train':
                        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                            print("Input data contains NaN or Inf.")
                            break
                        save_tensors_as_single_image(inputs[:3]) # 可视化输入图像
                        if labels.size(0) in torch.bincount(labels): # 检查每个batch的类别是否单一
                            print(f'one class {labels}')
                            # input('one class')
                    
                    if self.loss_type == 'mwnl':
                        self.criterion.num_class_list = np.bincount(labels) # MWNLoss

                    labels = labels.to(self.device)
                    onehot_labels = F.one_hot(labels, self.num_classes).to(torch.float32)
                    
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1) # 梯度裁剪

                    set_seed(args) # remove after loader to here on 1204
                    
                    # 前向传播
                    outputs = []
                    with torch.set_grad_enabled(phase == 'train'):
                        # with autocast():
                        with torch.amp.autocast('cuda'):
                            if self.model_type == 'mil':
                                inputs = swap_first_two_dims(inputs)
                                if DEBUG:
                                    print(len(inputs))
                                    print(inputs[0][0].size())

                                for bag in inputs:
                                    bag_tensor = [instance.to(device) for instance in bag]
                                    # print(len(bag_tensor))
                                    output = self.model(bag_tensor)
                                    outputs.append(output)
                                outputs = torch.stack(outputs).requires_grad_(phase == 'train').to(device)
                            elif self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                                inputs = inputs.to(self.device)
                                outputs = self.model(inputs, features)
                            else:
                                inputs = inputs.to(self.device)
                                outputs = self.model(inputs)

                            # 视图组合方案2：合并相邻的相同标签和输出
                            # outputs, labels = self.merge_adjacent_labels(outputs, labels)
                            if DEBUG or debug:
                                if torch.isnan(outputs).any():
                                    for name, module in self.model.named_children():
                                        inputs = module(inputs)
                                        print(f"Layer: {name}, Output: {inputs}")
                                        if torch.isnan(inputs).any():
                                            print(f"NaN detected in layer: {name}.")
                                            break
                                    print(outputs.shape)
                                    input("enter to continue")

                            score = torch.softmax(outputs, 1)
                            max_probs, preds = torch.max(score, 1)                            

                            if DEBUG:
                                print(outputs)
                                print(score) # 预测概率
                                print(preds)
                                print(labels)
                                # input('enter')
                            
                            if self.loss_type in ['sf1', 'bce', 'mcc', 'wcl']:
                                loss = self.criterion(outputs, onehot_labels)
                            elif self.loss_type == 'ce+sf1':
                                loss = self.criterion(outputs, labels) + self.criterion1(outputs, onehot_labels)
                            else:
                                loss = self.criterion(outputs, labels) # * 3.0, labels)
                            
                            if phase == 'train' and args.train_diff and epoch > save_epoch:
                                # 记录难样本
                                difficult_indices = (max_probs < args.train_diff).nonzero(as_tuple=True)[0]
                                if len(difficult_indices) > 0:
                                    filtered_inputs.append(inputs[difficult_indices])
                                    filtered_labels.append(labels[difficult_indices])

                        # 只有在训练阶段才执行反向传播和优化（注意缩进位置）
                        if phase == 'train':
                            
                            # print(loss, self.optimizer)
                            scaler.scale(loss).backward()
                            scaler.unscale_(self.optimizer) # added on 1212
                            scaler.step(self.optimizer)
                            scaler.update()
                            '''
                            loss.backward()
                            self.optimizer.step()
                            '''
                    # 统计损失和准确率
                    running_loss += loss.item() * len(inputs) # inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)
                
                if phase == 'train':
                    log_train_loss.append(epoch_loss)
                    log_train_acc.append(epoch_acc.cpu())
                    if args.train_debug:
                        if epoch > save_epoch and len(set(log_train_loss[-5:])) <= 1:
                            debug = True
                            print("train acc has not changed for 5 epoch, check if a mistake")
                        else:
                            debug = False
                else:
                    log_val_loss.append(epoch_loss)
                    log_val_acc.append(epoch_acc.cpu())

                # 更新学习率
                if phase == 'train':
                    self.scheduler.step()
                
                # 早停策略
                if phase == 'val':
                    if epoch_loss < self.best_val_loss:
                        self.best_val_loss = epoch_loss
                        self.early_stopping_counter = 0
                        if epoch - last_save_epoch >= save_epoch:
                            self.save_model(model_path.rsplit('_')[0] + '_best_loss.pth')
                            last_save_epoch = epoch
                    else:
                        self.early_stopping_counter += 1
                        if self.early_stopping_counter >= self.early_stopping_patience:
                            print("Early stopping")
                            self.save_model(model_path)
                            plot_loss_and_accuracy(log_train_loss, log_val_loss, log_train_acc, log_val_acc, filename=filename)
                            if args.train_diff:
                                self.filtered_loader = get_filtered_loader(args, filtered_inputs, filtered_labels) # 难分类样本做了2次增强
                                self.train_diff(args, model_path)
                            return
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                     
            if not epoch % save_epoch:
                self.save_model(model_path)
        # print(filename)
        plot_loss_and_accuracy(log_train_loss, log_val_loss, log_train_acc, log_val_acc, filename=filename)
        if args.train_diff:
            self.filtered_loader = get_filtered_loader(args, filtered_inputs, filtered_labels) # 难分类样本做了2次增强
            self.train_diff(args, model_path)

    def train_models(self, args, model_path='model/mosub.pth', filename = 'logging.png'):
        model2 = SimpleCNN()
        num_ftrs = model2.classifier[2].in_features # resnet, SimpleCNN
        model2.classifier[2] = nn.Linear(num_ftrs, self.num_classes, bias=True)
        model2.features[0] = nn.Conv2d(args.input_channel, 16, kernel_size=3, stride=1, padding=1) # 输入通道变换
        # Kaiming 初始化
        def kaiming_init(module):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        # 应用初始化方法
        model2.apply(kaiming_init)
        model2.to(device)
        optimizer2 = optim.AdamW(model2.parameters(), lr=args.lr, weight_decay=args.decay)
        
        torch.cuda.empty_cache()

        num_epochs = args.num_epochs
        save_epoch = args.save_epoch

        log_train_loss = []
        log_val_loss = []
        log_train_acc = []
        log_val_acc = []
        last_save_epoch = 0
        debug = False
        filtered_inputs = []
        filtered_labels = []
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            if self.loss_type == 'mwnl':
                self.criterion.reset_epoch(epoch) # MWNLoss
            
            # 每个epoch都有训练和验证阶段
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # 设置模型为训练模式
                    model2.train()  # 设置模型为训练模式
                    dataloader = self.train_loader
                else:
                    self.model.eval()   # 设置模型为评估模式
                    model2.eval()
                    dataloader = self.val_loader
                
                running_loss = 0.0
                running_corrects = 0
                
                # 迭代数据
                for batch in dataloader:
                    if self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                        inputs, features, labels = batch[0], batch[1], batch[2]
                        features = features.to(self.device)
                    else:
                        inputs, labels = batch[0], batch[1]

                    # 清零参数梯度
                    self.optimizer.zero_grad()
                    optimizer2.zero_grad()

                    if (DEBUG or debug) and phase == 'train':
                        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                            print("Input data contains NaN or Inf.")
                            break
                        save_tensors_as_single_image(inputs[:3]) # 可视化输入图像
                        if labels.size(0) in torch.bincount(labels): # 检查每个batch的类别是否单一
                            print(f'one class {labels}')
                            # input('one class')
                    
                    if self.loss_type == 'mwnl':
                        self.criterion.num_class_list = np.bincount(labels) # MWNLoss

                    labels = labels.to(self.device)
                    onehot_labels = F.one_hot(labels, self.num_classes).to(torch.float32)
                    
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1) # 梯度裁剪

                    set_seed(args) # remove after loader to here on 1204
                    
                    # 前向传播
                    outputs = []
                    outputs2 = []
                    with torch.set_grad_enabled(phase == 'train'):
                        # with autocast():
                        with torch.amp.autocast('cuda'):
                            if self.model_type == 'mil':
                                inputs = swap_first_two_dims(inputs)
                                if DEBUG:
                                    print(len(inputs))
                                    print(inputs[0][0].size())

                                for bag in inputs:
                                    bag_tensor = [instance.to(device) for instance in bag]
                                    # print(len(bag_tensor))
                                    output = self.model(bag_tensor)
                                    output2 = model2(bag_tensor)
                                    outputs.append(output)
                                    outputs2.append(output2)
                                outputs = torch.stack(outputs).requires_grad_(phase == 'train').to(device)
                                outputs2 = torch.stack(outputs2).requires_grad_(phase == 'train').to(device)
                            elif self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                                inputs = inputs.to(self.device)
                                outputs = self.model(inputs, features)
                                outputs2 = model2(inputs, features)
                            else:
                                inputs = inputs.to(self.device)
                                outputs = self.model(inputs)
                                outputs2 = model2(inputs)

                            # 视图组合方案2：合并相邻的相同标签和输出
                            # outputs, labels = self.merge_adjacent_labels(outputs, labels)
                            if DEBUG or debug:
                                if torch.isnan(outputs).any():
                                    for name, module in self.model.named_children():
                                        inputs = module(inputs)
                                        print(f"Layer: {name}, Output: {inputs}")
                                        if torch.isnan(inputs).any():
                                            print(f"NaN detected in layer: {name}.")
                                            break
                                    print(outputs.shape)
                                    input("enter to continue")

                            score = torch.softmax(outputs, 1)
                            score2 = torch.softmax(outputs2, 1)
                            final_score = (score + score2) / 2
                            max_probs, preds = torch.max(final_score, 1)                      

                            if DEBUG:
                                print(outputs)
                                print(score) # 预测概率
                                print(preds)
                                print(labels)
                                # input('enter')
                            
                            if self.loss_type in ['sf1', 'bce', 'mcc', 'wcl']:
                                loss = self.criterion(outputs, onehot_labels) + self.criterion(outputs2, onehot_labels)
                            elif self.loss_type == 'ce+sf1':
                                loss = self.criterion(outputs, labels) + self.criterion1(outputs, onehot_labels) + self.criterion(outputs2, labels) + self.criterion1(outputs2, onehot_labels)
                            else:
                                loss = self.criterion(outputs, labels) + self.criterion(outputs2, labels)
                            
                            if phase == 'train' and args.train_diff and epoch > save_epoch:
                                # 记录难样本
                                difficult_indices = (max_probs < args.train_diff).nonzero(as_tuple=True)[0]
                                if len(difficult_indices) > 0:
                                    filtered_inputs.append(inputs[difficult_indices])
                                    filtered_labels.append(labels[difficult_indices])

                        # 只有在训练阶段才执行反向传播和优化（注意缩进位置）
                        if phase == 'train':
                            # print(loss, self.optimizer)
                            scaler.scale(loss).backward()
                            scaler.unscale_(self.optimizer) # added on 1212
                            scaler.step(self.optimizer)
                            scaler.unscale_(optimizer2) # added on 1212
                            scaler.step(optimizer2)
                            scaler.update()
                            # loss.backward()
                            # self.optimizer.step()
                    
                    # 统计损失和准确率
                    running_loss += loss.item() * len(inputs) # inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)
                
                if phase == 'train':
                    log_train_loss.append(epoch_loss)
                    log_train_acc.append(epoch_acc.cpu())
                    if args.train_debug:
                        if epoch > save_epoch and len(set(log_train_loss[-5:])) <= 1:
                            debug = True
                            print("train acc has not changed for 5 epoch, check if a mistake")
                        else:
                            debug = False
                else:
                    log_val_loss.append(epoch_loss)
                    log_val_acc.append(epoch_acc.cpu())

                # 更新学习率
                if phase == 'train':
                    self.scheduler.step()
                
                # 早停策略
                if phase == 'val':
                    if epoch_loss < self.best_val_loss:
                        self.best_val_loss = epoch_loss
                        self.early_stopping_counter = 0
                        if epoch - last_save_epoch >= save_epoch:
                            self.save_model(model_path.rsplit('_')[0] + '_best_loss.pth')
                            torch.save(model2, model_path.rsplit('_')[0] + '_model2_best_loss.pth')
                            last_save_epoch = epoch
                    else:
                        self.early_stopping_counter += 1
                        if self.early_stopping_counter >= self.early_stopping_patience:
                            print("Early stopping")
                            self.save_model(model_path)
                            torch.save(model2, model_path)
                            plot_loss_and_accuracy(log_train_loss, log_val_loss, log_train_acc, log_val_acc, filename=filename)
                            if args.train_diff:
                                self.filtered_inputs = filtered_inputs
                                self.filtered_labels = filtered_labels
                                self.train_diff(args, model_path)
                            return model2
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                     
            if not epoch % save_epoch:
                self.save_model(model_path)
                torch.save(model2, model_path)
        # print(filename)
        plot_loss_and_accuracy(log_train_loss, log_val_loss, log_train_acc, log_val_acc, filename=filename)
        if args.train_diff:
            self.filtered_inputs = filtered_inputs
            self.filtered_labels = filtered_labels
            self.train_diff(args, model_path)
        return model2

    def train_diff(self, args, model_path='model/mosub.pth'):    
        torch.cuda.empty_cache()

        num_epochs = args.train_diff_epochs

        diff_log_train_loss = []
        diff_log_train_acc = []
        for epoch in range(num_epochs):
            print(f'难分类 Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            if self.loss_type == 'mwnl':
                self.criterion.reset_epoch(epoch) # MWNLoss
            self.model.train()  # 设置模型为训练模式
            dataloader = self.filtered_loader
            running_loss = 0.0
            running_corrects = 0
            for batch in dataloader:
                if self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                    inputs, features, labels = batch[0], batch[1], batch[2]
                    features = features.to(self.device)
                else:
                    inputs, labels = batch[0], batch[1]
                if self.loss_type == 'mwnl':
                    self.criterion.num_class_list = np.bincount(labels) # MWNLoss
                
                labels = labels.to(self.device)
                onehot_labels = F.one_hot(labels, self.num_classes).to(torch.float32)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1) # 梯度裁剪
                self.optimizer.zero_grad()
                set_seed(args) # remove after loader to here on 1204
                outputs = []
                with torch.set_grad_enabled(True):
                    with torch.amp.autocast('cuda'):

                        if self.model_type == 'mil':
                            inputs = swap_first_two_dims(inputs) 
                            if DEBUG:
                                print(len(inputs))
                                
                            for bag in inputs:
                                bag_tensor = [instance.to(device) for instance in bag]
                                output = self.model(bag_tensor)
                                outputs.append(output)
                            outputs = torch.stack(outputs).requires_grad_(True).to(device)
                        elif self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                            inputs = inputs.to(self.device)
                            outputs = self.model(inputs, features)
                        else:
                            inputs = inputs.to(self.device)
                            outputs = self.model(inputs)

                        score = torch.softmax(outputs, 1)
                        _, preds = torch.max(score, 1)                        
                        if self.loss_type in ['sf1', 'bce', 'mcc', 'wcl']:
                            loss = self.criterion(outputs, onehot_labels)
                        elif self.loss_type == 'ce+sf1':
                            loss = self.criterion(outputs, labels) + self.criterion1(outputs, onehot_labels)
                        else:
                            loss = self.criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()                
                running_loss += loss.item() * len(inputs)# inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            diff_log_train_loss.append(epoch_loss)
            diff_log_train_acc.append(epoch_acc.cpu())
            self.scheduler.step()
            print(f'难分类{epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')     
        self.save_model(model_path)
        
    def train_visual(self, args, unlabeled_loader=None, label_stability_threshold = 3, model_path='model/mosub.pth', filename = 'logging.png'):
        torch.cuda.empty_cache()

        num_epochs = args.num_epochs
        save_epoch = args.save_epoch

        log_train_loss = []
        log_val_loss = []
        log_train_acc = []
        log_val_acc = []
        last_save_epoch = 0
        # 用于存储已经加进来的伪标签样本
        pseudo_labeled_samples = set()
        # 用于存储样本的历史预测信息
        sample_history = {}
        labeled_loader = self.train_loader

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            if self.loss_type == 'mwnl':
                self.criterion.reset_epoch(epoch) # MWNLoss
            
            # 每个epoch都有训练和验证阶段
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # 设置模型为训练模式
                    dataloader = labeled_loader
                else:
                    self.model.eval()   # 设置模型为评估模式
                    dataloader = self.val_loader
                    pseudo_labels = []
                    pseudo_images = []

                running_loss = 0.0
                running_corrects = 0
                
                # 迭代数据
                for batch in dataloader:
                    if self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                        inputs, features, labels = batch[0], batch[1], batch[2]
                        features = features.to(self.device)
                    else:
                        inputs, labels = batch[0], batch[1]
                    # 清零参数梯度
                    self.optimizer.zero_grad()
                    if self.loss_type == 'mwnl':
                        self.criterion.num_class_list = np.bincount(labels) # MWNLoss
                    labels = labels.to(self.device)
                    onehot_labels = F.one_hot(labels, self.num_classes).to(torch.float32)
                    set_seed(args) # remove after loader to here on 1204                    
                    # 前向传播
                    outputs = []
                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.amp.autocast('cuda'):
                            if self.model_type == 'mil':
                                inputs = swap_first_two_dims(inputs)
                                for bag in inputs:
                                    bag_tensor = [instance.to(device) for instance in bag]
                                    # print(len(bag_tensor))
                                    output = self.model(bag_tensor)
                                    outputs.append(output)
                                outputs = torch.stack(outputs).requires_grad_(phase == 'train').to(device)
                            elif self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                                inputs = inputs.to(self.device)
                                outputs = self.model(inputs, features)
                            else:
                                inputs = inputs.to(self.device)
                                outputs = self.model(inputs)
                            score = torch.softmax(outputs, 1)
                            max_probs, preds = torch.max(score, 1)                            
                            
                            if self.loss_type in ['sf1', 'bce', 'mcc', 'wcl']:
                                loss = self.criterion(outputs, onehot_labels)
                            elif self.loss_type == 'ce+sf1':
                                loss = self.criterion(outputs, labels) + self.criterion1(outputs, onehot_labels)
                            else:
                                loss = self.criterion(outputs, labels)

                        # 只有在训练阶段才执行反向传播和优化（注意缩进位置）
                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.unscale_(self.optimizer) # added on 1212
                            scaler.step(self.optimizer)
                            scaler.update()

                    # 统计损失和准确率
                    running_loss += loss.item() * len(inputs) # inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)
                
                if phase == 'train':
                    log_train_loss.append(epoch_loss)
                    log_train_acc.append(epoch_acc.cpu())
                else:
                    log_val_loss.append(epoch_loss)
                    log_val_acc.append(epoch_acc.cpu())
                    
                    '''
                    # logging/output_cnn_usm3_wei_test_qianzhan.log 
                    # 每一轮都筛：（全部）使用当前模型为未标注数据生成伪标签
                    with torch.no_grad():
                        for batch in unlabeled_loader:
                            if self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                                inputs, features, labels = batch[0], batch[1], batch[2]
                                features = features.to(self.device)
                            else:
                                inputs, labels = batch[0], batch[1]
                            outputs = []
                            labels = labels.to(self.device)
                            if self.model_type == 'mil':
                                inputs = swap_first_two_dims(inputs)
                                if DEBUG:
                                    print(len(inputs))
                                    
                                for bag in inputs:
                                    bag_tensor = [instance.to(device) for instance in bag]
                                    output = self.model(bag_tensor)
                                    outputs.append(output)
                                outputs = torch.stack(outputs).to(device)
                            elif self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                                inputs = inputs.to(self.device)
                                outputs = self.model(inputs, features)
                            else:
                                inputs = inputs.to(self.device)
                                outputs = self.model(inputs)
                            score = torch.softmax(outputs, 1)
                            max_probs, preds = torch.max(score, 1)
                            
                            # 只选择置信度超过阈值的样本作为伪标签，并且确保不重复
                            image_hashes = [get_image_hash(img) for img in inputs]
                            mask = (max_probs > args.confidence_threshold)

                            for m, img, img_hash, pred, prob in zip(mask, inputs, image_hashes, preds, max_probs):
                                if not m:
                                    continue
                                if img_hash not in sample_history:
                                    sample_history[img_hash] = [(pred.item(), prob.item())]
                                else:
                                    history = sample_history[img_hash]
                                    if len(history) >= label_stability_threshold:
                                        # 如果历史记录中有足够的相同预测，更新伪标签
                                        if all(h[0] == pred.item() for h in history[-label_stability_threshold:]):
                                            history.append((pred.item(), prob.item()))
                                        else:
                                            # 不更新伪标签，继续使用旧的预测
                                            pred = history[-1][0]
                                            prob = history[-1][1]
                                    else:
                                        history.append((pred.item(), prob.item()))
                                    
                                    # 确保只保留最近的历史记录
                                    sample_history[img_hash] = history[-label_stability_threshold:]
                                
                                if img_hash in pseudo_labeled_samples:
                                    continue
                                else:
                                    pseudo_images.append(img)
                                    pseudo_labels.append(pred)
                                    # 更新已经加进来的伪标签样本集合
                                    pseudo_labeled_samples.add(img_hash)                     

                    # 将新的伪标签数据与原始标记数据合并
                    if len(pseudo_labels) > 0:
                        pseudo_images =  torch.stack(pseudo_images) # if len(pseudo_images) > 1 else pseudo_images[0]
                        pseudo_labels =  torch.stack(pseudo_labels) # if len(pseudo_labels) > 1 else pseudo_labels[0]
                        print(pseudo_images.shape, pseudo_labels.shape)
                        pseudo_dataset = TensorDataset(pseudo_images.cpu(), pseudo_labels.cpu())
                        
                        combined_dataset = CombinedDataset(labeled_loader.dataset, pseudo_dataset)
                        labeled_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)
                        print(f"Added {len(pseudo_labels)} pseudo-labeled samples.")
                    '''
                # 更新学习率
                if phase == 'train':
                    self.scheduler.step()
                
                # 早停策略
                if phase == 'val':
                    if epoch_loss < self.best_val_loss:
                        self.best_val_loss = epoch_loss
                        self.early_stopping_counter = 0
                        if epoch - last_save_epoch >= save_epoch:
                            self.save_model(model_path.rsplit('_')[0] + '_best_loss.pth')
                            last_save_epoch = epoch
                    else:
                        self.early_stopping_counter += 1
                        if self.early_stopping_counter >= self.early_stopping_patience:
                            print("Early stopping")
                            self.save_model(model_path)
                            plot_loss_and_accuracy(log_train_loss, log_val_loss, log_train_acc, log_val_acc, filename=filename)
                            # 训完之后筛，然后train_diff
                            true_labels = []
                            with torch.no_grad():
                                for batch in unlabeled_loader:
                                    if self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                                        inputs, features, labels = batch[0], batch[1], batch[2]
                                        features = features.to(self.device)
                                    else:
                                        inputs, labels = batch[0], batch[1]
                                    outputs = []
                                    if self.model_type == 'mil':
                                        inputs = swap_first_two_dims(inputs)
                                        for bag in inputs:
                                            bag_tensor = [instance.to(device) for instance in bag]
                                            output = self.model(bag_tensor)
                                            outputs.append(output)
                                        outputs = torch.stack(outputs).to(device)
                                    elif self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                                        inputs = inputs.to(self.device)
                                        outputs = self.model(inputs, features)
                                    else:
                                        inputs = inputs.to(self.device)
                                        outputs = self.model(inputs)
                                    score = torch.softmax(outputs, 1)
                                    max_probs, preds = torch.max(score, 1)
                                    
                                    # 只选择置信度超过阈值的样本作为伪标签
                                    mask = (max_probs > args.confidence_threshold)
                                    filtered_inputs = inputs[mask]
                                    filtered_preds = preds[mask]

                                    labels = labels.to(self.device)
                                    filtered_labels = labels[mask]
                                    if len(filtered_labels):
                                        # 逐元素比较并统计相同元素数量
                                        num_equal_elements = (filtered_preds == filtered_labels).sum().item()
                                        print(f"Number of equal elements: {num_equal_elements}/{len(filtered_labels)}, acc:{num_equal_elements/len(filtered_labels)}")

                                    for img, pred, label in zip(filtered_inputs, filtered_preds, filtered_labels):
                                        # 只保留少数类的伪样本
                                        if True:# pred in labeled_loader.dataset.minority_classes:
                                            pseudo_images.append(img)
                                            pseudo_labels.append(pred)
                                            true_labels.append(label)


                            # 将新的伪标签数据与原始标记数据合并
                            if len(pseudo_labels) > 0:
                                num_equal_elements = sum(x == y for x, y in zip(pseudo_labels, true_labels))
                                print(f"Number of equal min only elements: {num_equal_elements}, acc:{num_equal_elements/len(true_labels)}")
                                pseudo_images =  torch.stack(pseudo_images) # if len(pseudo_images) > 1 else pseudo_images[0]
                                pseudo_labels =  torch.stack(pseudo_labels) # if len(pseudo_labels) > 1 else pseudo_labels[0]
                                pseudo_dataset = TensorDataset(pseudo_images.cpu(), pseudo_labels.cpu())                            
                                combined_dataset = CombinedDataset(labeled_loader.dataset, pseudo_dataset)
                                self.filtered_loader = DataLoader(combined_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
                                print(f"Added {len(pseudo_labels)} pseudo-labeled samples.")
                                del pseudo_images, pseudo_labels, pseudo_dataset, combined_dataset
                                self.train_diff(args, model_path)
                            else:
                                print('没有置信度足够高的样本')
                            return
                        
                        '''
                        if len(pseudo_labeled_samples) <= 200 and epoch - last_save_epoch >= save_epoch: # log_train_acc[-1] >= args.confidence_threshold:
                            # 最多选和训练集一样多，只在# x epoch之后 # 训练准确率大于阈值时选择
                            # logging/output_cnn_usm3_wei_test_qianzhan_whenvalbad.log 
                            # 预测集指标下降时筛：（全部）使用当前模型为未标注数据生成伪标签
                            with torch.no_grad():
                                for batch in unlabeled_loader:
                                    if self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                                        inputs, features = batch[0], batch[1]
                                        features = features.to(self.device)
                                    else:
                                        inputs = batch[0]
                                    outputs = []
                                    if self.model_type == 'mil':
                                        inputs = swap_first_two_dims(inputs)
                                        for bag in inputs:
                                            bag_tensor = [instance.to(device) for instance in bag]
                                            output = self.model(bag_tensor)
                                            outputs.append(output)
                                        outputs = torch.stack(outputs).to(device)
                                    elif self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                                        inputs = inputs.to(self.device)
                                        outputs = self.model(inputs, features)
                                    else:
                                        inputs = inputs.to(self.device)
                                        outputs = self.model(inputs)
                                    score = torch.softmax(outputs, 1)
                                    max_probs, preds = torch.max(score, 1)
                                    
                                    # 只选择置信度超过阈值的样本作为伪标签，并且确保不重复
                                    mask = (max_probs > args.confidence_threshold)
                                    filtered_inputs = inputs[mask]
                                    filtered_preds = preds[mask]
                                    
                                    for img, pred in zip(filtered_inputs, filtered_preds):
                                        img_hash = get_image_hash(img)
                                        if img_hash in pseudo_labeled_samples:
                                            continue
                                        
                                        history = sample_history.get(img_hash, [])
                                        history.append(pred.item())
                                        if len(history) >= label_stability_threshold and len(set(history)) == 1:
                                            pseudo_images.append(img)
                                            pseudo_labels.append(pred)
                                            # 更新已经加进来的伪标签样本集合
                                            pseudo_labeled_samples.add(img_hash)
                                            del sample_history[img_hash]
                                        
                                        # 确保只保留最近的历史记录
                                        sample_history[img_hash] = history[-label_stability_threshold:]                   

                            # 将新的伪标签数据与原始标记数据合并
                            if len(pseudo_labels) > 0:
                                pseudo_images =  torch.stack(pseudo_images) # if len(pseudo_images) > 1 else pseudo_images[0]
                                pseudo_labels =  torch.stack(pseudo_labels) # if len(pseudo_labels) > 1 else pseudo_labels[0]
                                pseudo_dataset = TensorDataset(pseudo_images.cpu(), pseudo_labels.cpu())                            
                                combined_dataset = CombinedDataset(labeled_loader.dataset, pseudo_dataset)
                                labeled_loader = DataLoader(combined_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
                                print(f"Added {len(pseudo_labels)} pseudo-labeled samples.")
                                del pseudo_images, pseudo_labels, pseudo_dataset, combined_dataset
                        '''
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                     
            if not epoch % save_epoch:
                self.save_model(model_path)
        # print(filename)
        plot_loss_and_accuracy(log_train_loss, log_val_loss, log_train_acc, log_val_acc, filename=filename)
        # 训完之后筛，然后train_diff
        true_labels = []
        with torch.no_grad():
            for batch in unlabeled_loader:
                if self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                    inputs, features, labels = batch[0], batch[1], batch[2]
                    features = features.to(self.device)
                else:
                    inputs, labels = batch[0], batch[1]
                
                outputs = []
                if self.model_type == 'mil':
                    inputs = swap_first_two_dims(inputs)
                    for bag in inputs:
                        bag_tensor = [instance.to(device) for instance in bag]
                        output = self.model(bag_tensor)
                        outputs.append(output)
                    outputs = torch.stack(outputs).to(device)
                elif self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs, features)
                else:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                score = torch.softmax(outputs, 1)
                max_probs, preds = torch.max(score, 1)
                
                # 只选择置信度超过阈值的样本作为伪标签
                mask = (max_probs > args.confidence_threshold)
                filtered_inputs = inputs[mask]
                filtered_preds = preds[mask]

                labels = labels.to(self.device)
                filtered_labels = labels[mask]
                if len(filtered_labels):
                    # 逐元素比较并统计相同元素数量
                    num_equal_elements = (filtered_preds == filtered_labels).sum().item()
                    print(f"Number of equal elements: {num_equal_elements}/{len(filtered_labels)}, acc:{num_equal_elements/len(filtered_labels)}")

                for img, pred, label in zip(filtered_inputs, filtered_preds, filtered_labels):
                    # 只保留少数类的伪样本
                    if True:# pred in labeled_loader.dataset.minority_classes:
                        pseudo_images.append(img)
                        pseudo_labels.append(pred)
                        true_labels.append(label)

                    
        # 将新的伪标签数据与原始标记数据合并
        if len(pseudo_labels) > 0:
            num_equal_elements = sum(x == y for x, y in zip(pseudo_labels, true_labels))
            print(f"Number of equal min only elements: {num_equal_elements}, acc:{num_equal_elements/len(true_labels)}")
            pseudo_images =  torch.stack(pseudo_images) # if len(pseudo_images) > 1 else pseudo_images[0]
            pseudo_labels =  torch.stack(pseudo_labels) # if len(pseudo_labels) > 1 else pseudo_labels[0]
            pseudo_dataset = TensorDataset(pseudo_images.cpu(), pseudo_labels.cpu())                            
            combined_dataset = CombinedDataset(labeled_loader.dataset, pseudo_dataset)
            self.filtered_loader = DataLoader(combined_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
            print(f"Added {len(pseudo_labels)} pseudo-labeled samples.")
            del pseudo_images, pseudo_labels, pseudo_dataset, combined_dataset
            self.train_diff(args, model_path)
        else:
            print('没有置信度足够高的样本')

    def train_one_mxp_epoch(self, args, dataloader, epoch):
        device = self.device
        
        dataset_size = 0.0
        running_loss = 0.0
        running_corrects = 0

        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for _, data in bar:
            
            images, targets = data[0][0], data[0][1]
            images, targets = images.to(device), targets.squeeze().to(device)
            balanced_images, balanced_targets = data[1][0], data[1][1]
            balanced_images, balanced_targets = balanced_images.to(device), balanced_targets.squeeze().to(device)
            targets = targets.long()
            balanced_targets = balanced_targets.long()
            
            n_classes = self.num_classes
            lam = np.random.beta(a=1.0, b=1)
            if images.size(0) != balanced_images.size(0):
                continue  # 跳过这个批次
            
            mixed_images = (1 - lam) * images + lam * balanced_images
            if n_classes == 2:
                mixed_targets = (1-lam)*targets + lam*balanced_targets
            elif n_classes >= 3:
                mixed_targets = (1-lam)*F.one_hot(targets, n_classes) + lam*F.one_hot(balanced_targets, n_classes)
            
            del images, balanced_images
            del targets, balanced_targets

            # 清零参数梯度
            self.optimizer.zero_grad()
            set_seed(args)

            # 前向传播
            outputs = []
            with torch.set_grad_enabled(True):
                with autocast():
                    if self.model_type == 'mil':
                        inputs = swap_first_two_dims(mixed_images)
                        if DEBUG:
                            print(len(inputs))
                            
                        for bag in inputs:
                            bag_tensor = [instance.to(device) for instance in bag]
                            output = self.model(bag_tensor)
                            outputs.append(output)
                        outputs = torch.stack(outputs).requires_grad_(True).to(device)
                    else:
                        inputs = inputs.to(self.device)
                        outputs = self.model(mixed_images)
                    # 视图组合方案2：合并相邻的相同标签和输出
                    # outputs, labels = self.merge_adjacent_labels(outputs, labels)
                    score = torch.softmax(outputs, 1)
                    _, preds = torch.max(score, 1)
                    loss = self.criterion(outputs.float(), mixed_targets.long())
                    # loss = self.criterion(outputs, onehot_labels)
                    # loss = self.criterion(outputs, labels) + self.criterion1(outputs, onehot_labels)
            
            batch_size = mixed_images.size(0)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
                       
            # 统计损失和准确率
            running_corrects += torch.sum(preds == mixed_targets.data)            
            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size
            bar.set_postfix(Epoch=epoch, 
                            Train_Loss=epoch_loss,
                            LR=self.optimizer.param_groups[0]['lr'])
        
        if self.scheduler is not None:
            self.scheduler.step()

        return epoch_loss, epoch_acc
    
    def train_mxp_model(self, args, model_path='model/mosub.pth', filename = 'logging.png'):
        num_epochs = args.num_epochs
        save_epoch = args.save_epoch

        log_train_loss = []
        log_val_loss = []
        log_train_acc = []
        log_val_acc = []
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            
            # 每个epoch都有训练和验证阶段
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # 设置模型为训练模式
                    dataloader = get_combo_loader(self.train_loader, args.base_sampling)
                    epoch_loss, epoch_acc = self.train_one_mxp_epoch(args, dataloader, epoch=epoch)
                else:
                    self.model.eval()   # 设置模型为评估模式
                    dataloader = self.val_loader
                    running_loss = 0.0
                    running_corrects = 0
                    # 迭代数据
                    for batch in dataloader:
                        if self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                            inputs, features, labels = batch[0], batch[1], batch[2]
                            features = features.to(self.device)
                        else:
                            inputs, labels = batch[0], batch[1]
                        
                        labels = labels.to(self.device)
                        onehot_labels = F.one_hot(labels, self.num_classes).to(torch.float32)
                        # 清零参数梯度
                        self.optimizer.zero_grad()
                        set_seed(args)
                        # 前向传播
                        outputs = []
                        with torch.set_grad_enabled(phase == 'train'):
                            with autocast():
                                if self.model_type == 'mil':
                                    inputs = swap_first_two_dims(inputs)
                                    if DEBUG:
                                        print(len(inputs))
                                        
                                    for bag in inputs:
                                        bag_tensor = [instance.to(device) for instance in bag]
                                        output = self.model(bag_tensor)
                                        outputs.append(output)
                                    outputs = torch.stack(outputs).requires_grad_(phase == 'train').to(device)
                                elif self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                                    inputs = inputs.to(self.device)
                                    outputs = self.model(inputs, features)
                                else:
                                    inputs = inputs.to(self.device)
                                    outputs = self.model(inputs)
                                # 视图组合方案2：合并相邻的相同标签和输出
                                # outputs, labels = self.merge_adjacent_labels(outputs, labels)
                                score = torch.softmax(outputs, 1)
                                _, preds = torch.max(score, 1)
                                loss = self.criterion(outputs, labels)
                                # loss = self.criterion(outputs, onehot_labels)
                                # loss = self.criterion(outputs, labels) + self.criterion1(outputs, onehot_labels)
                        # 统计损失和准确率
                        running_loss += loss.item() * len(inputs)# inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    epoch_loss = running_loss / len(dataloader.dataset)
                    epoch_acc = running_corrects.double() / len(dataloader.dataset)
                
                if phase == 'train':
                    log_train_loss.append(epoch_loss)
                    log_train_acc.append(epoch_acc.cpu())
                else:
                    log_val_loss.append(epoch_loss)
                    log_val_acc.append(epoch_acc.cpu())
                
                # 早停策略
                if phase == 'val':
                    if epoch_loss < self.best_val_loss:
                        self.best_val_loss = epoch_loss
                        self.early_stopping_counter = 0
                        self.save_model(model_path)
                    else:
                        self.early_stopping_counter += 1
                        if self.early_stopping_counter >= self.early_stopping_patience:
                            print("Early stopping")
                            self.save_model(model_path.rsplit('.')[0] + 'best_loss.pth')
                            plot_loss_and_accuracy(log_train_loss, log_val_loss, log_train_acc, log_val_acc, filename=filename)
                            return
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                     
            if not epoch % save_epoch:
                self.save_model(model_path)
        # print(filename)
        plot_loss_and_accuracy(log_train_loss, log_val_loss, log_train_acc, log_val_acc, filename=filename)

    def predict(self, data_loader):
        self.model.eval()
        images = []
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                if self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                    inputs, features, labels = batch[0], batch[1], batch[2]
                    features = features.to(self.device)
                else:
                    inputs, labels = batch[0], batch[1]
                outputs = []
                labels = labels.to(self.device)
                if self.model_type == 'mil':
                    inputs = swap_first_two_dims(inputs)
                    if DEBUG:
                        print(len(inputs))
                        
                    for bag in inputs:
                        bag_tensor = [instance.to(device) for instance in bag]
                        output = self.model(bag_tensor)
                        outputs.append(output)
                    outputs = torch.stack(outputs).to(device)
                elif self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs, features)
                else:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                score = torch.softmax(outputs, 1)
                _, preds = torch.max(score, 1)
                images.extend(inputs.squeeze().cpu().numpy())
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        return np.array(images), np.array(predictions), np.array(true_labels)

    def predict_img(self, image, feature = None, label = None):
        self.model.eval()
        
        with torch.no_grad():
            if feature:
                features = features.to(self.device)
            inputs = image
            if self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs, features)
            else:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
            score = torch.softmax(outputs, 1)
            pred_score, preds = torch.max(score, 1)

        return pred_score.item(), preds.item()

    def outsee_results(self, data_loader):
        images, predictions, labels = self.predict(data_loader)

        # 初始化字典来存储分类后的图像路径
        classified_images = {'TT': [], 'TF': [], 'FF': [], 'FT': []}

        # 分类图像
        for i, (true_label, pred_label) in enumerate(zip(labels, predictions)):
            if true_label == pred_label == 1:
                classified_images['TT'].append(images[i])
            elif true_label == 1 and pred_label == 0:
                classified_images['TF'].append(images[i])
            elif true_label == pred_label == 0:
                classified_images['FF'].append(images[i])
            elif true_label == 0 and pred_label == 1:
                classified_images['FT'].append(images[i])
        
        see_dict_images(classified_images)
        return classified_images

    def evaluate(self, data_loader):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        scores = []

        with torch.no_grad():
            for batch in data_loader:
                if self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                    inputs, features, labels = batch[0], batch[1], batch[2]
                    features = features.to(self.device)
                else:
                    inputs, labels = batch[0], batch[1]
                outputs = []
                labels = labels.to(self.device)
                onehot_labels = F.one_hot(labels, self.num_classes).to(torch.float32)
                if self.model_type == 'mil':
                    inputs = swap_first_two_dims(inputs)
                    if DEBUG:
                        print(len(inputs))
                        
                    for bag in inputs:
                        bag_tensor = [instance.to(device) for instance in bag]
                        output = self.model(bag_tensor)
                        outputs.append(output)
                    outputs = torch.stack(outputs).to(device)
                elif self.model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs, features)
                else:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                score = torch.softmax(outputs, 1)
                _, preds = torch.max(score, 1)

                if DEBUG:
                    print(outputs, labels)
                    print(outputs.shape, labels.shape)
                if self.loss_type in ['sf1', 'bce', 'mcc', 'wcl']:
                    loss = self.criterion(outputs, onehot_labels)
                elif self.loss_type == 'ce+sf1':
                    loss = self.criterion(outputs, labels) + self.criterion1(outputs, onehot_labels)
                else:
                    loss = self.criterion(outputs, labels) #  * 3.0, labels)
                if DEBUG:
                    print(torch.isnan(inputs).any(), torch.isinf(inputs).any())
                    print(torch.isnan(loss).any(), torch.isinf(loss).any())
                    print(f"Loss item: {loss.item()}, Inputs size: {len(inputs)}")
                    print(f"Running loss: {running_loss}")
                running_loss += loss.item() * len(inputs)# inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                scores.extend(score.cpu().numpy())

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = running_corrects.double() / len(data_loader.dataset)

        custom_y_pred1 = np.array(scores)[:,1].reshape(-1,1)
        Y_test = np.array(all_labels)
        Y_test1 = to_categorical(Y_test, self.num_classes)
        '''
        # Fit Platt scaling 
        lr = LogisticRegression(C=99999999999, solver='lbfgs')
        # fit the model 
        t=time.time()
        lr.fit(custom_y_pred1, Y_test)
        print('Training time: %s' % (time.time()-t))
        score_calibrated = lr.predict_proba(custom_y_pred1)
        print('Prediction time: %s' % (time.time()-t))
        '''
        '''
        # Fit three-parameter beta calibration
        bc = BetaCalibration()
        t=time.time()
        bc.fit(custom_y_pred1, Y_test)
        print('Training time: %s' % (time.time()-t))
        # perform beta calibration
        t=time.time()
        score_calibrated = bc.predict(custom_y_pred1)
        print('Prediction time: %s' % (time.time()-t))
        custom_y_pred1_beta_1 = np.array(1-score_calibrated)
        score_calibrated = np.c_[custom_y_pred1_beta_1, score_calibrated]
        '''
        # adjust Spline paramters for a better fit suiting your problem
        # Wrong Para
        # splinecalib = mli.SplineCalib(penalty='l2',
        #                             knot_sample_size=40,
        #                             cv_spline=5,
        #                             unity_prior=False,
        #                             unity_prior_weight=128)
        # t=time.time()
        # splinecalib.fit(custom_y_pred1, Y_test)
        # print('Training time: %s' % (time.time()-t))
        # score_calibrated = splinecalib.predict(custom_y_pred1)
        # print('Prediction time: %s' % (time.time()-t))
        # custom_y_pred1_spline_1 = np.array(1-score_calibrated)
        # score_calibrated = np.c_[custom_y_pred1_spline_1, score_calibrated]

        # preds_calibrated = score_calibrated.argmax(axis=-1)

        # return epoch_loss, epoch_acc, preds_calibrated, np.array(all_labels), score_calibrated
        return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels), np.array(scores)

    def compute_metrics(self, y_true, y_pred, y_scores, file_path='logging/roc.png'):
        # 计算准确率
        accuracy = accuracy_score(y_true, y_pred)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 绘图（混淆矩阵）
        # save_confusion_matrix_as_image(cm, class_names=['Luminal A', 'Luminal B', 'HER2(HR+)', 'HER2(HR-)', 'TN'], output_path=f'{file_path.split()[0]}_cm_densenet121-cbam_ms.png')
        # ['Non-Luminal', 'Luminal'] ['Non-TN', 'TN'] ['HER2', 'Non-HER2'] ['Luminal A', 'Luminal B', 'HER2(HR+)', 'HER2(HR-)', 'TN']
        
        # 计算灵敏性\特异性\PPV\NPV
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tn = cm[0, 0]

        sensitivity = tp / (tp + fn) if tp + fn > 0 else float('nan')
        specificity = tn / (tn + fp) if tn + fp > 0 else float('nan')
        ppv = tp / (tp + fp) if tp + fp > 0 else float('nan')
        npv = tn / (tn + fn) if tn + fn > 0 else float('nan')
        
        auc_scores = []

        for class_idx in range(self.num_classes):
            # 创建二分类标签
            binary_labels = (y_true == class_idx).astype(int)
            # 提取对应类别的预测概率
            binary_scores = y_scores[:, class_idx]
            # 计算 AUC

            try:
                auc = roc_auc_score(binary_labels, binary_scores)
                auc_scores.append(auc)
                fpr, tpr, _ = roc_curve(binary_labels, binary_scores)
                
                # 绘图（ROC）和日志
                '''
                draw_roc(fpr, tpr, auc, file_path=file_path)
                content = f"{self.model_type} {file_path} {class_idx} {fpr} {tpr} {auc}\n"
                with open('roc_data_dense-cbam_0424.txt', 'a', encoding='utf-8') as file:
                    file.write(content)
                '''
                
            except ValueError as e:
                print(f"Error for class {class_idx}: {e}")
                auc_scores.append(0.0)
                

        # 分类报告（精确率、召回率、F1分数等指标）
        class_report = classification_report(y_true, y_pred)
        # print(class_report)

        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv':ppv,
            'npv':npv,
            'auc': auc_scores,
            'avg_auc': np.mean(auc_scores)
        }, class_report
    
    def save_model(self, model_path='molsub.pth'):
        # 创建保存路径的目录（如果它不存在）
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 确保之前的文件不会干扰
        if os.path.exists(model_path):
            os.remove(model_path)

        try:
            torch.save(self.model, model_path)
        except:
            torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path='molsub.pth'):
        try:
            self.model = torch.load(model_path).to(self.device)
        except:
            self.model.load_state_dict(torch.load(model_path, map_location=device)).to(self.device)