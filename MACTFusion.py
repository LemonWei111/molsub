__author__ = "Lemon Wei"
__email__ = "Lemon2922436985@gmail.com"
__version__ = "1.1.0"

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

class DenseBlock1(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super(DenseBlock1, self).__init__()
        
        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))
        out3 = F.relu(self.conv3(out2))
        out4 = F.relu(self.conv4(out3))
        out = torch.cat([out1, out2, out3, out4], 1)
        return out

class DenseBlock2(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super(DenseBlock2, self).__init__()
        
        # block1
        self.block1_conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1)
        self.block1_conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, stride=1, padding=1)
        self.block1_conv3 = nn.Conv2d(in_channels + growth_rate * 2, growth_rate, kernel_size=1, stride=1, padding=0)
        
        # block2
        self.gradient_operator = GradientOperator(in_channels=in_channels)
        self.block2_conv3 = nn.Conv2d(in_channels, growth_rate, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # block1
        block1_out1 = F.relu(self.block1_conv1(x))
        block1_out2 = F.relu(self.block1_conv2(torch.cat([x, block1_out1], 1)))
        block1_out3 = F.relu(self.block1_conv3(torch.cat([x, block1_out1, block1_out2], 1)))  # 直连到Conv3之前
        
        # block2
        gradient_out = self.gradient_operator(x)
        block2_out = F.relu(self.block2_conv3(gradient_out))
        
        # 合并block1和block2的输出
        final_output = torch.cat([block1_out3, block2_out], 1)
        return final_output

class GradientOperator(nn.Module):
    def __init__(self, in_channels=1):
        super(GradientOperator, self).__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # 扩展Sobel滤波器以适应多个输入通道
        self.sobel_x = sobel_x.repeat(in_channels, 1, 1, 1).to(device)
        self.sobel_y = sobel_y.repeat(in_channels, 1, 1, 1).to(device)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # 应用Sobel滤波器到每个通道
        grad_x = F.conv2d(x, self.sobel_x, groups=channels, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, groups=channels, padding=1)
        
        # 计算每个像素点的梯度大小（L2范数）
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        # 如果需要，可以在所有通道上取均值或求和
        # 这里我们直接返回每个通道的梯度大小
        return gradient_magnitude

class ShallowFE(nn.Module):
    def __init__(self, in_channels=3, growth_rate=32):
        super(ShallowFE, self).__init__()
        self.denseblock1 = DenseBlock1(in_channels, growth_rate)
        self.denseblock2 = DenseBlock2(4 * growth_rate, growth_rate)

    def forward(self, x):
        dense1_out = self.denseblock1(x)
        final_output = self.denseblock2(dense1_out)
        return final_output

class SimAM(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

class SDConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SDConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        self.simam = SimAM()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.depthwise_conv(x))
        x = self.simam(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.cat((x, identity), dim=1)  # 输入直连到输出
        return x

class WindowAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7):
        super(WindowAttentionBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.LayerNorm(dim),
            WindowSelfAttention(dim, num_heads=num_heads, window_size=window_size)
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForwardNetwork(dim)
        )

    def forward(self, x):
        # block1: LN → Windows-SA，输入直连到输出
        residual = x
        x = self.block1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x += residual
        
        # block2: LN → FFN，输入直连到输出
        residual = x
        x = self.block2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x += residual
        
        return x

class GridAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, grid_size=7):
        super(GridAttentionBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.LayerNorm(dim),
            GridSelfAttention(dim, num_heads=num_heads, grid_size=grid_size)
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForwardNetwork(dim)
        )

    def forward(self, x):
        # block1: LN → Grid-SA，输入直连到输出
        residual = x
        x = self.block1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x += residual
        
        # block2: LN → FFN，输入直连到输出
        residual = x
        x = self.block2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x += residual
        
        return x

class WindowPartition(nn.Module):
    def __init__(self, window_size):
        super(WindowPartition, self).__init__()
        self.window_size = window_size

    def forward(self, x):
        B, H, W, C = x.shape  # 输入形状 [B, H, W, C]
        if DEBUG:
            print(f"Input shape: {x.shape}")  # 调试信息：输入形状
        
        # 将特征图划分为窗口
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        if DEBUG:
            print(f"After view: {x.shape}")  # 调试信息：view 后的形状
        
        # 重新排列维度并将窗口展平
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        if DEBUG:
            print(f"After permute and reshape (windows): {windows.shape}")  # 调试信息：划分后的窗口形状
        
        return windows, (H // self.window_size, W // self.window_size)

class WindowReverse(nn.Module):
    def __init__(self, window_size):
        super(WindowReverse, self).__init__()
        self.window_size = window_size

    def forward(self, windows, Wh, Ww):
        B_windows = windows.shape[0]
        C = windows.shape[-1]
        B = int(B_windows / (Wh * Ww))
        if DEBUG:
            print(f"Reversing windows with B={B}, Wh={Wh}, Ww={Ww}, C={C}")  # 调试信息：反向转换参数
        
        # 重组窗口为原始特征图
        x = windows.view(B, Wh, Ww, self.window_size, self.window_size, C)
        if DEBUG:
            print(f"After reshaping for reverse: {x.shape}")  # 调试信息：重组后的形状
        
        # 重新排列维度并展平
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Wh * self.window_size, Ww * self.window_size, C)
        if DEBUG:
            print(f"After permute and reshape (reversed): {x.shape}")  # 调试信息：最终形状
        
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape  # 获取批量大小、序列长度（窗口或网格内的元素数量）、通道数
        if DEBUG:
            print(f"MultiHeadSelfAttention input shape: {x.shape}")  # 调试信息：输入形状
        
        # 线性变换生成 qkv
        qkv = self.qkv(x)  # [B, N, 3 * C]
        if DEBUG:
            print(f"qkv shape after linear: {qkv.shape}")  # 调试信息：线性变换后的形状
        
        # 分割 qkv 并调整维度
        # [B, N, 3 * C] -> [3, B, num_heads, N, head_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        if DEBUG:
            print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")  # 调试信息：qkv 形状
        
        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if DEBUG:
            print(f"Attention matrix shape: {attn.shape}")  # 调试信息：注意力矩阵形状
        
        attn = attn.softmax(dim=-1)
        if DEBUG:
            print(f"Softmaxed attention matrix shape: {attn.shape}")  # 调试信息：softmax 后的注意力矩阵形状
        
        # 应用注意力权重
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if DEBUG:
            print(f"Output shape after applying attention: {x.shape}")  # 调试信息：应用注意力后的形状
        
        # 投影回原始维度
        x = self.proj(x)
        if DEBUG:
            print(f"Final output shape after projection: {x.shape}")  # 调试信息：最终输出形状
        
        return x

class WindowSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super(WindowSelfAttention, self).__init__()
        self.window_partition = WindowPartition(window_size)
        self.window_reverse = WindowReverse(window_size)
        self.attn = MultiHeadSelfAttention(dim, num_heads)

    def forward(self, x):
        if DEBUG:
            print(f"WindowSelfAttention input shape: {x.shape}")  # 调试信息：输入形状
        
        # 划分窗口
        windows, (Wh, Ww) = self.window_partition(x)  # 输入已经是 [B, H, W, C]
        if DEBUG:
            print(f"Window partition result shape: {windows.shape}")  # 调试信息：划分后的窗口形状
        
        # 应用多头自注意力机制
        windows = self.attn(windows)
        if DEBUG:
            print(f"Shape after MultiHeadSelfAttention: {windows.shape}")  # 调试信息：多头自注意力后的形状
        
        # 重组窗口
        x = self.window_reverse(windows, Wh, Ww)  # 输出保持 [B, H, W, C]
        if DEBUG:
            print(f"Final output shape after WindowSelfAttention: {x.shape}")  # 调试信息：最终输出形状
        
        return x

class GridPartition(nn.Module):
    def __init__(self, grid_size):
        super(GridPartition, self).__init__()
        self.grid_size = grid_size

    def forward(self, x):
        B, H, W, C = x.shape
        if DEBUG:
            print(f"GridPartition input shape: {x.shape}")  # 调试信息：输入形状
        
        # 将窗口划分为网格
        x = x.view(B, H // self.grid_size, self.grid_size, W // self.grid_size, self.grid_size, C)
        if DEBUG:
            print(f"After view: {x.shape}")  # 调试信息：view 后的形状
        
        grids = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.grid_size * self.grid_size, C)
        if DEBUG:
            print(f"After permute and reshape (grids): {grids.shape}")  # 调试信息：划分后的网格形状
        
        return grids, (H // self.grid_size, W // self.grid_size)

class GridReverse(nn.Module):
    def __init__(self, grid_size):
        super(GridReverse, self).__init__()
        self.grid_size = grid_size

    def forward(self, grids, Gh, Gw):
        B_grids = grids.shape[0]
        C = grids.shape[-1]
        B = int(B_grids / (Gh * Gw))
        if DEBUG:
            print(f"Reversing grids with B={B}, Gh={Gh}, Gw={Gw}, C={C}")  # 调试信息：反向转换参数
        
        # 重组网格为原始特征图
        x = grids.view(B, Gh, Gw, self.grid_size, self.grid_size, C)
        if DEBUG:
            print(f"After reshaping for reverse: {x.shape}")  # 调试信息：重组后的形状
        
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Gh * self.grid_size, Gw * self.grid_size, C)
        if DEBUG:
            print(f"After permute and reshape (reversed): {x.shape}")  # 调试信息：最终形状
        
        return x

class GridSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, grid_size):
        super(GridSelfAttention, self).__init__()
        self.grid_partition = GridPartition(grid_size)
        self.grid_reverse = GridReverse(grid_size)
        self.attn = MultiHeadSelfAttention(dim, num_heads)

    def forward(self, x):
        if DEBUG:
            print(f"GridSelfAttention input shape: {x.shape}")  # 调试信息：输入形状
        
        grids, (Gh, Gw) = self.grid_partition(x)  # 输入已经是 [B, H, W, C]
        if DEBUG:
            print(f"Grid partition result shape: {grids.shape}")  # 调试信息：划分后的网格形状
        
        grids = self.attn(grids)
        if DEBUG:
            print(f"Shape after MultiHeadSelfAttention: {grids.shape}")  # 调试信息：多头自注意力后的形状
        
        x = self.grid_reverse(grids, Gh, Gw)  # 输出保持 [B, H, W, C]
        if DEBUG:
            print(f"Final output shape after GridSelfAttention: {x.shape}")  # 调试信息：最终输出形状
        
        return x
        
class FeedForwardNetwork(nn.Module):
    def __init__(self, dim, hidden_dim=4*256):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class DeepFE(nn.Module):
    def __init__(self, in_channels=3, dim=256, num_heads=8, window_size=7, grid_size=7):
        super(DeepFE, self).__init__()
        self.sd_conv = SDConvBlock(in_channels, dim - in_channels)
        self.window_attention = WindowAttentionBlock(dim, num_heads, window_size)
        self.grid_attention = GridAttentionBlock(dim, num_heads, grid_size)

    def forward(self, x):
        # SD-conv 模块
        x = self.sd_conv(x)
        if DEBUG:
            print(x.shape)
        # Window_Attention 模块
        x = self.window_attention(x)
        
        # Grid_Attention 模块
        x = self.grid_attention(x)
        
        return x

class ShallowClassifier(nn.Module):
    def __init__(self, in_channels=3, input_size=224, growth_rate=32, num_classes=2):
        super(ShallowClassifier, self).__init__()
        # 使用nn.Sequential定义卷积层、激活函数和池化层
        self.features = ShallowFE(in_channels=in_channels, growth_rate=growth_rate)
        
        # 全连接层，用于分类
        self.classifier = nn.Sequential(
            nn.Linear(growth_rate * 2 * input_size * input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平特征图以输入到全连接层
        x = self.classifier(x)
        return x

class DeepClassifier(nn.Module):
    def __init__(self, in_channels=3, input_size=224, num_classes=2, dim=32, num_heads=4):
        super(DeepClassifier, self).__init__()
        # 使用nn.Sequential定义卷积层、激活函数和池化层
        self.features = DeepFE(in_channels=in_channels, dim=dim, num_heads=num_heads)
        
        # 全连接层，用于分类
        self.classifier = nn.Sequential(
            nn.Linear(dim * input_size * input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.contiguous().view(x.size(0), -1)  # 展平特征图以输入到全连接层
        x = self.classifier(x)
        return x
  
# 测试代码
if __name__ == "__main__":
    # 创建模型实例
    shallow_model = ShallowFE(in_channels=3).to(device)
    deep_model = DeepFE(in_channels=3, dim=256, num_heads=8, window_size=7, grid_size=7).to(device)

    # 创建一个示例输入张量 (batch_size, channels, height, width)
    input_tensor = torch.randn((1, 3, 224, 224)).to(device)

    # 获取输出
    shallow_output_tensor = shallow_model(input_tensor)
    deep_output_tensor = deep_model(input_tensor)

    print("shallow_model Output tensor shape:", shallow_output_tensor.shape) # torch.Size([1, 64, 224, 224]
    print("deep_model Output tensor shape:", deep_output_tensor.shape) # torch.Size([1, 256, 224, 224])