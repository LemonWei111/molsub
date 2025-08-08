# partly cited from https://github.com/cv-lee/Camelyon17
# partly cited from https://github.com/sivaramakrishnan-rajaraman/Model_calibration
__author__ = "Lemon Wei"
__email__ = "Lemon2922436985@gmail.com"
__version__ = "1.1.0"

import os
import sys
import math
import time
import torch
import random
import hashlib
import numpy as np
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit


from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
# 层输出可视化
from PIL import Image
from scipy.stats import sem, t

from torchvision.ops.boxes import box_area

def plot_loss_and_accuracy(log_train_loss, log_val_loss, log_train_acc, log_val_acc, filename='logging.png'):
    # 创建一个包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 绘制训练损失和验证损失
    ax1.plot(log_train_loss, label='Training Loss', marker='o')
    ax1.plot(log_val_loss, label='Validation Loss', marker='s')
    ax1.set_title('Training and Validation Loss Over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 绘制训练准确率和验证准确率
    ax2.plot(log_train_acc, label='Training Accuracy', marker='o')
    ax2.plot(log_val_acc, label='Validation Accuracy', marker='s')
    ax2.set_title('Training and Validation Accuracy Over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # 调整布局以防止重叠
    plt.tight_layout()

    # 保存图表为图片文件
    plt.savefig(filename)

# 提取指定层的输出
class FeatureExtractor:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.features = {}
        
        for layer_name in layers:
            layer = dict([*self.model.named_modules()])[layer_name]
            layer.register_forward_hook(self.save_output(layer_name))
    
    def save_output(self, layer_name):
        def hook(module, input, output):
            self.features[layer_name] = output
        return hook
    
    def __call__(self, x, feature=None):
        _ = self.model(x, feature) if feature is not None else self.model(x)
        return self.features
    
    def visualize_and_save(self, save_dir='layer_output'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for layer_name, layer_output in self.features.items():
            fig, axes = plt.subplots(4, 8, figsize=(16, 8))  # 假设每个层有32个通道
            for i, ax in enumerate(axes.flat):
                if i < layer_output.shape[1]:
                    ax.imshow(layer_output[0, i].detach().cpu().numpy(), cmap='viridis')
                    ax.axis('off')
                else:
                    fig.delaxes(ax)
            plt.suptitle(f'Features of {layer_name}')
            plt.tight_layout()
            plt.show()
            # 保存图像
            fig.savefig(os.path.join(save_dir, f'{layer_name}_features.png'))
            plt.close(fig)

    def visualize_attention(self, original_image, save_dir='layer_output'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
           
        for layer_name, layer_output in self.features.items():
            # print(layer_output) # tuple
            attention_weights = layer_output[0].detach().cpu().numpy()
            # TODO
            # print(attention_weights.shape) # 151296 = (768, 14, 14)

            # 特征图尺寸
            nh, feature_map_height, feature_map_width = 768, 14, 14
            attention_weights = attention_weights.reshape(nh, feature_map_height, feature_map_width)

            # 将注意力图上采样到原始图像尺寸
            patch_size = 16  # 假设每个特征点对应 16x16 的区域
            attention = nn.functional.interpolate(
                torch.tensor(attention_weights).unsqueeze(0),
                scale_factor=patch_size,
                mode="nearest"
            )[0].cpu().numpy()
            
            plt.figure(figsize=(10, 10))
            text = ["Original Image", "Head Mean"]
            for i, fig in enumerate([original_image, np.mean(attention, 0)]):
                plt.subplot(1, 2, i+1)
                plt.imshow(fig, cmap='inferno')
                plt.title(text[i])
                plt.show()
            plt.imsave(f'{layer_name}_mean_attention_map.png')

            plt.figure(figsize=(10, 10))
            for i in range(nh):
                plt.subplot(nh//3, 3, i+1)
                plt.imshow(attention[i], cmap='inferno')
                plt.title(f"Head n: {i+1}")
                plt.tight_layout()
                plt.show()
            plt.imsave(f'{layer_name}_attention_map.png')

def get_mid_outputs(input_tensor, model, layers_to_visualize, feature=None):
    # 创建特征提取器
    feature_extractor = FeatureExtractor(model, layers_to_visualize)
    # 获取指定层的输出
    features = feature_extractor(input_tensor, feature=feature)
    layer_outputs = []
    for layer_output in features.items():
        layer_outputs.append(layer_output)
    return layer_outputs

def visulize(image_path, model, layers_to_visualize, args, mask_path='', save_dir='layer_output', device='cpu'):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    # 定义数据增强管道（Resize的时候忽略了原始大小） resnet(224, 224) vit(256, 256)
    image_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # z 归一化  # none visual # 在ImageNet训练数据集上计算的均值、标准差（R,G,B）
        # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # min-max 归一化 # 归一化到 [-1, 1] 范围内
        ToTensorV2()
    ])
        
    clip_limit = args.clip_limit
    bound_size = args.bound_size
    input_channel = args.input_channel

    if mask_path:
        from data_process import get_roi
        original_image, mask = get_roi(image_path, mask_path, clip_limit, bound_size)
    else:
        image = Image.open(image_path) # .convert('RGB')
        if image.mode != 'L':
            image = image.convert('L')
        original_image = np.array(image)
        mask = None

    if args.feature:
        import joblib
        from classify import bag_of_words_representation_v3_1
        visual_vocabulary_path = "model/visual_vocabulary.pkl"  # vocablary保存路径
        visual_vocabulary = joblib.load(visual_vocabulary_path)
        feature = torch.tensor([bag_of_words_representation_v3_1(original_image, visual_vocabulary)]).to(device)
    else:
        feature = None

    if input_channel == 3:
        from data_process import process_rgb_image
        original_image = process_rgb_image(original_image, mask=mask)
    elif (input_channel == 1) and args.mask:
        from data_process import fill_border_with_background_mean
        # original_image = original_image * mask # 聚焦
        original_image = fill_border_with_background_mean(original_image, mask=mask)
                         
    input_tensor = image_transform(image=original_image)['image'].unsqueeze(0).to(device)

    # 创建特征提取器
    feature_extractor = FeatureExtractor(model, layers_to_visualize)
    # 获取指定层的输出
    features = feature_extractor(input_tensor, feature=feature)
    
    try:
        # 可视化并保存图像
        feature_extractor.visualize_and_save(save_dir=save_dir)
    except:
        feature_extractor.visualize_attention(original_image)
    
    return input_tensor

def normalize_image(image):
    """将图像数据归一化到 [0, 1] 范围内"""
    min_val = np.min(image)
    max_val = np.max(image)
    
    if max_val == min_val:
        # 如果最大值和最小值相同，将图像数据设为0
        normalized_image = np.zeros_like(image)
    else:
        normalized_image = (image - min_val) / (max_val - min_val)
    
    return normalized_image

def save_tensors_as_single_image(inputs, output_path='data/temp.jpg'):
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 将输入张量转换为NumPy数组
    try:
        images = [tensor.numpy() for tensor in inputs]
        # 创建一个新的图形
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        # 显示每个图像
        for i, image in enumerate(images):
            axes[i].imshow(image.transpose(1, 2, 0))
            axes[i].axis('off')
            axes[i].set_title(f'Image {i+1}')
    except:
        images = [tensor for tensor in inputs]
        if len(images) <= 1:
            plt.imshow(images[0][0], cmap='gray')
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # 保存图像到文件
            plt.close()  # 关闭图形
            return
        # 创建一个新的图形
        fig, axes = plt.subplots(1, len(images) + 1, figsize=(15, 5))
        # 显示每个图像
        for i, image in enumerate(images):
            axes[i].imshow(image[0], cmap='gray')
            # axes[i].imshow(image)
            axes[i].axis('off')
            axes[i].set_title(f'Image {i+1}')
    
    # 保存图像
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def see_dict_images(classified_images, file_name='data_process_examples/classified_images.png'):
    # 从每个分类中随机选择最多3张图像
    selected_images = {key: random.sample(value, min(3, len(value))) for key, value in classified_images.items()}

    # 创建一个4x3的网格来展示这些图片
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(9, 12))
    fig.subplots_adjust(hspace=0.5)

    # 定义类别标签的位置
    categories = list(selected_images.keys())
    for i, category in enumerate(categories):
        for j, img in enumerate(selected_images[category]):
            # 在指定位置添加图片
            ax = axes[i, j]
            try:
                ax.imshow(img, cmap='gray')
            except:
                ax.imshow(img[0], cmap='gray')
            ax.axis('off')  # 关闭坐标轴
            # 只在每个子图的第一列添加类别标签
            if j == 0:
                ax.set_title(category)

    plt.show()
    plt.savefig(file_name)

def get_last_conv_out_channels(model):
    last_conv_layer = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):  # 检查是否为Conv2d层
            last_conv_layer = module
    if last_conv_layer is not None:
        return last_conv_layer.out_channels
    else:
        raise ValueError("The model does not contain any Conv2d layers.")

def find_smallest_2d_array_size(arrays):
    # 定义一个变量来存储最小的大小，初始化为无穷大
    min_size = float('inf')
    
    # 遍历列表中的每一个二维数组
    for array in arrays:
        # 计算当前二维数组的大小
        current_size = len(array)
        # 更新最小大小
        if current_size < min_size:
            min_size = current_size
    
    # 返回最小的大小
    return min_size

def to_categorical(y, num_classes, dtype=np.float32):
    """
    将整数标签转换为 one-hot 编码的 NumPy 数组。
    
    参数:
    y (np.ndarray): 整数标签的 NumPy 数组。
    num_classes (int): 类别的总数。
    dtype (type, optional): 输出数组的数据类型，默认为 np.float32。
    
    返回:
    np.ndarray: one-hot 编码的 NumPy 数组。
    """
    # 创建一个全零的数组，形状为 (len(y), num_classes)
    one_hot_encoded = np.zeros((len(y), num_classes), dtype=dtype)
    
    # 使用索引将相应位置设置为 1
    one_hot_encoded[np.arange(len(y)), y] = 1
    
    return one_hot_encoded

def swap_first_two_dims(data):
    # 假设 data 是一个形状为 [11, 8, 64, 64] 的四维列表
    new_data = []
    for i in range(len(data[0])):  # 遍历第二个维度 (8)
        new_slice = []
        for j in range(len(data)):  # 遍历第一个维度 (11)
            new_slice.append(data[j][i])  # 交换第一和第二维度
        new_data.append(new_slice)
    return new_data

def create_weight_matrix(image_size, center_weight=1.0, border_weight=0.1):
    # 创建一个全为边界的权重矩阵
    weight_matrix = np.ones((image_size, image_size)) * border_weight
   
    # 计算中心点
    center_x, center_y = image_size // 2, image_size // 2
   
    # 计算每个像素到中心点的距离
    x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
   
    # 根据距离设置权重
    max_distance = np.max(distance)
    weight_matrix[distance < max_distance * 0.5] = center_weight  # 中心区域的权重
   
    return weight_matrix

def get_image_hash(image):
    """获取图像的哈希值"""
    return hashlib.sha256(image.cpu().numpy().tobytes()).hexdigest()

def calculate_confidence_interval(acc_scores):
    """
    计算并打印给定准确率列表的95%置信区间。
    
    参数:
    acc_scores (list of float): 交叉验证后的准确率列表
    
    返回:
    tuple: 包含置信区间的下限和上限
    """
    if not acc_scores or len(acc_scores) < 2:
        raise ValueError("准确率列表不能为空且至少需要两个元素来计算置信区间")
    
    # 计算均值、标准误差
    mean_acc = np.mean(acc_scores)
    se = sem(acc_scores)

    # 自由度
    n = len(acc_scores)
    df = n - 1

    # 置信水平
    confidence = 0.95

    # 计算95%置信区间
    ci_lower, ci_upper = t.interval(confidence, df, loc=mean_acc, scale=se)

    # 打印结果
    print(f"{mean_acc:.3f}({ci_lower:.3f}, {ci_upper:.3f})")

    return ci_lower, ci_upper

def process_excel(file_path):
    """
    从指定的.xlsx文件读取表格，按列处理2-6行的数据，
    调用给定的函数function_f，并将返回值存入第8、9行。

    参数:
    file_path (str): Excel文件路径。
    function_f (function): 接受一个列表作为参数并返回两个值的函数。
    """
    # 确定文件扩展名
    if file_path.lower().endswith('.xls'):
        engine_read = 'xlrd'
        engine_write = 'xlwt'
    elif file_path.lower().endswith('.xlsx'):
        engine_read = None  # 默认使用 openpyxl
        engine_write = None  # 默认使用 openpyxl
    else:
        raise ValueError("Unsupported file format")

    # 读取Excel文件
    df = pd.read_excel(file_path, engine=engine_read)

    # 遍历DataFrame从第二列开始的每一列
    for col in df.columns[1:]:
        try:
            # 提取该列2-6行的数值（注意索引是从0开始的）
            a = df.loc[0:4, col].values.tolist()
            
            # 确保提取的数据都是数字类型
            if not all(isinstance(item, (int, float)) for item in a):
                print(f"Warning: Non-numeric data detected in column {col}. Skipping this column.")
                continue
            
            # 调用函数f并获取返回值
            x, y = calculate_confidence_interval(a)
            
            # 将返回值x存入该列的第8行，y存入第9行
            df.at[6, col] = x  # 注意这里的索引是7，对应第8行
            df.at[7, col] = y  # 注意这里的索引是8，对应第9行
        except Exception as e:
            print(f"Error processing column {col}: {e}")
            continue
    
    # 保存修改后的表格回文件
    df.to_excel(file_path, index=False, engine=engine_write)

def draw_roc(fpr, tpr, roc_auc, file_path='logging/roc.png'):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 对角线代表随机猜测
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(file_path, bbox_inches='tight')

def draw_multiple_roc(fprs, tprs, roc_aucs, names, file_path='logging/roc.png'):
    """
    绘制多条ROC曲线。
    
    参数:
    - fprs: 包含不同模型FPR（假阳性率）列表的列表。
    - tprs: 包含不同模型TPR（真阳性率）列表的列表。
    - roc_aucs: 包含不同模型的AUC值的列表。
    - names: 包含不同模型名称的列表，用于图例显示。
    - file_path: 保存图像的文件路径。
    """
    plt.figure()
    
    # 使用不同的颜色循环绘图
    colors = ['red', 'darkorange', 'yellow', 'green', 'blue', 'purple']  # 可根据需要调整或扩展
    
    for i, (fpr, tpr, roc_auc) in enumerate(zip(fprs, tprs, roc_aucs)):
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f'{names[i]} ROC curve (area = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 对角线代表随机猜测
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(prop={'size': 6}, loc="lower right")
    plt.savefig(file_path, bbox_inches='tight')


def draw_rocs(y_true_list, y_scores_list, class_idx=1, file_path='roc.png'):
    plt.figure()
    for y_true, y_scores in zip(y_true_list, y_scores_list):
        # 创建二分类标签
        binary_labels = (y_true == class_idx).astype(int)
        # 提取对应类别的预测概率
        binary_scores = y_scores[:, class_idx]
        # 计算 AUC
        try:
            auc = roc_auc_score(binary_labels, binary_scores)
            fpr, tpr, _ = roc_curve(binary_labels, binary_scores)
            plt.plot(fpr, tpr, lw=2, label=f'Class {class_idx} (AUC = {auc:.2f})')
        except ValueError as e:
            print(f"Error for class {class_idx}: {e}")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 对角线代表随机猜测
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(file_path, bbox_inches='tight')


def parse_data_from_file(file_path):
    """
    从文件中读取并解析数据。
    
    参数:
    - file_path: 文件路径
    
    返回:
    - data: 包含每行解析结果的列表，每个元素是一个元组 (x, y, z, a_list, b_list, c)
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    buffer = ""
    in_brackets = False
    bracket_depth = 0

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue  # 跳过空行

        buffer += " " + stripped_line

        # 使用栈匹配方括号
        for char in stripped_line:
            if char == '[':
                bracket_depth += 1
                in_brackets = True
            elif char == ']':
                bracket_depth -= 1
                if bracket_depth == 0:
                    in_brackets = False

        if not in_brackets and bracket_depth == 0:
            # 解析当前缓冲区中的完整数据行
            parsed_data = parse_line(buffer.strip())
            data.append(parsed_data)
            buffer = ""

    return data

def parse_line(line):
    """
    解析一行数据，返回 x, y, z, a_list, b_list, c。
    
    参数:
    - line: 一行文本字符串
    
    返回:
    - x, y, z: 整数或浮点数
    - a_list: 列表 [a1, a2, ..., an]
    - b_list: 列表 [b1, b2, ..., bn]
    - c: 整数或浮点数
    """
    parts = []
    current_part = ""
    in_brackets = False
    bracket_depth = 0

    # 逐字符处理字符串
    for char in line:
        if char == '[':
            if bracket_depth == 0:
                in_brackets = True
            bracket_depth += 1
        elif char == ']':
            bracket_depth -= 1
            if bracket_depth == 0:
                in_brackets = False
        if char == ' ' and not in_brackets:
            if current_part:
                parts.append(current_part)
                current_part = ""
        else:
            current_part += char
    if current_part:
        parts.append(current_part)

    print(parts)
    # 解析 x, y, z
    idx = 0
    x = parts[idx]
    idx += 1
    y = parts[idx]
    idx += 1
    z = parts[idx]
    idx += 1

    # 解析第一个方括号内的列表 a_list
    a_part = parts[idx]
    a_list = [float(num) for num in a_part.strip('[]').split()]
    idx += 1
    
    b_part = parts[idx]
    b_list = [float(num) for num in b_part.strip('[]').split()]
    idx += 1  # 跳过右括号 ']'

    # 解析 c
    c = float(parts[idx])

    return x, y, z, a_list, b_list, c

def save_confusion_matrix_as_image(cm, class_names=None, output_path='confusion_matrix.png'):
    """
    将给定的混淆矩阵保存为图像文件。
    
    参数:
    - cm: 混淆矩阵(numpy数组形式)。
    - class_names: 类别名称列表，默认为None。如果提供，应与混淆矩阵的维度匹配。
    - output_path: 保存输出图片的路径，默认为'confusion_matrix.png'。
    """
    # 如果未提供类别名，则默认使用索引值
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
        
    # 将混淆矩阵转换成Pandas DataFrame
    df_cm = pd.DataFrame(cm, index=[f"Actual {name}" for name in class_names], 
                         columns=[f"Predicted {name}" for name in class_names])

    # 绘制热图
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')

    # 添加标题和坐标轴标签
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 旋转x轴标签以防止重叠
    plt.xticks(rotation=45, ha='right')  # 'ha'参数用于对齐方式
    plt.yticks(rotation=0)

    # 调整布局以避免标签被裁剪
    plt.tight_layout()

    # 保存图像到文件
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")

    # 关闭图像以释放内存
    plt.close()

# https://github.com/sivaramakrishnan-rajaraman/Model_calibration/blob/main/model_calibration.ipynb
def calc_bins(y_test, preds):
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(preds, bins)  
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
        bin_accs[bin] = (y_test[binned==bin]).sum() / bin_sizes[bin]
        bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(y_test, preds):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(y_test, preds)
    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)
    return ECE,MCE

def draw_reliability_graph(y_test, preds, file_name='LogisticRegression.png'):
    ECE, MCE = get_metrics(y_test, preds)
    bins, _, bin_accs, _, _ = calc_bins(y_test, preds)
    fig = plt.figure(figsize=(15, 10), dpi=400)
    ax = fig.gca()
    # x/y limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    # x/y labels
    plt.xlabel('Prediction', fontsize=20)
    plt.ylabel('Truth', fontsize=20)
    # Create grid
    ax.set_axisbelow(True) 
    ax.grid(color='gray', linestyle='dashed')
    # Error bars
    plt.bar(bins, bins,width=0.1,alpha=0.3,edgecolor='black', color='orange', hatch='\\')
    # Draw bars and identity line
    plt.bar(bins, bin_accs, width=0.1, alpha=1, 
          edgecolor='black', color='red') # b before 
    plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)
    # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')
    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='blue',  #green before
                             label='ECE = {:.2f}%'.format(ECE*100))
    MCE_patch = mpatches.Patch(color='green', 
                                label='MCE = {:.2f}%'.format(MCE*100))
    plt.legend(handles=[ECE_patch, MCE_patch],prop={'size': 15})
    plt.savefig(file_name)

def convert_bbox_format(bbox):
    """
    Convert bbox from [x, y, width, height] to [x_min, y_min, x_max, y_max].
    
    Parameters:
        bbox: list or tuple of four elements [x, y, width, height]
        
    Returns:
        converted_bbox: list of four elements [x_min, y_min, x_max, y_max]
    """
    x, y, width, height = bbox
    return [x, y, x + width, y + height]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # 如果使用多个GPU
    # 确保每次调用cuDNN时都具有确定的行为
    torch.backends.cudnn.deterministic = True

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def detect_accuracy(boxes_list, predict_boxes_list, labels_list, predict_labels_list, correct_per_class={}, scores_per_class_list=[], iou_threshold=0.5, device='cuda:0'):
    correct_predictions = 0
    total_nums = 0
    for boxes, predict_boxes, labels, predict_labels in zip(boxes_list, predict_boxes_list, labels_list, predict_labels_list):
        # print(boxes, predict_boxes)
        # print(labels, predict_labels)
        # input('check')
        if len(boxes) < 1:
            '''
            # 没有目标框
            if len(predict_boxes) < 1:
                correct_predictions += 1
                total_nums += 1
            '''
            continue
        total_nums += len(boxes)
        if len(predict_boxes) < 1:
            continue

        # print(predict_labels)
        # print(labels)
        # Ensure input in inputs are tensors
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes, dtype=torch.float32).to(device) # for example: tensor([[ 678.,  415., 1010.,  952.]], device='cuda:0')
        if not isinstance(predict_boxes, torch.Tensor):
            predict_boxes = torch.tensor(predict_boxes, dtype=torch.float32).to(device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels).to(device) # for example: tensor([0], device='cuda:0')
        if not isinstance(predict_labels, torch.Tensor):
            predict_labels = torch.tensor(predict_labels).to(device)
        # if not isinstance(scores, torch.Tensor):
        #     scores = torch.tensor(scores).to(device)
        # print(predict_labels)
        # print(labels)
        
        matched_gt = set()
        # print(predict_boxes)
        # for pbox in predict_boxes:
        #     print(pbox, pbox[2:] > pbox[:2])
        # print(boxes)
        # print(boxes[:, 2:])
        # print(boxes[:, :2])
        giou_matrix = generalized_box_iou(predict_boxes, boxes)  # Shape: [N, M]
        # print(len(predict_boxes), len(boxes))
        # print(giou_matrix.shape)
        # print(giou_matrix)
        # input('check shape')
        # if giou_matrix is None:
        #     continue
        for pred_idx in range(len(predict_boxes)):
            pred_label = predict_labels[pred_idx]
            giou = giou_matrix[pred_idx]
            if giou is not None and giou.shape[0] > 0:
                # print(giou.shape)
                max_result = giou.max(dim=0)
            else:
                # print(giou, 'dim 0 zero')
                max_result = (0, -1)
            max_giou, best_match_idx = max_result
            # print(max_giou)
            if (max_giou >= iou_threshold) and (best_match_idx.item() not in matched_gt):
                gt_label = labels[best_match_idx]
                # print(pred_label, gt_label)
                # input('detect')
                if pred_label == gt_label:
                    correct_predictions += 1

                    # TODO. Used to be always {1:1}
                    print(pred_label.item(), type(pred_label.item()))
                    if pred_label.item() not in correct_per_class.keys():
                        correct_per_class[pred_label.item()] = 0
                    correct_per_class[pred_label.item()] += 1

                    matched_gt.add(best_match_idx.item())
                    if len(matched_gt) >= len(boxes):
                        break 
        # print(len(boxes), boxes)
        # print(correct_predictions, total_nums)
        # print(correct_per_class)
        # input('check')
    accuracy = correct_predictions / total_nums if total_nums > 0 else 0.0
    return accuracy if accuracy <= 1 else 1, correct_predictions, total_nums, correct_per_class

def compute_metrics(true_nums, pred_nums):
    """
    计算ACC、SENS、SPEC、PPV、NPV。
    
    :param true_nums: 真实正例数量列表
    :param pred_nums: 预测为正例的数量列表
    :return: 包含各指标值的字典
    """
    # 初始化总计数器
    total_true = sum(true_nums)
    total_pred = sum(pred_nums)
    total_cases = 0
    correct_preds = 0
    
    SENS_sum = 0
    SPEC_sum = 0
    PPV_sum = 0
    NPV_sum = 0
    
    for true_num, pred_num in zip(true_nums, pred_nums):
        FP = abs(true_num - pred_num)  # 假设非正即负
        FN = max(0, true_num - pred_num)
        TP = min(true_num, pred_num)
        TN = total_cases - (TP + FP + FN)
        
        total_cases += true_num
        
        if TP + FN > 0:
            SENS_sum += TP / (TP + FN)
        if TN + FP > 0:
            SPEC_sum += TN / (TN + FP)
        if TP + FP > 0:
            PPV_sum += TP / (TP + FP)
        if TN + FN > 0:
            NPV_sum += TN / (TN + FN)
        
        correct_preds += min(true_num, pred_num)
    
    n = len(true_nums)
    ACC = correct_preds / total_cases if total_cases > 0 else 0
    SENS = SENS_sum / n if n > 0 else 0
    SPEC = SPEC_sum / n if n > 0 else 0
    PPV = PPV_sum / n if n > 0 else 0
    NPV = NPV_sum / n if n > 0 else 0
    
    return {
        "ACC": ACC,
        "SENS": SENS,
        "SPEC": SPEC,
        "PPV": PPV,
        "NPV": NPV,
        # AUC cannot be calculated with provided data.
    }

def poly_func(x, *params):
    """
    多项式函数，参数数量由多项式的阶数决定。
    """
    return sum([p * x**i for i, p in enumerate(params)])

def fit_and_plot(data_str, degree=2):
    rows = data_str.strip().split('\n')
    first_elements = [float(row.split(',')[0]) for row in rows]
    x_values = np.array(range(1, len(first_elements) + 1))

    # 猜测初值，这里我们用np.ones(degree + 1)作为初值
    initial_guess = np.ones(degree + 1)
    
    # 使用curve_fit拟合数据
    params, covariance = curve_fit(poly_func, x_values, first_elements, p0=initial_guess)

    # 创建用于绘制平滑曲线的x值
    x_smooth = np.linspace(min(x_values), max(x_values), 300)
    y_smooth = poly_func(x_smooth, *params)

    # 绘制原始数据点和拟合曲线
    plt.figure(figsize=(10, 5))
    plt.scatter(x_values, first_elements, label='Data Points')  # 原始数据点
    plt.plot(x_smooth, y_smooth, '-r', label=f'Fitted Curve (Degree {degree})')  # 拟合曲线
    plt.title('First Element of Each Row with Fitted Curve')
    plt.xlabel('Row Index')
    plt.ylabel('First Element Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('tune_result.jpg')    


# compute performance metrics

def matrix_metrix(real_values,pred_values,beta):
    CM = confusion_matrix(real_values,pred_values)
    TN = CM[0][0]
    FN = CM[1][0] 
    TP = CM[1][1]
    FP = CM[0][1]
    Population = TN+FN+TP+FP
    Kappa = 2 * (TP * TN - FN * FP) / (TP * FN + TP * FP + 2 * TP * TN + FN**2 + FN * TN + FP**2 + FP * TN)
    Prevalence = round( (TP+FP) / Population,2)
    Accuracy   = round( (TP+TN) / Population,4)
    Precision  = round( TP / (TP+FP),4 )
    NPV        = round( TN / (TN+FN),4 )
    FDR        = round( FP / (TP+FP),4 )
    FOR        = round( FN / (TN+FN),4 ) 
    check_Pos  = Precision + FDR
    check_Neg  = NPV + FOR
    Recall     = round( TP / (TP+FN),4 )
    FPR        = round( FP / (TN+FP),4 )
    FNR        = round( FN / (TP+FN),4 )
    TNR        = round( TN / (TN+FP),4 ) 
    check_Pos2 = Recall + FNR
    check_Neg2 = FPR + TNR
    LRPos      = round( Recall/FPR,4 ) 
    LRNeg      = round( FNR / TNR ,4 )
    DOR        = round( LRPos/LRNeg)
    F1         = round ( 2 * ((Precision*Recall)/(Precision+Recall)),4)
    FBeta      = round ( (1+beta**2)*((Precision*Recall)/((beta**2 * Precision)+ Recall)) ,4)
    MCC        = round ( ((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))  ,4)
    BM         = Recall+TNR-1
    MK         = Precision+NPV-1
    mat_met = pd.DataFrame({'Metric':['TP','TN','FP','FN','Prevalence','Accuracy','Precision','NPV','FDR','FOR','check_Pos','check_Neg','Recall','FPR','FNR','TNR','check_Pos2','check_Neg2','LR+','LR-','DOR','F1','FBeta','MCC','BM','MK','Kappa'],     'Value':[TP,TN,FP,FN,Prevalence,Accuracy,Precision,NPV,FDR,FOR,check_Pos,check_Neg,Recall,FPR,FNR,TNR,check_Pos2,check_Neg2,LRPos,LRNeg,DOR,F1,FBeta,MCC,BM,MK, Kappa]})
    return (mat_met)

def progress_bar(current, total, msg=None):
    ''' print current result of train, valid
    
    Args:
        current (int): current batch idx
        total (int): total number of batch idx
        msg(str): loss and acc
    '''

    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    ''' calculate and formating time 

    Args:
        seconds (float): time
    '''

    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def stats(outputs, targets):
    ''' Using outputs and targets list, calculate true positive,
        false positive, true negative, false negative, accuracy, 
        recall, specificity, precision, F1 Score, AUC, best Threshold.
        And return them

    Args:
        outputs (numpy array): net outputs list
        targets (numpy array): correct result list

    '''
    
    num = len(np.arange(0,1.005,0.005))

    correct = [0] * num
    tp = [0] * num
    tn = [0] * num
    fp = [0] * num
    fn = [0] * num
    recall = [0] * num
    specificity = [0] * num

    outputs_num = outputs.shape[0]
    for i, threshold in enumerate(np.arange(0, 1.005, 0.005)):
            
        threshold = np.ones(outputs_num) * (1-threshold)
        _outputs = outputs + threshold
        _outputs = np.floor(_outputs)

        tp[i] = (_outputs*targets).sum()
        tn[i] = np.where((_outputs+targets)==0, 1, 0).sum()
        fp[i] = np.floor(((_outputs-targets)*0.5 + 0.5)).sum()
        fn[i] = np.floor(((-_outputs+targets)*0.5 + 0.5)).sum()
        correct[i] += (tp[i] + tn[i])

    thres_cost = fp[0]+fn[0]
    thres_idx = 0

    for i in range(num):
        recall[i] = tp[i] / (tp[i]+fn[i])
        specificity[i] = tn[i] / (fp[i]+tn[i])
        if thres_cost > (fp[i]+fn[i]):
            thres_cost = fp[i]+fn[i]
            thres_idx = i

    correct = correct[thres_idx]
    tp = tp[thres_idx]
    tn = tn[thres_idx]
    fp = fp[thres_idx]
    fn = fn[thres_idx]
    recall = (tp+1e-7)/(tp+fn+1e-7)
    precision = (tp+1e-7)/(tp+fp+1e-7)
    specificity = (tn+1e-7)/(fp+tn+1e-7)
    f1_score = 2.*precision*recall/(precision+recall+1e-7)
    auc = roc_auc_score(targets, outputs) 
    threshold = thres_idx * 0.005

    return correct, tp, tn, fp, fn, recall, precision, specificity, f1_score,auc,threshold

def convert_three_to_one_channel(model, model_type='dense'):
    """
    将模型的第一个卷积层从三通道改为一通道，
    并使用原三通道权重的平均值初始化新的单通道卷积层。
    
    :param model: 预训练的模型
    :return: 修改后的模型
    """
    # 获取第一个卷积层的权重
    if 'dense' in model_type:
        first_conv_layer = model.features.conv0  # 对于DenseNet，输入层位于model.features.conv0 / MOB-CBAM: features[0][0]
    elif 'mob' in model_type:
        first_conv_layer = model.features[0][0]
    else:
        print('Unsupport.')
        return model
    original_weights = first_conv_layer.weight.data
    
    # 计算原三通道权重的平均值
    new_weights = torch.mean(original_weights, dim=1, keepdim=True)
    
    # 修改卷积层为单通道输入
    out_channels = first_conv_layer.out_channels
    kernel_size = first_conv_layer.kernel_size
    stride = first_conv_layer.stride
    padding = first_conv_layer.padding
    
    # 创建新的卷积层并初始化其权重
    new_first_conv_layer = torch.nn.Conv2d(1, out_channels, kernel_size,
                                           stride=stride, padding=padding, bias=False)
    new_first_conv_layer.weight.data = new_weights
    
    # 替换旧的第一卷积层
    if 'dense' in model_type:
        model.features.conv0 = new_first_conv_layer  # 对于DenseNet，输入层位于model.features.conv0 / MOB-CBAM: features[0][0]
    elif 'mob' in model_type:
        model.features[0][0] = new_first_conv_layer
    
    return model

if __name__ == '__main__':
    log_train_loss, log_val_loss, log_train_acc, log_val_acc = [1, 2], [2, 1], [1, 2, 3], [2, 3, 4]
    # plot_loss_and_accuracy(log_train_loss, log_val_loss, log_train_acc, log_val_acc)
    accs = [0.746875, 0.7453125, 0.746875, 0.7390625, 0.7453125]
    sensitivity = [1, 0.9874476987447699, 1, 0.9895397489539749, 0.997907949790795]
    specificity = [0, 0.030864197530864196, 0, 0, 0]
    ppv = [0.746875, 0.7503974562798092, 0.746875, 0.7448818897637796, 0.7464788732394366]
    npv = [0, 0.45454545454545453, 0, 0, 0]
    accs = [0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.25, 0.625, 0.3333333333333333]
    f1s = [0.23214285714285715, 0.31666666666666665, 0.5357142857142857, 0.16666666666666666, 0.325, 0.4666666666666667, 0.16666666666666666, 0.4892857142857143, 0.21428571428571427, 0.5833333333333334, 0.3333333333333333]
    sensitivity = [1.0, 0.0, 0.6666666666666666, 1.0, 0.5, 1.0, 0.6666666666666666, 0.6666666666666666, 1.0, 0.8, 0.5] 
    specificity = [1.0, 0.0, 0.6666666666666666, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0] 
    ppv = [0.0, 0.1826, 0.1778, 0.1558, 0.1852] 
    npv = [1.0, 0.0, 0.6666666666666666, 0, 0.5, 0, 0.0, 0.5, 0, 0.5, 0.0] 
    aucs = [0.55, 0.39571428571428574, 0.4384523809523809, 0.49333333333333335, 0.4166666666666667, 0.6261904761904762, 0.5347619047619048, 0.6380952380952382, 0.2826190476190476, 0.7447619047619047, 0.1]
    '''
    {'accuracy': 0.746875, 'sensitivity': 1.0, 'specificity': 0.0, 'ppv': 0.746875, 'npv': nan, 'auc': [0.550377085593264, 0.550377085593264], 'avg_auc': 0.550377085593264}
    {'accuracy': 0.7453125, 'sensitivity': 0.9874476987447699, 'specificity': 0.030864197530864196, 'ppv': 0.7503974562798092, 'npv': 0.45454545454545453, 'auc': [0.5739578490624515, 0.5739578490624515], 'avg_auc': 0.5739578490624515}
    {'accuracy': 0.746875, 'sensitivity': 1.0, 'specificity': 0.0, 'ppv': 0.746875, 'npv': nan, 'auc': [0.5508548995299344, 0.5508548995299344], 'avg_auc': 0.5508548995299344}
    {'accuracy': 0.7390625, 'sensitivity': 0.9895397489539749, 'specificity': 0.0, 'ppv': 0.7448818897637796, 'npv': 0.0, 'auc': [0.5587711142104448, 0.5587711142104447], 'avg_auc': 0.5587711142104448}
    {'accuracy': 0.7453125, 'sensitivity': 0.997907949790795, 'specificity': 0.0, 'ppv': 0.7464788732394366, 'npv': 0.0, 'auc': [0.5879435921276926, 0.5879435921276925], 'avg_auc': 0.5879435921276925}
    '''
    # 示例调用
    data = """
    0.54592,0.0001,0.9,0.005,10,50,1e-05,0.3,8,1.3,0.0
    0.4686,0.0001,0.9,0.0042,10,50,1e-05,0.36873,8,1.3,0.0
    0.52531,0.0001,0.70018,0.00507,10,52,1e-05,0.29672,9,1.31174,0.0
    0.49843,0.0001,0.9,0.0051,10,51,1e-05,0.30046,8,1.28993,0.0
    0.48607,0.0001,0.93627,0.00497,10,50,1e-05,0.30524,8,1.32784,0.0
    0.50922,0.00011,0.98,0.00605,10,50,1e-05,0.29021,8,1.2693,0.0
    0.50353,0.0001,0.90001,0.005,10,50,1e-05,0.3,8,1.30011,0.0
    0.53356,9e-05,0.96628,0.005,9,50,1e-05,0.28033,8,1.49261,0.0
    0.50726,6e-05,0.85442,0.00444,12,50,1e-05,0.29306,11,0.93859,0.0
    0.49431,0.0001,0.91222,0.00516,9,59,1e-05,0.30708,8,1.35866,0.0
    0.50432,0.0001,0.92473,0.005,10,51,1e-05,0.3,8,1.28296,0.0
    0.52708,9e-05,0.9,0.00422,10,57,1e-05,0.3,7,1.3,0.0
    0.51217,9e-05,0.98,0.00727,10,50,1e-05,0.24408,7,1.45805,0.0
    0.53297,0.00011,0.98,0.005,9,54,1e-05,0.32511,8,1.3378,0.0
    0.50098,0.0001,0.98,0.00488,10,50,1e-05,0.29561,8,1.44172,0.0
    0.53984,0.00011,0.86506,0.00559,10,51,1e-05,0.30542,7,1.24967,0.0
    0.51845,0.0001,0.9,0.005,9,50,1e-05,0.29945,8,1.32199,0.0
    0.51491,8e-05,0.91852,0.00559,9,55,1e-05,0.3,8,1.3,0.0
    0.49195,0.00011,0.84476,0.00468,10,50,1e-05,0.35549,8,1.55047,0.0
    0.51943,0.0001,0.96675,0.00535,10,50,1e-05,0.29526,8,1.34739,0.0
    0.47841,0.00011,0.70775,0.00485,9,50,1e-05,0.35216,6,1.52128,0.0
    0.51805,9e-05,0.98,0.00407,11,50,1e-05,0.3,8,1.01119,0.0
    0.52845,0.0001,0.98,0.00609,9,50,1e-05,0.3,7,1.42381,0.0
    0.49529,0.0001,0.9,0.00534,10,50,1e-05,0.30486,8,1.30735,0.0
    0.5053,0.0001,0.90292,0.00502,10,50,1e-05,0.30151,8,1.28887,0.0
    0.53846,0.00011,0.90689,0.00489,9,50,1e-05,0.3,9,1.20443,0.0
    0.54906,9e-05,0.82013,0.00486,10,50,1e-05,0.27707,10,1.1766,0.0
    0.54906,9e-05,0.8624,0.00452,10,50,1e-05,0.27707,10,1.1766,0.0
    0.49529,8e-05,0.81053,0.00402,10,50,1e-05,0.30642,13,1.30065,0.0
    0.54611,9e-05,0.82013,0.0048,11,54,1e-05,0.27603,10,1.17868,0.0
    0.49863,0.00011,0.81146,0.0042,9,50,1e-05,0.24341,10,1.49141,0.0
    0.55239,9e-05,0.84584,0.00492,9,50,1e-05,0.29769,10,1.23919,0.0
    0.5679,0.0001,0.77781,0.00631,9,50,1e-05,0.23966,10,1.11213,0.0
    0.4741,9e-05,0.7947,0.00708,9,50,1e-05,0.18794,10,1.2339,0.0
    0.53846,9e-05,0.88202,0.00626,9,50,1e-05,0.29487,10,0.88171,0.0
    0.56829,0.0001,0.78022,0.00625,9,50,1e-05,0.23946,10,1.11187,0.0
    0.54749,9e-05,0.7,0.00513,9,55,1e-05,0.22117,11,1.1643,0.0
    0.56613,0.00011,0.7,0.00586,10,50,1e-05,0.21782,11,1.0224,0.0
    0.5314,0.0001,0.73526,0.00625,9,56,1e-05,0.22076,12,1.15314,0.0
    0.54297,9e-05,0.7737,0.00618,8,50,1e-05,0.26653,10,1.13707,0.0
    0.52983,0.00011,0.7,0.00742,7,58,1e-05,0.26698,10,1.1513,0.0
    0.54042,8e-05,0.77453,0.0048,10,52,1e-05,0.23946,9,1.08362,0.0
    0.56809,0.0001,0.80759,0.00686,8,50,1e-05,0.25716,9,1.18254,0.0
    0.4737,0.00011,0.78915,0.00625,9,59,1e-05,0.17086,10,1.30612,0.0
    0.53866,0.00011,0.76261,0.00629,9,50,1e-05,0.27612,11,1.20432,0.0
    0.54297,9e-05,0.85051,0.00667,8,50,1e-05,0.24965,10,1.11187,0.0
    0.56201,8e-05,0.78022,0.00552,8,50,1e-05,0.24807,9,1.01455,0.0
    0.56044,0.0001,0.76315,0.00643,9,50,1e-05,0.24455,10,1.11187,0.0
    0.56829,0.0001,0.78022,0.00601,9,50,1e-05,0.23946,10,1.11187,0.0
    0.55632,0.0001,0.71378,0.00547,9,50,1e-05,0.23946,9,1.11187,0.0
    0.48921,0.00011,0.7,0.00669,9,50,1e-05,0.25285,12,1.13615,0.0
    0.52728,0.0001,0.7,0.00601,10,58,1e-05,0.19923,11,1.12209,0.0
    0.53434,8e-05,0.7,0.00537,9,62,1e-05,0.19555,10,1.11187,0.0
    0.57673,0.00011,0.79337,0.00618,9,50,1e-05,0.23532,10,1.13467,0.0
    0.54631,0.00012,0.7,0.00623,8,50,1e-05,0.23532,8,1.09567,0.0
    0.48881,0.0001,0.70221,0.00644,11,50,1e-05,0.1873,10,1.04851,0.0
    0.57889,0.00012,0.79337,0.00612,9,50,1e-05,0.22512,10,1.13467,0.0
    0.5679,0.00012,0.79683,0.00624,9,50,1e-05,0.23109,10,1.16479,0.0
    0.52316,9e-05,0.79337,0.00615,9,55,1e-05,0.24921,7,1.20649,0.0
    0.55377,0.00012,0.78228,0.00594,7,50,1e-05,0.21405,10,1.28218,0.0
    0.53434,0.00015,0.7896,0.00622,9,50,1e-05,0.24261,7,1.13467,0.0
    0.54847,0.00013,0.72086,0.00552,7,55,1e-05,0.21323,10,1.35694,0.0
    0.52786,0.00013,0.81068,0.00633,8,50,1e-05,0.20895,12,1.0828,0.0
    0.4894,0.00013,0.81511,0.00474,11,50,1e-05,0.20128,9,0.89267,0.0
    0.47194,9e-05,0.76166,0.00591,9,50,1e-05,0.22512,11,1.73125,0.0
    0.54356,0.00014,0.79337,0.00595,10,54,1e-05,0.23864,9,1.05678,0.0
    0.55612,0.00012,0.78561,0.00615,9,50,1e-05,0.22512,10,1.1398,0.0
    0.53787,0.00012,0.79337,0.00643,9,50,1e-05,0.21837,11,1.0181,0.0
    0.5522,8e-05,0.79337,0.0064,11,50,1e-05,0.22512,10,1.04274,0.0
    0.48803,8e-05,0.79757,0.00622,9,58,1e-05,0.22512,14,1.30694,0.0
    0.55632,0.00013,0.79337,0.00581,9,50,1e-05,0.25391,10,1.23182,0.0
    0.56378,0.00012,0.80864,0.00602,9,51,1e-05,0.22137,10,1.1483,0.0
    0.54101,0.00012,0.7,0.00612,10,50,1e-05,0.22298,11,1.13467,0.0
    0.58065,0.00011,0.7,0.00723,9,50,1e-05,0.22512,10,1.10273,0.0
    0.55082,0.00011,0.7,0.0067,9,50,1e-05,0.23854,9,1.00656,0.0
    0.54513,0.00011,0.7,0.00692,8,57,1e-05,0.22512,11,1.15474,0.0
    0.5575,0.00011,0.70875,0.00664,7,56,1e-05,0.22512,10,1.10273,0.0
    0.55024,0.00012,0.7,0.00817,8,53,1e-05,0.21469,11,1.29402,0.0
    0.54729,0.00011,0.7,0.0073,8,50,1e-05,0.24872,10,1.10273,0.0
    0.56417,0.00011,0.79803,0.0074,9,58,1e-05,0.22388,10,1.21668,0.0
    0.55553,0.00011,0.82094,0.0074,9,52,1e-05,0.20807,11,1.2077,0.0
    0.5675,0.00011,0.7,0.00752,8,50,1e-05,0.22512,10,1.10273,0.0
    0.54553,9e-05,0.7,0.00723,8,50,1e-05,0.25617,9,1.10273,0.0
    0.53434,9e-05,0.7,0.00743,9,50,1e-05,0.25531,11,0.89329,0.0
    0.55926,0.00011,0.75229,0.00743,9,51,1e-05,0.22534,10,1.13778,0.0
    0.51688,0.00012,0.72058,0.00697,9,50,1e-05,0.25029,11,1.15903,0.0
    0.59262,0.0001,0.7,0.00711,8,53,1e-05,0.2151,10,1.15893,0.0
    0.58536,0.0001,0.7,0.00708,8,53,1e-05,0.21916,10,1.15956,0.0
    0.51177,0.0001,0.7,0.00785,6,66,1e-05,0.16811,8,1.15893,0.0
    0.55514,0.0001,0.70872,0.0073,8,54,1e-05,0.20708,10,1.15893,0.0
    0.59243,0.0001,0.7,0.0071,8,53,1e-05,0.21607,10,1.15893,0.0
    0.55004,0.0001,0.7,0.00747,8,50,1e-05,0.21156,10,1.17142,0.0
    0.57457,0.00011,0.7,0.0082,10,50,1e-05,0.2318,10,1.00463,0.0
    0.54886,9e-05,0.7,0.00584,10,56,1e-05,0.27092,10,0.79268,0.0
    0.57104,0.0001,0.7111,0.00718,8,53,1e-05,0.21806,10,1.08664,0.0
    0.55436,0.00011,0.7,0.00676,8,58,1e-05,0.24014,10,1.42356,0.0
    0.55848,0.00011,0.7,0.00711,8,53,1e-05,0.24684,10,1.15893,0.0
    0.51531,0.0001,0.7,0.0074,8,53,1e-05,0.20154,10,1.08717,0.0
    0.5363,8e-05,0.7,0.00753,9,53,1e-05,0.2151,12,1.15205,0.0
    0.56927,9e-05,0.75583,0.00711,9,53,1e-05,0.21616,10,1.16924,0.0
    """

    # fit_and_plot(data)
    
    calculate_confidence_interval(ppv)
    input('plot ok')
    calculate_confidence_interval(accs)
    calculate_confidence_interval(f1s)
    calculate_confidence_interval(sensitivity)
    calculate_confidence_interval(specificity)
    calculate_confidence_interval(npv)
    calculate_confidence_interval(aucs)
    # 示例数据
    true_nums = [19, 12, 31, 11, 10] # val
    pred_nums = [7, 1, 26, 0, 3] # DEIM
    metrics = compute_metrics(true_nums, pred_nums)
    print(metrics)
    pred_nums = [1, 0, 17, 0, 0]# yolo


    metrics = compute_metrics(true_nums, pred_nums)
    print(metrics)
    # process_excel('temp.xlsx')
    true_nums = [102, 42, 135, 41, 28] # test
    pred_nums = [39, 1, 114, 0, 7] # DEIM
    metrics = compute_metrics(true_nums, pred_nums)
    print(metrics)
    input('ok')
    
    '''
    # 层输出可视化
    from torchvision import models

    # 加载预训练模型
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()  # 设置模型为评估模式

    # 指定需要可视化的层
    layers_to_visualize = ['layer1', 'layer2']

    # 获取图像
    image_path = 'data/images/0.jpg'  # 替换为你的图像路径
    visulize(image_path, model, layers_to_visualize)
    '''

    y_test, preds = np.array([1, 0, 0, 1, 0, 1, 0, 0]), np.array([0, 1, 1, 0, 0, 0, 0, 1])
    y_scores = np.array([0.2, 0.6, 0.55, 0.08, 0.08, 0.18, 0.27, 0.67])
    # draw_reliability_graph(y_test, preds)
    # cm = confusion_matrix(y_test, preds)

    # cm = np.array([[22, 11, 3, 4, 3], [33, 14, 4, 4, 6], [3, 4, 7, 4, 3], [2, 4, 2, 2, 0], [3, 5, 4, 3, 3]])  # 示例数据
    # save_confusion_matrix_as_image(cm, class_names=['Luminal A', 'Luminal B', 'HER2(HR+)', 'HER2(HR-)', 'TN'])
    
    file_path = 'roc_data.txt'  # 替换为你的文件路径
    parsed_data = parse_data_from_file(file_path)
    fprs, tprs, roc_aucs, names = [], [], [], []
    classes = ['Luminal A', 'Luminal B', 'HER2(HR+)', 'HER2(HR-)', 'TN']
    for i, (x, y, z, fpr, tpr, auc) in enumerate(parsed_data):
        if (y == 'logging/roc_ms_1.png' or y == 'logging/roc_test_ms_1.png') and x == 'densenet121-CBAM':
            print(f"Line {i + 1}:")
            print(f"x: {x}, y: {y}, z: {z}")
            print(f"fpr: {fpr}")
            print(f"tpr: {tpr}")
            print(f"auc: {auc}")
            fprs.append(fpr)
            tprs.append(tpr)
            roc_aucs.append(auc)
            names.append(classes[int(z)])

    draw_multiple_roc(fprs, tprs, roc_aucs, names, 'roc_ms.png')
    '''
    auc = roc_auc_score(y_test, y_scores)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    draw_roc(fpr, tpr, auc)

    y_scores = np.array([[0.8, 0.2], [0.4, 0.6], [0.45, 0.55], [0.92, 0.08], [0.92, 0.08], [0.82, 0.18], [0.73, 0.27], [0.33, 0.67]])
    y_scores1 = np.array([[0.83, 0.17], [0.42, 0.58], [0.41, 0.59], [0.91, 0.09], [0.22, 0.78], [0.89, 0.11], [0.43, 0.57], [0.43, 0.57]])
    draw_rocs([y_test, y_test], [y_scores, y_scores1], 1)
    '''