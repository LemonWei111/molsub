__author__ = "Lemon Wei"
__email__ = "Lemon2922436985@gmail.com"
__version__ = "1.1.0"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_target_layer_by_name(model, layer_name):
    """
    根据层名获取对应的层对象。
    
    参数:
    - model: 预训练模型
    - layer_name: 目标层的名称（字符串）
    
    返回:
    - target_layer: 对应的层对象
    """
    def find_layer(module, name_parts):
        if len(name_parts) == 1:
            return getattr(module, name_parts[0])
        else:
            return find_layer(getattr(module, name_parts[0]), name_parts[1:])
    
    # 将层名按 '.' 分割成列表
    name_parts = layer_name.split('.')
    return find_layer(model, name_parts)

def np_to_pil(np_array):
    """
    将NumPy数组转换为PIL图像对象。
    
    参数:
    - np_array: 归一化到[0, 1]范围的NumPy数组
    
    返回:
    - pil_image: PIL图像对象
    """
    # 如果数组是浮点数类型且在[0, 1]范围内，先乘以255并转换为uint8类型
    if np_array.dtype == np.float32 or np_array.dtype == np.float64:
        np_array = (np_array * 255).astype(np.uint8)
    
    # 确保数组的维度正确
    if len(np_array.shape) == 2:
        # 单通道灰度图像
        pil_image = Image.fromarray(np_array, mode='L')
    elif len(np_array.shape) == 3 and np_array.shape[2] == 3:
        # 三通道RGB图像
        pil_image = Image.fromarray(np_array, mode='RGB')
    else:
        raise ValueError("Unsupported array shape or dimension")
    
    return pil_image

def generate_gradcam_visualization(image_path, model, target_layer_names, args, target_class_index, mask_path='', pred=-1, pred_score=-1, save_dir='layer_output', device='cpu'):
    """
    生成并显示Grad-CAM可视化结果。
    
    参数:
    - image_path: 测试图像的路径
    - target_class_index: 目标类别的索引（例如：0表示负类，1表示正类）
    - model_name: 使用的预训练模型名称，默认为'resnet50'
    - target_layer: 目标卷积层，默认为模型的最后一层卷积层
    - use_cuda: 是否使用CUDA（GPU），默认为True
    """
    
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
    
    # 设置模型为评估模式
    model.to(device)
    model.eval()
    img = np_to_pil(original_image).convert('RGB') if mask_path else Image.open(image_path).convert('RGB')
    
    # 指定目标类别
    targets = [ClassifierOutputTarget(target_class_index)]

    # 计算需要的子图数量
    num_layers = len(target_layer_names)
    cols = min(num_layers + 1, 4)  # 每行最多放4个图，包括原始图像
    rows = (num_layers + 1) // 4 + 1 # 计算所需的行数
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    
    # 显示原始图像
    if rows == 1:
        axes[0].imshow(np.array(img.resize((224, 224))) / 255.0)
        axes[0].set_title(f'Original Image, Lable: {target_class_index}, {pred_score} Pred: {pred}')
        axes[0].axis('off')
    else:
        axes[0, 0].imshow(np.array(img.resize((224, 224))) / 255.0)
        axes[0, 0].set_title(f'Original Image, Lable: {target_class_index}, {pred_score} Pred: {pred}')
        axes[0, 0].axis('off')
    
    # 生成并显示每个层的热图
    for idx, target_layer_name in enumerate(target_layer_names):
        target_layer = get_target_layer_by_name(model, target_layer_name)
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        img_np = np.array(img.resize((224, 224))) / 255.0  # 将PIL图像转换为numpy数组并归一化到[0, 1]
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        row_idx = (idx + 1) // cols
        col_idx = (idx + 1) % cols
        
        if rows == 1:
            axes[col_idx].imshow(visualization)
            axes[col_idx].set_title(f'Attention Map ({target_layer_names[idx]})')
            axes[col_idx].axis('off')
        else:
            # print(row_idx, col_idx, rows, cols)
            axes[row_idx, col_idx].imshow(visualization)
            axes[row_idx, col_idx].set_title(f'Attention Map ({target_layer_names[idx]})')
            axes[row_idx, col_idx].axis('off')
    
    # 如果有空余的子图，隐藏它们
    for i in range(num_layers + 1, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        if rows == 1:
            axes[col_idx].axis('off')
        else:
            axes[row_idx, col_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/temp_atten_{target_class_index}.jpg')

