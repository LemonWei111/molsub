__author__ = "Lemon Wei"
__email__ = "Lemon2922436985@gmail.com"
__version__ = "1.1.0"

import torch
import random
import joblib
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from albumentations.pytorch import ToTensorV2
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# from collections import Counter


from utils import save_tensors_as_single_image
from classify import bag_of_words_representation_v3_1
from data_process import fill_border_with_background_mean, process_rgb_image, unsharp_mask

visual_vocabulary_path = "model/visual_vocabulary.pkl"  # vocablary保存路径


def get_resampled_indices(dataset, oversample_ratio=1.0, downsample_ratio=1.0):
    minority_class = dataset.minority_class
    class_counts = dataset.class_counts
    max_class = np.argmax(class_counts)
    oversampled_indices = np.where(dataset.labels == minority_class)[0]
    downsampled_indices = np.where(dataset.labels == max_class)[0]

    resampled_indices = []

    print('resampled_indices')

    return random.shuffle(resampled_indices)

class ResampledRandomSampler(Sampler):
    def __init__(self, resampled_indices):
        super().__init__()
        self.resampled_indices = resampled_indices

    def __iter__(self):
        return iter(self.resampled_indices)

    def __len__(self):
        return len(self.resampled_indices)
    
def create_binary_labels(labels, neg=None, pos=None):
    if (neg is not None) and (pos is None):
        labels_binary = [int(label not in neg) for label in labels]
    elif pos:
        for id, label in enumerate(labels):
            if label in neg:
                labels[id] = 0
            elif label in pos:
                labels[id] = 1
            else:
                labels[id] = 2
        return labels
    else:
        return labels
    return labels_binary


class MolSubDataset(Dataset):
    def __init__(self, data, labels, feature=0, input_channel=1, mask=None, patch_size=None, num_patches=10, neg=None, pos=None, transform=None, augment=False, oversample_ratio=0.0, downsample_ratio=0.0, gan_data=None):
        self.data = data
        self.labels = create_binary_labels(labels, neg, pos)
        
        if pos:
            self.data = [d for d, l in zip(self.data, self.labels) if l in [0, 1]]
            self.labels = [l for l in self.labels if l in [0, 1]]

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.feature = feature
        # from data_process import save_mask
        # save_mask(self.labels, '1212labels.pkl')
        # input('train_data_saved')
        
        self.transform = transform
        self.augment = augment
        if self.augment:
            # 进行随机的放射变换
            self.augmentation = A.Compose([
                A.HorizontalFlip(p=0.5), # 左右随机反射
                A.VerticalFlip(p=0.5),    # 上下随机反射
                A.Rotate(limit=20, p=0.5),  # ±20°之间的旋转 # none visual
                A.Affine(shear=(-0.1, 0.1), p=0.5)  # ±10°之间的水平剪切
            ])

        self.oversample_ratio = oversample_ratio
        self.downsample_ratio = downsample_ratio
        
        # 统计每个类别的样本数，并找到样本数最少的类别（即少数类）
        self.class_counts = np.bincount(self.labels)
        if len(self.class_counts) <= 2:
            self.minority_classes = [np.argmin(self.class_counts)]
        else:
            average_count = np.mean(self.class_counts[self.class_counts > 0])  # 忽略不存在的类（计数为0）
            self.minority_classes = np.where(self.class_counts < average_count)[0].tolist()
        print(f'原始数据分布：{self.class_counts}')

        ## 预处理放在这可能增加内存负担
        if input_channel == 3:
            # Ostu/mask、双边滤波增强三通道
            if mask is not None:
                self.data = [process_rgb_image(d, m) for d, m in zip(self.data, mask)]
            else:
                self.data = [process_rgb_image(d) for d in self.data]
        elif (input_channel == 1) and (mask is not None):
            # mask聚焦单通道
            # DEBUG
            # print(mask[:3])
            # input()
            # save_tensors_as_single_image(self.data[:3], 'data_process_examples/img.png')
            # save_tensors_as_single_image(mask[:3], 'data_process_examples/mask.png')
            # self.data = [d * m for d, m in zip(self.data, mask)] # 聚焦
            self.data = [fill_border_with_background_mean(d, mask=m) for d, m in zip(self.data, mask)] # 边界填充
        # else:
        #     self.data = [unsharp_mask(d) for d in self.data] # 单独USM
        # print(self.data[0])
        # input()
        # save_tensors_as_single_image(self.data[:3], 'data_process_examples/img*mask.png')
        
        # 直接扩展输入图像的通道数
        # self.data = [np.stack([d, d, d], axis=-1) for d in self.data]

        if self.transform is not None:
            self.data = [self.transform(image=image)['image'] for image in self.data]
        # TO Check: normalize_image !!
        # save_tensors_as_single_image(self.data[:3])
        
        # 检查图像大小是否为 (224, 224)
        # invalid_indices = []
        # for i, img in enumerate(self.data):
        #     print(img.shape)
        #     if img.shape[1:] != (224, 224):
        #         invalid_indices.append(i)
        # print(invalid_indices)

        steps = []
        if self.oversample_ratio > 0 and self.oversample_ratio < 1:
            over = SMOTE(sampling_strategy=self.oversample_ratio, random_state=21)
            steps.append(('o', over))
        if self.downsample_ratio != 0.0:
            under = RandomUnderSampler(sampling_strategy=self.downsample_ratio, random_state=21)
            steps.append(('u', under))
        
        if steps:
            try:
                # 将图像展平为一维向量(单通道)
                X_flattened = np.stack([img.view(-1).numpy() for img in self.data])
                # 创建 Pipeline
                pipeline = Pipeline(steps=steps)
                # 应用 Pipeline 进行采样
                X_resampled, y_resampled = pipeline.fit_resample(X_flattened, self.labels)
                
                self.data = [img.reshape(1, 224, 224) for img in X_resampled]
                # print(len(self.data))
                self.labels = y_resampled
            except:
                print(f'error in {steps}, over {self.oversample_ratio}, down {self.downsample_ratio}')

        if self.feature:
            visual_vocabulary = joblib.load(visual_vocabulary_path)
            self.features = [bag_of_words_representation_v3_1(image.permute(1, 2, 0).numpy(), visual_vocabulary) for image in self.data]

        # 如果需要过采样，计算过采样的次数
        if self.oversample_ratio > 1.0:
            # 简单随机
            # print(self.class_counts)
            # input()
            # self.oversample_times = int((self.oversample_ratio - 1.0) * min(self.class_counts))
            self.oversample_times = int((self.oversample_ratio - 1.0) * len(self.data))
            # print(self.class_counts[0] + self.oversample_times, self.class_counts[1])
            # input()
        else:
            self.oversample_times = 0

        self.oversampled_indices = np.where(np.isin(self.labels, self.minority_classes))[0]
        # print(self.labels)
        # SMOTE wrong (要求输入的数据是一个二维数组，其中每一行是一个样本，每一列是一个特征。即 self.data 中的元素形状要一致)
        # smote = SMOTE(sampling_strategy='auto', random_state=42)
        # self.data, self.labels = smote.fit_resample(self.data, self.labels)
        self.gan_data = gan_data

    def __len__(self):
        # return len(self.data) # SMOTE
        return len(self.data) + self.oversample_times # 简单随机
    
    def __getitem__(self, idx):
        if idx >= len(self.data):
            if self.gan_data:
                idx = np.random.randint(0, len(self.gan_data))
                image = self.gan_data[idx]
                image = self.transform(image=image)['image']
                label = self.minority_class
            else:
                # Simple random oversampling: For the oversampled portion, randomly select a minority class sample
                idx = np.random.choice(self.oversampled_indices) # 简单随机过采样：对于过采样的部分，随机选择一个少数类样本
                image = self.data[idx]
                label = self.labels[idx]
        else:
            image = self.data[idx]
            label = self.labels[idx]

        # if self.transform is not None:
        #     image = self.transform(image=image)['image']
        
        # mil patch
        if self.patch_size is not None:
            # Extract patches from the image
            patches = []
            image = image.squeeze()
            # print(image.shape)
            width, height = image.shape[0], image.shape[1]
            for i in range(self.num_patches):
                left = torch.randint(0, width - self.patch_size[0], (1,)).item()
                top = torch.randint(0, height - self.patch_size[1], (1,)).item()
                patch = image[left:left + self.patch_size[0], top:top + self.patch_size[1]]
                if self.augment:
                    patch = self.augmentation(image=np.array(patch))['image']
                if self.transform:
                    patch = ToTensorV2()(image=patch)['image']
                patches.append(patch)
            
            # Add the scaled-down whole image as one of the instances in the bag
            scaled_image = A.Resize(self.patch_size[0], self.patch_size[1])(image=np.array(image))['image']
            if self.augment:
                scaled_image = self.augmentation(image=np.array(scaled_image))['image']
            if self.transform:
                scaled_image = ToTensorV2()(image=scaled_image)['image']
            patches.append(scaled_image)
            return patches, label
        
        if self.augment:
            image = self.augmentation(image=np.array(image))['image']

        if self.feature:
            feature = self.features[idx]
            return np.array(image), np.array(feature), np.array(label)

        return np.array(image), np.array(label)

class MolSubDataLoader():
    def __init__(self, train_data, train_labels, val_data, val_labels, feature=0, input_channel=1, patch_size=None, neg=None, pos=None, train_mask=None, val_mask=None, test_data=None, test_labels = None, test_mask=None, batch_size=64, num_workers=4, oversample_ratio=1.0, downsample_ratio=1.0, sampler=None, gan_data=None, img_size=224):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.oversample_ratio = oversample_ratio
        self.downsample_ratio = downsample_ratio
        
        # 定义数据增强管道（Resize的时候忽略了原始大小） resnet(224, 224) vit(256, 256)
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # z 归一化  # none visual # 在ImageNet训练数据集上计算的均值、标准差（R,G,B）
            # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # min-max 归一化 # 归一化到 [-1, 1] 范围内
            ToTensorV2()
        ])
        
        # 创建数据集
        self.train_dataset = MolSubDataset(train_data, train_labels, feature=feature, input_channel=input_channel, mask=train_mask, patch_size=patch_size, neg=neg, pos=pos, transform=self.transform, augment=True, oversample_ratio=self.oversample_ratio, downsample_ratio=self.downsample_ratio, gan_data=gan_data) if train_data and train_labels else None
        self.val_dataset = MolSubDataset(val_data, val_labels, feature=feature, input_channel=input_channel, mask=val_mask, patch_size=patch_size, neg=neg, pos=pos, transform=self.transform, augment=False) if val_data and val_labels else None
        self.test_dataset = MolSubDataset(test_data, test_labels, feature=feature, input_channel=input_channel, mask=test_mask, patch_size=patch_size, neg=neg, pos=pos, transform=self.transform, augment=False) if test_data and test_labels else None

        # MIL ROIBoundingBoxPatchDataset 1230
        # self.train_dataset = ROIBoundingBoxPatchDataset(train_data, train_labels, feature=feature, neg=neg, pos=pos, transform=self.transform, augment=True, oversample_ratio=self.oversample_ratio) if train_data and train_labels else None
        # self.val_dataset = ROIBoundingBoxPatchDataset(val_data, val_labels, feature=feature, neg=neg, pos=pos, transform=self.transform, augment=False) if val_data and val_labels else None
        # self.test_dataset = ROIBoundingBoxPatchDataset(test_data, test_labels, feature=feature, neg=neg, pos=pos, transform=self.transform, augment=False) if test_data and test_labels else None

        # 创建数据加载器
        if sampler:
            # 计算每个类别的初始权重
            class_counts = self.train_dataset.class_counts
            total_samples = sum(class_counts.values())
            initial_weights = {cls: total_samples / count for cls, count in class_counts.items()}
            # 归一化权重，确保总和为1
            sum_weights = sum(initial_weights.values())
            initial_weights = {cls: weight / sum_weights for cls, weight in initial_weights.items()}
            # 根据类别标签获取每个样本的权重
            sample_weights = [initial_weights[label.item()] for label in train_labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(self.train_dataset),
                replacement=True
            )
            # 采样率更新暂未实现
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, sampler=sampler) if self.train_dataset else None
        else:
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, sampler=sampler) if self.train_dataset else None
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers) if self.val_dataset else None
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers) if self.test_dataset else None
            
    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader
    
# 创建一个包装类来组合两个数据集
class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.lengths = [len(dataset1), len(dataset2)]
        self.total_length = sum(self.lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        if index < self.lengths[0]:
            return self.dataset1[index][0], self.dataset1[index][1]
        else:
            return np.array(self.dataset2[index - self.lengths[0]][0]), np.array(self.dataset2[index - self.lengths[0]][1])

class ROIBoundingBoxPatchDataset(Dataset):
    def __init__(self, data, labels, num_patches=10, max_black_ratio=0.4, feature=0, neg=None, transform=None, augment=False, oversample_ratio=0.0):
        """
        初始化数据集
        
        :param image: 原始图像，可以是numpy数组或能够索引获取图像切片的对象。
        :param roi_boxes: 包含所有ROI边界的列表，每个元素应为 (top, bottom, left, right) 形式的元组。
        :param num_patches: 除了原始ROI外要提取的额外patch数量，默认9个。
        """
        self.data = data
        self.num_patches = num_patches
        self.max_black_ratio = max_black_ratio
        self.labels = create_binary_labels(labels, neg)
        self.feature = feature
        
        self.transform = transform
        self.augment = augment
        if self.augment:
            self.augmentation = A.Compose([
                A.HorizontalFlip(p=0.5), # 左右随机反射
                A.VerticalFlip(p=0.5),    # 上下随机反射
                A.Rotate(limit=20, p=0.5),  # ±20°之间的旋转 # none visual
                A.Affine(shear=(-0.1, 0.1), p=0.5)  # ±10°之间的水平剪切
            ])

        self.oversample_ratio = oversample_ratio
        
        self.class_counts = np.bincount(self.labels)
        self.minority_class = np.argmin(self.class_counts)
        print(f'原始数据分布：{self.class_counts}')

        if self.oversample_ratio > 1.0:
            self.oversample_times = int((self.oversample_ratio - 1.0) * len(self.data))
        else:
            self.oversample_times = 0

        self.oversampled_indices = np.where(self.labels == self.minority_class)[0]

    def __len__(self):
        return len(self.data) + self.oversample_times # 简单随机

    def __getitem__(self, idx):
        if idx >= len(self.data):
            idx = np.random.choice(self.oversampled_indices) # 简单随机过采样：对于过采样的部分，随机选择一个少数类样本
            image, roi_boxes = self.data[idx][0], self.data[idx][1]
            label = self.labels[idx]
        else:
            image, roi_boxes = self.data[idx][0], self.data[idx][1]
            label = self.labels[idx]
        
        # 获取ROI边界
        top, bottom, left, right = roi_boxes[0], roi_boxes[1], roi_boxes[2], roi_boxes[3]
        patch_size = bottom - top

        # 创建一个列表来保存所有的patches
        patches = []
        
        # 提取原ROI对应的patch
        original_patch = image[top:bottom, left:right].copy()
        if self.transform is not None:
            original_patch = self.transform(image=original_patch)['image']
        original_patch = original_patch.squeeze()
        if self.augment:
            original_patch = self.augmentation(image=np.array(original_patch))['image']
        if self.transform:
            original_patch = ToTensorV2()(image=original_patch)['image']# .contiguous().permute(1, 0, 2)
        patches.append(original_patch)
        
        min_top = max(top - 100, 0)
        max_top = min(top + 100, image.shape[0] - patch_size)
        min_left = max(left - 100, 0)
        max_left = min(left + 100, image.shape[1] - patch_size)
        for _ in range(self.num_patches):
            valid_patch_found = False
            while not valid_patch_found:
                new_top = torch.randint(min_top, max_top, (1,)).item() if min_top != max_top else min_top
                new_left = torch.randint(min_left, max_left, (1,)).item() if min_left != max_left else min_left
                patch = image[new_top:new_top + patch_size, new_left:new_left + patch_size].copy()
                # 检查黑色像素的比例
                if self.check_patch_validity(patch, patch_size):
                    if self.transform is not None:
                        patch = self.transform(image=patch)['image']
                    patch = patch.squeeze()
                    if self.augment:
                        patch = self.augmentation(image=np.array(patch))['image']
                    if self.transform:
                        tensor_patch = ToTensorV2()(image=patch)['image']# .contiguous().permute(1, 0, 2)
                    patches.append(tensor_patch)
                    valid_patch_found = True
        
        return patches, label

    def check_patch_validity(self, patch, patch_size):
        """
        检查给定的patch是否满足条件（即黑色像素不超过max_black_ratio）
        
        :param patch: 要检查的patch
        :return: 如果patch有效则返回True，否则返回False
        """
        total_pixels = patch_size ** 2
        black_pixels = np.sum(patch == 0)  # Assuming black is represented by 0
        
        return black_pixels / total_pixels <= self.max_black_ratio
                
def get_filtered_loader(args, filtered_inputs, filtered_labels):
    # 假设 filtered_inputs 和 filtered_labels 已经准备好了
    # 如果它们是列表，请确保使用 torch.cat 或类似方法将它们转换为张量
    filtered_inputs = torch.cat(filtered_inputs).cpu()
    filtered_labels = torch.cat(filtered_labels).cpu()
    '''
    # 已经增强过的样本基本不会重复
    # 将 filtered_inputs 转换为哈希类型以便于计数 (例如，将张量转换为元组)
    input_hashes = [tuple(input_.cpu().numpy().flatten()) for input_ in filtered_inputs]

    # 统计每个样本的出现次数
    counter = Counter(input_hashes)

    # 计算平均出现次数
    average_occurrences = sum(counter.values()) / len(counter)

    # 筛选并调整样本
    new_filtered_inputs = []
    new_filtered_labels = []

    for idx, hash_ in enumerate(input_hashes):
        occurrences = counter[hash_]
        
        # 根据出现次数添加样本
        copies_to_add = 2 if occurrences > average_occurrences else 1
        
        for _ in range(copies_to_add):
            new_filtered_inputs.append(filtered_inputs[idx])
            new_filtered_labels.append(filtered_labels[idx])

    # 将结果转换回张量
    new_filtered_inputs = torch.stack(new_filtered_inputs)
    new_filtered_labels = torch.tensor(new_filtered_labels)
    
    print(f"Original number of samples: {len(filtered_inputs)}")
    print(f"New number of samples after processing: {len(new_filtered_inputs)}")
    '''
    # 定义新的DataLoader
    filtered_dataset = MolSubDataset(filtered_inputs, filtered_labels, mask=None)
    print(filtered_dataset.class_counts) 
    filtered_loader = DataLoader(filtered_dataset, batch_size=args.batch_size, shuffle=True)
    return filtered_loader