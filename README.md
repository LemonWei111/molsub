# Predicting Molecular Subtype in Breast Cancer Using Deep Learning on mammography images

## 📌 Overview

This code base implements the DenseNet121-CBAM model proposed in the paper, a framework based on deep learning, which is used to non-invasive predict the molecular subtypes of breast cancer from routine mammography images, aiming to reduce the dependence on tissue biopsy.

### Main functions:

Support dichotomous tasks of molecular subtypes in breast cancer（Luminal vs. non-Luminal、HER2+ vs. HER2-、TN vs. non-TN）
Support multi category classification task（Luminal A、Luminal B、HER2+/HR+、HER2+/HR−、TN）
Provide model training, validation, and testing processes
Support performance evaluation (AUC, accuracy, sensitivity, specificity, NPV, PPV and other indicators)
Integrate Grad CAM visualization function to generate attention heatmaps to enhance model interpretability
Include data preprocessing and image enhancement modules, adapted for mammography image input

### Key topics:

The model integrates CBAM attention mechanism and DenseNet121 backbone network to enhance the feature extraction ability of tumors and peripheral features
Modular design facilitates the replacement of backbone networks, attention modules, or classification heads
Provide clear configuration file interface, support hyperparameter adjustment and experimental reproduction
Include visualization tools to visually display the model's focus areas and assist in clinical interpretation

## 🚀 Quick Start

### 1. Create the environment using Anaconda (recommended)

```bash
# Clone project
git clone https://github.com/LemonWei111/molsub.git
cd molsub

# Create Conda environment (automatically install all dependencies)
conda env create -f environment.yml

# Activate Environment
conda activate molsub
```

Python版本：明确要求的Python版本（如 Python >= 3.10）。

### 2. Data Preparation

数据集：mammography subtype dataset(chaoyang huigu, chaoyang qianzhan)。
原始数据：基于患者隐私考虑，暂时保密。示例数据链接：[examples.zip](https://drive.google.com/drive/folders/1aVJjBz9f3nkS-HtQ3xevpfWhtnHUafi2?usp=sharing)（一个钼靶图像对应一个标注）
预处理后的数据：[data.zip](https://drive.google.com/drive/folders/1E_zJ66rPS6bFNrO_sTY7tFTXe6WZIEkn?usp=sharing)

```bash
data/
└── dataset_name/                              # 原始数据
    └── excel/
    └── subset_name/
        ├── patient_id {R/L}{CC/MLO}/
            ├── .nii.gz
            └── .dcm
└── processed/                              # 预处理后的数据
    └── .pkl
```

预处理后的excel示例(0: Luminal A, 1: Luminal B, 2: HER2\HR+, 3: HER2\HR-, 4: TN)：
| name | img_path | annotation_path | label |
| :---: | :---: | :---: | :---: |
| 1 | examples/img1.dcm | examples/anno1.nii.gz | 1 |
| 2 | examples/img2.dcm | examples/anno2.nii.gz | 4 |
| 3 | examples/img3.dcm | examples/anno3.nii.gz | 0 |

How to use your own dataset?

你应该创建一个表头如上的数据表格，命名以_processed结尾。
Then you can run as the followings:
```bash
chmod +x data_process.sh
./data_process.sh -l {ms/}HER2 -P data/mammography\ subtype\ dataset/beiyou\ excel/chaoyang\ retrospective_processed.xlsx -D examples
```
In this way, data can be preprocessed and saved under 'data/processed'.

DenseNet121的预训练模型在执行过程中自动下载。

训练好的DenseNet121-CBAM模型权重的下载链接：[model.zip](https://drive.google.com/drive/folders/1rYldK579H_BmYjJNUrBdBWUenpg89E_k?usp=sharing)
```bash
model/
└── prefered_model_for_ms.pth                       # 0: Luminal A, 1: Luminal B, 2: HER2\HR+, 3: HER2\HR-, 4: TN
└── prefered_model_for_l.pth                        # 0: Non-Luminal, 1: Luminal(include Luminal A and Luminal B)
└── prefered_model_for_tn.pth                       # 0: Non-TN, 1: TN
└── prefered_model_for_HER2.pth                     # 0: HER2(include HER2\HR+ and HER2\HR-), 1: Non-HER2
```
结合训练好的模型和示例数据，您可以实现快速推理和可视化。

### 3. Running Examples

Training & Evaluation:
```bash
chmod +x train.sh
./train.sh -l {ms/l/tn/HER2}
```
运行结束后，您可以查看日志文件和logging文件夹下训练测试损失、准确率随训练轮次的变化曲线(logging_{ms/l/tn/HER2}_{fold}.png)

Inference & Outsee:
```bash
chmod +x inference.sh
./inference.sh -l {ms/l/tn/HER2} -O model/prefered_model_for_{ms/l/tn/HER2}.pth -G examples/img1.dcm -A examples/anno1.nii.gz
```
运行结束后，您可以查看日志文件和layer_output/{ms/l/tn/HER2}文件夹下可视化的特征图({layer_name}_features.png)的注意力热图(temp_atten_{class_index}.jpg)

## 🛠️ Usage

### Configuration
训练脚本参数说明：

 -l, --label VALUE           分类任务类型：ms, l, tn, HER2 (默认: ms)
 -b, --bound-size N          肿瘤边界扩展大小 (与预处理的数据保持一致，默认: 100)
 -c, --clip-limit VALUE      CLAHE 对比度限制 (与预处理的数据保持一致，默认: 0.003)
 -i, --img-size N            输入图像尺寸 (与模型输入匹配，默认: 224)
 -C, --input-channel N       输入通道数（1=灰度图）(默认: 1)
 -O, --oversample-ratio V    过采样比例 (Luminal/Non-Luminal: 1.3, TN/Non-TN: 1.7, others: 1.5)
 -D, --downsample-ratio V    欠采样比例 (默认: 0.0)
 -M, --model-type NAME       模型类型 (默认: densenet121-cbam)
 -P, --pretrain 0|1          是否使用预训练权重 (1=是, 0=否) (默认: 1)
 -r, --dropout VALUE         Dropout 比例 (默认: 0.3)
 -L, --loss-type NAME        损失函数类型 (默认: ce)
 -R, --lr VALUE              初始学习率 (默认: 0.0001)
 -e, --decay VALUE           权重衰减 (默认: 0.005)
 -k N                        交叉验证折数 (默认: 5)
 -s, --batch-size N          Batch大小 (默认: 8)
 -w, --num-workers N         数据加载线程数 (默认: 2)
 -E, --num-epochs N          总epoch数 (默认: 300)
 -S, --save-epoch N          每N个epoch保存一次 (默认: 10)
 -N, --early-stopping-patience N 早停耐心值 (默认: 100)
 -I, --seed N                随机种子 (默认: 21)

### Evaluation Metrics
| Metric | Definition | Reason for selection |
| :---: | :---: | :---: |
| AUC | Area under the ROC curve | It is particularly useful for imbalanced datasets as it considers all possible classification thresholds. In the classification of molecular subtypes in breast cancer, the number of samples in different subtypes may be unbalanced (for example, there are relatively few TN breast cancer). AUC does not rely on a single classification threshold and can comprehensively evaluate the performance of the model under all thresholds. It is a robust indicator for measuring the discriminative ability of the model. |
| ACC(accuracy) | (TP + TN) / (TP + FP + FN + TN) | Measuring overall classification accuracy, suitable for datasets with balanced category. Providing a visual understanding of the overall performance of the model for horizontal comparison with other studies. Although bias may occur in imbalanced data, it remains a fundamental reference indicator. |
| SENS(sensitivity) | TP / (TP + FN) | This reflects the model's ability to correctly identify positive classes. In clinical diagnosis, it is related to "no missed diagnosis". |
| SPEC(specificity) | TN / (TN + FP) | Measuring the ability of the model to correctly identify negative classes. In clinical diagnosis, it is related to "no misdiagnosis". |
| NPV（negative predictive value）| TN / (TN + FN) | The proportion of actual negative classes predicted by the model, and high NPV means that negative predictions can effectively eliminate risks. |
| PPV（positive predictive value）| TP / (TP + FP) | The proportion of positive classes predicted by the model, and high PPV means that positive predictions are more reliable. |

Among them, the ROC curve is a curve constructed with true positive rate as the vertical axis and false positive rate as the horizontal axis. 
TP represents the number of samples that are actually positive and correctly classified. 
Similarly, TN represents the number of true negative cases, FP refers to the number of false positive cases, and FN refers to the number of false negative cases.

We use the scikit-learn library to efficiently calculate these metrics.

We use 5 fold cross validation, and all indicators are reported on an independent test set at each fold to ensure the objectivity and generalization ability of the evaluation results.

### Model Checkpoints
Save the model at 'model/mosub.pth' every {save_epoch} time
Save the model with the least loss for each fold at 'model/molsub_{model_type}_{label}_ {fold}.pth'

### Customization
How to modify the model architecture? How to add a new loss function or evaluation metric?

Follow 'model.py', where class 'MolSub' defined,
In the __init__ function, we have predefined over 20 model architectures and 9 loss functions for use,
In the compute_metrics function, we defined evaluation metrics.

一些模型预训练权重的下载链接：[checkpoint.zip](https://drive.google.com/drive/folders/1l6Bpg5YeDuI-DKfx1DClgpwKaN_N1aDX?usp=sharing)
```bash
checkpoint/
└── DenseNet121.pt                       # for model_type='rad_dense', link:
└── refers_checkpoint.pth                # for model_type='refers' link:
```

## 📂 Project Structure
```bash
molsub/
├── data/                                # 数据集 (需下载)
├── examples/                            # 原始数据示例(可下载)
├── model/                               # 保存的模型（可下载）
├── data_loader.py                       # 数据加载
├── data_process.py                      # 预处理
├── mob_cbam.py                          # CBAM模块加载函数
├── model.py                             # 模型类定义
├── test_auc_acc.py                      # DeLong's method，McNemar's method 统计学检验函数
├── train.py                             # 主函数
├── utils.py                             # 工具函数 (日志、评估等)
├── view_atten.py                        # 注意力图可视化函数
├── environment.yml                      # Conda环境包
├── requirements.txt                     # Python依赖
├── data_process.sh                      # 数据预处理脚本
├── train.sh                             # 训练和评估主脚本
├── inference.sh                         # 推理和可视化脚本
└── README.md                            # 本文件
```

## ❓ FAQ
您可能遇到的常见问题及解决方案。
Q: 运行时出现CUDA out of memory错误怎么办？ A: 尝试减小batch_size或num_workers。

## 🤝 我们期待您的贡献！

## 📜 License
明确说明代码库的开源许可证（如MIT, Apache 2.0, GPL等）。
示例： "本项目采用 MIT 许可证。详情见 LICENSE 文件。"

## 📬 Contact: Lemon2922436985@gmail.com

## 🎯 Thanks for the computing resource support provided by [Intelligent Perception and Computing Research Center of Beijing University of Posts and Telecommunications], and the data support provided by [Beijing Chaoyang Hospital, Capital Medical University]!
