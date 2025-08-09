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

### 2. Data Preparation

Dataset：mammography subtype dataset(chaoyang huigu, chaoyang qianzhan)  
Original Data：Due to the considerations of patients's privacy, it is temporarily confidential. Example data link：[examples.zip](https://drive.google.com/drive/folders/1aVJjBz9f3nkS-HtQ3xevpfWhtnHUafi2?usp=sharing)（One mammography image corresponds to one annotation）  
Preprocessed Data：[data.zip](https://drive.google.com/drive/folders/1E_zJ66rPS6bFNrO_sTY7tFTXe6WZIEkn?usp=sharing)  

```bash
data/
└── dataset_name/                              # Original Data
    └── excel/
    └── subset_name/
        ├── patient_id {R/L}{CC/MLO}/
            ├── .nii.gz
            └── .dcm
└── processed/                              # Preprocessed Data
    └── .pkl
```

Example of preprocessed excel(0: Luminal A, 1: Luminal B, 2: HER2\HR+, 3: HER2\HR-, 4: TN)：
| name | img_path | annotation_path | label |
| :---: | :---: | :---: | :---: |
| 1 | examples/img1.dcm | examples/anno1.nii.gz | 1 |
| 2 | examples/img2.dcm | examples/anno2.nii.gz | 4 |
| 3 | examples/img3.dcm | examples/anno3.nii.gz | 0 |

> How to use your own dataset?

You should create a table with the header as above, and name it with '_processed' at the end.  
Then you can run as the followings:  
```bash
chmod +x data_process.sh
./data_process.sh -l {ms/}HER2 -P data/mammography\ subtype\ dataset/beiyou\ excel/chaoyang\ retrospective_processed.xlsx -D examples
```
In this way, data can be preprocessed and saved under 'data/processed'.

The pre-trained model of DenseNet121 will be automatically downloaded during execution.

Download link for the trained weights of the DenseNet121-CBAM model：[model.zip](https://drive.google.com/drive/folders/1rYldK579H_BmYjJNUrBdBWUenpg89E_k?usp=sharing)
```bash
model/
└── prefered_model_for_ms.pth                       # 0: Luminal A, 1: Luminal B, 2: HER2\HR+, 3: HER2\HR-, 4: TN
└── prefered_model_for_l.pth                        # 0: Non-Luminal, 1: Luminal(include Luminal A and Luminal B)
└── prefered_model_for_tn.pth                       # 0: Non-TN, 1: TN
└── prefered_model_for_HER2.pth                     # 0: HER2(include HER2\HR+ and HER2\HR-), 1: Non-HER2
```
By combining the trained model with data for example, you can achieve rapid inference and visualization.

### 3. Running Examples

Training & Evaluation:
```bash
chmod +x train.sh
./train.sh -l {ms/l/tn/HER2}
```
After that, you can view the log file and the curve showing the changes in training and testing loss and accuracy with training epochs in the folder 'logging', and the curves are named as 'logging_{ms/l/tn/HER2}_{fold}.png'

Inference & Outsee:
```bash
chmod +x inference.sh
./inference.sh -l {ms/l/tn/HER2} -O model/prefered_model_for_{ms/l/tn/HER2}.pth -G examples/img1.dcm -A examples/anno1.nii.gz
```
After that, you can view the log file, the visualized feature maps ({layer_name}_features.png) the attention heatmaps (temp_atten_{class_index}.jpg) in the folder 'layer_output/{ms/l/tn/HER2}'.

## 🛠️ Usage

### Configuration
Parameter description of training script (overview):

| Command | Parameter | DataType | Description |
| :---: | :---: | :---: | :---: |
| -l | label | VALUE | Classification task types: ms, l, tn, HER2 (default: ms) |
| -b | bound_size | N | Tumor boundary expansion size (consistent with preprocessed data, default: 100) |
| -c | clip_limit | VALUE | CLAHE contrast limit (consistent with preprocessed data, default: 0.003) |
| -i | img_size | N | Input image size (matches model input, default: 224) |
| -C | input_channel | N | Number of input channels (1=grayscale image) (default: 1) |
| -O | oversample_ratio | V | Oversampling ratio (Luminal/Non-Luminal: 1.3, TN/Non-TN: 1.7, others: 1.5) |
| -D | downsample_ratio | V | Undersampling ratio (default: 0.0) |
| -M | model_type | NAME | Model type (default: densenet121 cbam) |
| -P | pretrain | 0/1 | Whether to use pre trained weights (1=Yes, 0=No) (default: 1) |
| -r | dropout | VALUE | Dropout ratio (default: 0.3) |
| -L | loss_type | NAME | Loss function type (default: ce) |
| -R | lr | VALUE | Initial learning rate (default: 0.0001) |
| -e | decay | VALUE | Weight decay (default: 0.005) |
| -k | k | N | Cross validation folds (default: 5) |
| -s | batch_size | N | Batch size (default: 8) |
| -w | num_workers | N | Number of data loading threads (default: 2) |
| -E | num_epochs | N | Total number of epochs (default: 300) |
| -S | save_epoch | N | Save every N epochs (default: 10) |
| -N | early_stopping_patience | N | Early stopping patience (default: 100) |
| -I | seed | N | Random seed (default: 21) |

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
Save the model at 'model/mosub.pth' every {save_epoch} time.  
Save the model with the least loss for each fold at 'model/molsub_{model_type}_{label}_ {fold}.pth'.

### Customization
How to modify the model architecture? How to add a new loss function or evaluation metric?

Follow 'model.py', where class 'MolSub' defined,  
In the __init__ function, we have predefined over 20 model architectures and 9 loss functions for use,  
In the compute_metrics function, we defined evaluation metrics.

Download links for pretrained weights of some models：[checkpoint.zip](https://drive.google.com/drive/folders/1l6Bpg5YeDuI-DKfx1DClgpwKaN_N1aDX?usp=sharing)
```bash
checkpoint/
└── DenseNet121.pt                       # for model_type='rad_dense', link:
└── refers_checkpoint.pth                # for model_type='refers' link:
```

## 📂 Project Structure
```bash
molsub/
├── data/                                # Dataset (requires download)
├── examples/                            # Raw Data Example (downloadable)
├── model/                               # Saved model (downloadable)
├── data_loader.py                       # Data load
├── data_process.py                      # Data process
├── mob_cbam.py                          # Loading functions of CBAM module
├── model.py                             # Model definition
├── test_auc_acc.py                      # Statistical test functions (DeLong's method，McNemar's method)
├── train.py                             # main
├── utils.py                             # Tool functions (logs, evaluations, etc.)
├── view_atten.py                        # Visualization of attention heatmap
├── environment.yml
├── requirements.txt
├── data_process.sh                      # Data preprocessing script
├── train.sh                             # Training and evaluation script
├── inference.sh                         # Inferencing and visualization scripts
└── README.md                            # This document
```

## ❓ FAQ
Common problems you may encounter and solutions.

Q: What should I do if there is a 'CUDA out of memory' error during runtime?  
A: Try reducing batch_size or num_workers.

## 🤝 We are looking forward to your contribution!

## 📜 This project adopts the MIT license. Please refer to the LICENSE document for details.

## 📬 Contact: Lemon2922436985@gmail.com

## 🎯 Thanks for the computing resource support provided by Intelligent Perception and Computing Research Center of [School of Artificial Intelligence, Beijing University of Posts and Telecommunications](https://ai.bupt.edu.cn/en/), and the data support provided by [Beijing Chaoyang Hospital, Capital Medical University](https://www.bjcyh.com.cn/Html/News/Articles/21569.html)!
