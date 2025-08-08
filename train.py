__author__ = "Lemon Wei"
__email__ = "Lemon2922436985@gmail.com"
__version__ = "1.1.0"

import math
import torch
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from data_process import load_data, load_mask, process_gan_data
from data_loader import MolSubDataLoader
from model import MolSub, set_seed
from utils import visulize, calculate_confidence_interval
# from tuner import Tuner
# from mgto import mgto_optimization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tuple_type(strings):
    try:
        # 将输入字符串解析为元组
        # 例如："(32,32)" 或 "32,32"
        return tuple(map(int, strings.strip('()').split(',')))
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Tuple must be in the format (int,int): {strings}")

def combine_classification(model1_output, model2_output):
    """
    根据两个模型的输出决定最终分类结果。

    参数:
        model1_output (torch.Tensor): 模型1的输出张量。
        model2_output (torch.Tensor): 模型2的输出张量。

    返回:
        torch.Tensor: 最终分类结果张量。如果任意一个模型输出为0，则对应位置的结果为0；否则为1。
    """
    # print(type(model1_output))
    final_output = torch.tensor([(o1.cpu().numpy() + o2.cpu().numpy())/2 for o1, o2 in zip(model1_output, model2_output)], device=device)
    # print(final_output[0], type(final_output))
    return final_output

def combine_evaluate(data_loader, model1, model2):
    model1.eval()
    model2.eval()
    running_corrects = 0
    all_preds = []
    all_labels = []
    scores = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            outputs = combine_classification(outputs1, outputs2)
            
            score = torch.softmax(outputs, 1)
            _, preds = torch.max(score, 1)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            scores.extend(score.cpu().numpy())

    epoch_acc = running_corrects.double() / len(data_loader.dataset)
    return epoch_acc, np.array(all_preds), np.array(all_labels), np.array(scores)

def test(test_loader, model, args, model2=None, num_classes=5, device=device, file_path='logging/roc.png'):
    # 验证模型
    if type(model) == str:
        # load
        model1_path = model
        model = MolSub(None, None, args, num_classes=num_classes, device=device)
        model.load_model(model1_path)
        print(f'load {type(model)} model from {model1_path}')
    if type(model2) == str:
        model2_path = model2
        model2 = MolSub(None, None, args, num_classes=num_classes, device=device)
        model2.load_model(model2_path)
        print(f'load {type(model2)} model from {model2_path}')
    if model2:
        test_loss = None
        try:
            test_acc, test_preds, test_labels, test_scores = combine_evaluate(test_loader, model.model, model2.model)
        except:
            test_acc, test_preds, test_labels, test_scores = combine_evaluate(test_loader, model.model, model2)
    else:
        test_loss, test_acc, test_preds, test_labels, test_scores = model.evaluate(test_loader)
    metrics, class_report = model.compute_metrics(test_labels, test_preds, test_scores, file_path=file_path)
    print(test_acc)
    print(class_report)
    return test_loss, test_acc, metrics, test_preds, test_labels, test_scores

def train(train_data, train_labels, val_data, val_labels, args, neg=None, pos=None, train_mask=None, val_mask=None, test_data=None, test_labels=None, test_mask=None, num_classes=5, device=device, model_path='model/molsub.pth', filename='logging.png'):
    batch_size = args.batch_size
    num_workers = args.num_workers
    oversample_ratio = args.oversample_ratio
    downsample_ratio = args.downsample_ratio
    gan_dir = args.gan_dir
    patch_size = args.patch_size
    input_channel = args.input_channel
    feature = args.feature

    set_seed(args)

    loader = MolSubDataLoader(train_data, train_labels, val_data, val_labels, feature=feature, input_channel=input_channel, neg=neg, pos=pos, patch_size=patch_size, train_mask=train_mask, val_mask=val_mask, test_data=test_data, test_labels=test_labels, test_mask=test_mask, batch_size=batch_size, num_workers=num_workers, oversample_ratio=oversample_ratio, downsample_ratio=downsample_ratio, gan_data=process_gan_data(gan_dir), img_size=args.img_size)
    '''
    train_data: Any,
    train_labels: Any,
    val_data: Any,
    val_labels: Any,
    neg: Any | None = None,
    patch_size: Any | None = None,
    train_mask: Any | None = None,
    val_mask: Any | None = None,
    test_data: Any | None = None,
    test_labels: Any | None = None,
    test_mask: Any | None = None,
    '''
    print('success to init loader')
    
    train_loader, val_loader = loader.get_train_loader(), loader.get_val_loader()
    test_loader = loader.get_test_loader()
    del loader

    # WARNING：反面教材：loader不要多次加载！
    # labels = []
    # for _, label in train_loader:
    #     labels.extend(label)
    # print(f'训练数据分布：{torch.bincount(torch.tensor(labels))}')

    model = MolSub(train_loader, val_loader, args, num_classes=num_classes, device=device)
    model2 = None

    # train
    if not args.base_sampling:
        if (args.train_mode in [1, 2]) and args.confidence_threshold:
            print('use wei labels')
            final_test_data, wei_data, final_test_labels, wei_labels = train_test_split(test_data, test_labels, test_size=0.3, random_state=42)
            loader = MolSubDataLoader(None, None, wei_data, wei_labels, feature=feature, input_channel=input_channel, neg=neg, pos=pos, patch_size=patch_size, test_data=final_test_data, test_labels=final_test_labels, batch_size=batch_size, num_workers=num_workers, oversample_ratio=oversample_ratio, downsample_ratio=downsample_ratio, gan_data=process_gan_data(gan_dir), img_size=args.img_size)
            unlabeled_loader, test_loader = loader.get_val_loader(), loader.get_test_loader()
            model.train_visual(args, unlabeled_loader=unlabeled_loader, model_path=model_path, filename=filename) # 使用伪标签
            # model.train_model(args, model_path=model_path, filename=filename)
        else:
            model.train_model(args, model_path=model_path, filename=filename)
            # model2 = model.train_models(args, model_path=model_path, filename=filename)
    else:
        model.train_mxp_model(args, model_path=model_path, filename=filename)
    print('trained!')
    _, _, val_metrics, _, _, _ = test(val_loader, model, args, model2=model2, num_classes=model.num_classes, file_path=filename.replace('logging_', 'roc_val_'))
    test_loss, test_acc, metrics = None, None, None
    del train_loader, val_loader
    if test_loader:
        test_loss, test_acc, metrics, _, _, _ = test(test_loader, model, args, model2=model2, num_classes=model.num_classes, file_path=filename.replace('logging_', 'roc_test_'))
    return test_loss, test_acc, metrics, val_metrics

def train_next(train_data, train_labels, val_data, val_labels, args, neg=None, pos=None, train_mask=None, val_mask=None, test_data=None, test_labels=None, num_classes=5, device=device, model_path='model/molsub.pth', filename='logging.png'):
    batch_size = args.batch_size
    num_workers = args.num_workers
    patch_size = args.patch_size
    input_channel = args.input_channel
    feature = args.feature

    set_seed(args)

    loader = MolSubDataLoader(train_data, train_labels, val_data, val_labels, feature=feature, input_channel=input_channel, neg=neg, pos=pos, patch_size=patch_size, train_mask=train_mask, val_mask=val_mask, test_data=test_data, test_labels=test_labels, test_mask=test_mask, batch_size=batch_size, num_workers=num_workers, img_size=args.img_size)
    '''
    train_data: Any,
    train_labels: Any,
    val_data: Any,
    val_labels: Any,
    neg: Any | None = None,
    patch_size: Any | None = None,
    train_mask: Any | None = None,
    val_mask: Any | None = None,
    test_data: Any | None = None,
    test_labels: Any | None = None,
    test_mask: Any | None = None,
    '''
    print('success to init loader')
    
    train_loader, val_loader = loader.get_train_loader(), loader.get_val_loader()
    test_loader = loader.get_test_loader()
    del loader

    # WARNING：反面教材：loader不要多次加载！
    # labels = []
    # for _, label in train_loader:
    #     labels.extend(label)
    # print(f'训练数据分布：{torch.bincount(torch.tensor(labels))}')

    model = MolSub(train_loader, val_loader, args, num_classes=num_classes, device=device)
    model.load_model(model_path)
    # train
    if not args.base_sampling:
        model.train_model(args, model_path=model_path, filename=filename)
    else:
        model.train_mxp_model(args, model_path=model_path, filename=filename)
    print('trained!')
    test_loss, test_acc, metrics = None, None, None
    del train_loader, val_loader
    if test_loader:
        test_loss, test_acc, metrics, _, _, _ = test(test_loader, model, args, num_classes=model.num_classes, file_path=filename.replace('logging_', 'roc_'))
    return test_loss, test_acc, metrics

def k_fold_cross_validation(data, labels, args, neg=None, pos=None, mask=None, test_data=None, test_labels=None, test_mask=None, num_classes=5, device=device):
    k = args.k
    label = args.label
    model_type = args.model_type

    if k <= 1:
        train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
        test_data = val_data if test_data is None else test_data
        test_labels = val_labels if test_labels is None else test_labels
        _, _, metrics = train(train_data, train_labels, val_data, val_labels, args, neg=neg, pos=pos, test_data=test_data, test_labels=test_labels, num_classes=num_classes, device=device, model_path = f'model/molsub_{model_type}_{label}.pth', filename=f'logging/logging_{label}.png')
        results=[{
            'acc': round(metrics['accuracy'], 4),
            'sensitivity': round(metrics['sensitivity'], 4),
            'specificity': round(metrics['specificity'], 4),
            'ppv': round(metrics['ppv'], 4),
            'npv': round(metrics['npv'], 4),
            'avg_auc': round(metrics['avg_auc'], 4),
            'auc': [round(auc, 4) for auc in metrics['auc']]
        }]
        return results
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results, val_fold_results = [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f'Fold {fold + 1}/{k}', type(data), kf.split(data), train_idx.dtype)
        print(len(train_idx), len(val_idx))

        train_data = [data[idx] for idx in train_idx]
        train_labels = [labels[idx] for idx in train_idx]
        val_data = [data[idx] for idx in val_idx]
        val_labels = [labels[idx] for idx in val_idx]
        fold_test_data = val_data if test_data is None else test_data
        fold_test_labels = val_labels if test_labels is None else test_labels

        if mask is not None:
            train_mask = [mask[idx] for idx in train_idx]
            val_mask = [mask[idx] for idx in val_idx]
            test_mask = val_mask if test_mask is None else test_mask
        else:
            train_mask, val_mask, test_mask = None, None, None

        _, _, metrics, val_metrics = train(train_data, train_labels, val_data, val_labels, args, neg=neg, pos=pos, train_mask=train_mask, val_mask=val_mask, test_data=fold_test_data, test_labels=fold_test_labels, test_mask=test_mask, num_classes=num_classes, device=device, model_path = f'model/molsub_{model_type}_{label}_{fold}.pth', filename=f'logging/logging_{label}_{fold}.png')
        # if args.downsample_ratio:
        #     # 第二阶段训练(优化器状态有问题)
        #     _, _, metrics = train_next(train_data, train_labels, val_data, val_labels, args, neg=neg, pos=pos, train_mask=train_mask, val_mask=val_mask, test_data=val_data, test_labels=val_labels, num_classes=num_classes, device=device, model_path = f'model/molsub_{model_type}_{label}_{fold}_next.pth', filename=f'logging/logging_{label}_{fold}_next.png')
        '''
        train_data: Any,
        train_labels: Any,
        val_data: Any,
        val_labels: Any,
        args: Any,
        neg: Any | None = None,
        train_mask: Any | None = None,
        val_mask: Any | None = None,
        test_data: Any | None = None,
        test_labels: Any | None = None,
        test_mask
        '''
        
        fold_results.append({
            'fold': fold + 1,
            'acc': round(metrics['accuracy'], 4),
            'sensitivity': round(metrics['sensitivity'], 4),
            'specificity': round(metrics['specificity'], 4),
            'ppv': round(metrics['ppv'], 4),
            'npv': round(metrics['npv'], 4),
            'avg_auc': round(metrics['avg_auc'], 4),
            'auc': [round(auc, 4) for auc in metrics['auc']]
        })
        val_fold_results.append({
            'fold': fold + 1,
            'acc': round(val_metrics['accuracy'], 4),
            'sensitivity': round(val_metrics['sensitivity'], 4),
            'specificity': round(val_metrics['specificity'], 4),
            'ppv': round(val_metrics['ppv'], 4),
            'npv': round(val_metrics['npv'], 4),
            'avg_auc': round(val_metrics['avg_auc'], 4),
            'auc': [round(auc, 4) for auc in val_metrics['auc']]
        })
    
    calculate_confidence_interval([r['acc'] if not math.isnan(r['acc']) else 0 for r in fold_results])
    calculate_confidence_interval([r['sensitivity'] if not math.isnan(r['acc']) else 0 for r in fold_results])
    calculate_confidence_interval([r['specificity'] if not math.isnan(r['acc']) else 0 for r in fold_results])
    calculate_confidence_interval([r['ppv'] if not math.isnan(r['acc']) else 0 for r in fold_results])
    calculate_confidence_interval([r['npv'] if not math.isnan(r['acc']) else 0 for r in fold_results])


    calculate_confidence_interval([r['acc'] if not math.isnan(r['acc']) else 0 for r in val_fold_results])
    calculate_confidence_interval([r['sensitivity'] if not math.isnan(r['acc']) else 0 for r in val_fold_results])
    calculate_confidence_interval([r['specificity'] if not math.isnan(r['acc']) else 0 for r in val_fold_results])
    calculate_confidence_interval([r['ppv'] if not math.isnan(r['acc']) else 0 for r in val_fold_results])
    calculate_confidence_interval([r['npv'] if not math.isnan(r['acc']) else 0 for r in val_fold_results])
    return fold_results, val_fold_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument('--train_mode', type=int, default=0, help='An integer value, default is 0.')

    parser.add_argument('--k', type=int, default=5, help='An integer value, default is 5.')
    parser.add_argument('--batch_size', type=int, default=64, help='An integer value, default is 64.')
    parser.add_argument('--num_workers', type=int, default=2, help='An integer value, default is 2.')

    parser.add_argument('--num_epochs', type=int, default=300, help='An integer value, default is 300.')
    parser.add_argument('--save_epoch', type=int, default=10, help='An integer value, default is 10.')
    parser.add_argument('--train_diff_epochs', type=int, default=30, help='An integer value, default is 30.')
    
    parser.add_argument('--lr', type=float, default=0.0001, help='A float value, default is 0.0001.')
    parser.add_argument('--decay', type=float, default=0.005, help='A float value, default is 0.005.')
    parser.add_argument('--momentum', type=float, default=0.9, help='A float value, default is 0.9.')
    parser.add_argument('--pretrain', type=int, default=1, help='A float value, default is 1.')
    parser.add_argument('--dropout', type=float, default=0.5, help='A float value, default is 0.5.')
    parser.add_argument('--model_type', type=str, default='cnn',
                        help='A str value, default is cnn,\
                            chosen in [refers, resnet101, resnet18, cnn, densenet121, densenet169, lora, mobilevit, mob-cbam, mil].')
    parser.add_argument('--feature', type=int, default=0, help='An int value, default is 0.')
    parser.add_argument('--input_channel', type=int, default=1, help='An int value, default is 1.')
    parser.add_argument('--loss_type', type=str, default='ce', 
                        help='A str value, default is ce, chosen in [ce, focal, mwnl, sf1, ce+sf1, bce].')
    parser.add_argument('--train_diff', type=float, default=0.0, help='A float value, default is 0.0.')
    parser.add_argument('--confidence_threshold', type=float, default=0.0, help='A float value, default is 0.0.')
    parser.add_argument('--early_stopping_patience', type=int, default=100, help='An int value, default is 100.')

    parser.add_argument('--img_size', type=int, default=224, help='An int value, default is 224.')
    parser.add_argument('--patch_size', type=tuple_type, default=None, 
                        help='MIL Patch size in the format (width,height) with floating point numbers')
   
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/refers_checkpoint.pth",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument('--seed', type=int, default=21,
                        help="random seed for initialization")

    parser.add_argument('--label', type=str, default='ms', help='A str value, default is 分子分型.')
    parser.add_argument('--oversample_ratio', type=float, default=0.0, help='A float value, default is 0.0.')
    parser.add_argument('--downsample_ratio', type=float, default=0.0, help='A float value, default is 0.0.')
    parser.add_argument('--base_sampling', type=str, default='',
                        help='A str value, default is None, chosen in [instance, class, sqrt, cbrt].')
    parser.add_argument('--gan_dir', type=str, default='',
                        help='A str value, default is None, chosen in [data/images/gan_0_HER2].')
    
    parser.add_argument('--bound_size', type=int, default=100, help='An integer value, default is 100.')
    parser.add_argument('--clip_limit', type=float, default=0.003, help='A float value, default is 0.003.')
    parser.add_argument('--combine_data', type=int, default=0, help='An int value, default is 0.')
    parser.add_argument('--mask', type=int, default=0, help='An int value, default is 0.')
    parser.add_argument('--train_debug', type=int, default=0, help='An int value, default is 0.')
    
    parser.add_argument('--inf_model_path', type=str, default='model/prefered_model.pth', help='A str')
    parser.add_argument('--inf_img_path', type=str, default='examples/img1.dcm', help='A str')
    parser.add_argument('--inf_anno_path', type=str, default='examples/anno1.nii.gz', help='A str')

    args = parser.parse_args()

    k=args.k
    batch_size=args.batch_size
    num_workers=args.num_workers
    num_epochs=args.num_epochs
    save_epoch = args.save_epoch

    learning_rate=args.lr
    weight_decay=args.decay
    pretrained=True if args.pretrain == 1 else False

    bound_size = args.bound_size
    data_label = 'ms' if args.label in ['ms', 'l', 'tn', 'lab'] else args.label
    clip_limit = args.clip_limit

    oversample_ratio=args.oversample_ratio
    downsample_ratio=args.downsample_ratio
    dropout=args.dropout

    if args.combine_data:
        train_save_path = f'data/processed/train_data_{clip_limit}_{bound_size}_{data_label}_combine.pkl'
        val_save_path = f'data/processed/val_data_{clip_limit}_{bound_size}_{data_label}_combine.pkl'
        test_save_path = f'data/processed/test_data_{clip_limit}_{bound_size}_{data_label}_combine.pkl'
        args.input_channel = 2
    else:
        train_save_path = f'data/processed/train_data_{clip_limit}_{bound_size}_{data_label}.pkl'
        val_save_path = f'data/processed/val_data_{clip_limit}_{bound_size}_{data_label}.pkl'
        test_save_path = f'data/processed/test_data_{clip_limit}_{bound_size}_{data_label}.pkl'

        train_mask_path = f'data/processed/train_mask_{clip_limit}_{bound_size}_{data_label}.pkl'
        val_mask_path = f'data/processed/val_mask_{clip_limit}_{bound_size}_{data_label}.pkl'

    model_type = args.model_type
    if model_type == 'mobilevit':
        args.img_size = 256
    elif model_type == 'mil':
        args.patch_size = (64, 64) if args.patch_size is None else args.patch_size
        # args.loss_type = 'bce'
    elif model_type in ['cnn+sift', 'mob+sift', 'dense+sift', 'moe']:
        args.feature = 1
    
    if args.patch_size is not None:
        args.model_type = 'mil'
        
    train_data, train_labels, num_classes, min_size = load_data(train_save_path)
    train_mask = load_mask(train_mask_path) if args.mask else None
    print('train_mask is None?', train_mask is None)
    print(num_classes)
    print("Min size of train data", min_size)

    model_type = args.model_type
    neg, pos = None, None
    if num_classes > 2:
        if args.label == 'tn':
            neg = [0, 1, 2, 3]
        elif args.label == 'l':
            neg = [0, 1]
        elif args.label == 'lab':
            neg = [0]
            pos = [1]
        # else:
        #     neg = None # list(map(int, input('neg lables').split()))
        if neg is not None:
            num_classes = 2

    print(f'neg is {neg}, pos is {pos}')

    if args.train_mode == 0:
        print("K Fold on train")
        fold_results, val_fold_results = k_fold_cross_validation(train_data, train_labels, args, neg=neg, pos=pos, mask=train_mask, num_classes=num_classes, device=device)
        '''
        data: Any,
        labels: Any,
        args: Any,
        neg: Any | None = None,
        mask: Any | None = None,
        num_classes: int = 5,
        device: Any = device
        '''
        print('测试集上结果')
        for result in fold_results:
            print(result)
        print('验证集上结果')
        for result in val_fold_results:
            print(result)
    elif args.train_mode == 1:
        val_data, val_labels, _, _ = load_data(val_save_path)
        # train_data.extend(val_data)
        # train_labels.extend(val_labels)
        # del val_data, val_labels
        print("K Fold on chaoyang huigu, test on chaoyang qianzhan")
        fold_results, val_fold_results = k_fold_cross_validation(train_data, train_labels, args, neg=neg, pos=pos, test_data=val_data, test_labels=val_labels, num_classes=num_classes, device=device)
        print('测试集上结果')
        for result in fold_results:
            print(result)
        print('验证集上结果')
        for result in val_fold_results:
            print(result)
        
    elif args.train_mode == 2:
        test_data, test_labels, _, _ = load_data(test_save_path)
        # train_data.extend(test_data)
        # train_labels.extend(test_labels)
        # del test_data, test_labels
        print("K Fold on chaoyang huigu, test on luhe")
        fold_results, val_fold_results = k_fold_cross_validation(train_data, train_labels, args, neg=neg, pos=pos, test_data=test_data, test_labels=test_labels, num_classes=num_classes, device=device)
        '''
        data: Any,
        labels: Any,
        args: Any,
        neg: Any | None = None,
        mask: Any | None = None,
        num_classes: int = 5,
        device: Any = device
        '''
        print('测试集上结果')
        for result in fold_results:
            print(result)
        print('验证集上结果')
        for result in val_fold_results:
            print(result)
    elif args.train_mode == 3:
        print("HR combine_test on train")
        val_data, val_labels, _, _ = load_data(val_save_path)
        # test_data, test_labels, _, _ = load_data(test_save_path)
        loader = MolSubDataLoader(None, None, val_data, val_labels, input_channel=args.input_channel, batch_size=batch_size, num_workers=num_workers, oversample_ratio=oversample_ratio, downsample_ratio=downsample_ratio, img_size=args.img_size)
        '''
        train_data: Any,
        train_labels: Any,
        val_data: Any,
        val_labels: Any,
        neg: Any | None = None,
        patch_size: Any | None = None,
        train_mask: Any | None = None,
        val_mask: Any | None = None,
        test_data: Any | None = None,
        test_labels: Any | None = None,
        test_mask: Any | None = None,
        '''
        # 0.889788 0.513051 0.487712（联合prob训练的单模型泛化性能不足）
        model1 = f'model/molsub_cnn_HER2_3.pth'      # 0.77522  0.539868 0.481333
        # model2 = f'model/molsub_densenet121_HER2_3.pth' # 0.723168 0.662303 0.496849
        #                                               # 0.791376 0.632591 0.47323
        model2 = f'model/molsub_cnn_HER2_1.pth' # 
        _, test_acc, metrics, _, _, _ = test(loader.get_val_loader(), model1, args, model2=model2, num_classes=num_classes, device=device, file_path='logging/roc_cnn_HER2_c13.png')
        print(pd.DataFrame(metrics))
    elif args.train_mode == 4:
        print("Train on chaoyang huigu, val on chaoyang qianzhan, test on chaoyang qianzhan")
        val_data, val_labels, _, _ = load_data(val_save_path)
        test_data, test_labels = val_data, val_labels
        # test_data, test_labels, _, _ = load_data(test_save_path)
        test_loss, test_acc, metrics = train(train_data, train_labels, val_data, val_labels, args, neg=neg, pos=pos, test_data=test_data, test_labels=test_labels, num_classes=num_classes, device=device, model_path = f'model/molsub_{args.label}.pth')
        '''
        train_data: Any,
        train_labels: Any,
        val_data: Any,
        val_labels: Any,
        args: Any,
        neg: Any | None = None,
        train_mask: Any | None = None,
        val_mask: Any | None = None,
        test_data: Any | None = None,
        test_labels: Any | None = None,
        '''
        print('Loss:', test_loss, 'Acc:', test_acc)
        print(pd.DataFrame(metrics))
    elif args.train_mode == 5:
        val_data, val_labels, _, _ = load_data(val_save_path)
        test_data, test_labels, _, _ = load_data(test_save_path)
        loader = MolSubDataLoader(None, None, val_data, val_labels, test_data=test_data, test_labels=test_labels, batch_size=batch_size, num_workers=num_workers, oversample_ratio=oversample_ratio, downsample_ratio=downsample_ratio, img_size=args.img_size)
        '''
        train_data: Any,
        train_labels: Any,
        val_data: Any,
        val_labels: Any,
        neg: Any | None = None,
        patch_size: Any | None = None,
        train_mask: Any | None = None,
        val_mask: Any | None = None,
        test_data: Any | None = None,
        test_labels: Any | None = None,
        test_mask: Any | None = None,
        '''
        print('test HER2 on chaoyang qianzhan')
        _, test_acc, metrics, _, _, _ = test(loader.get_val_loader(), 'model/molsub_cnn_HER2_3.pth', args, 'model/7769_molsub_densenet121_tcAta_HER2.pth', num_classes, device, file_path='logging/roc_cnn_HER2_test_qianzhan.png')
        print(pd.DataFrame(metrics))
        print('test HER2 on luhe')
        _, test_acc, metrics, _, _, _ = test(loader.get_test_loader(), 'model/molsub_cnn_HER2_3.pth', args, 'model/7769_molsub_densenet121_tcAta_HER2.pth', num_classes, device, file_path='logging/roc_cnn_HER2_test_luhe.png')
        print(pd.DataFrame(metrics))
    elif args.train_mode == 6:
        from view_atten import generate_gradcam_visualization
        import os
        os.environ['TORCH_USE_CUDA_DSA'] = '1'

        input_channel = args.input_channel
        model = MolSub(None, None, args, num_classes=num_classes, device=device)

        # model.load_model(f'model/molsub_{model_type}_{args.label}_4.pth')
        model.load_model(args.inf_model_path) # ('model/densenet121-cbam/molsub_densenet121_ms_1.pth')

        print(model.model)
        # input('check model')
        print("num_classes", num_classes)
        '''
        if num_classes == 2:
            loader = MolSubDataLoader(None, None, train_data, train_labels, feature=args.feature, input_channel=input_channel, neg=neg, pos=pos, batch_size=batch_size, num_workers=num_workers, oversample_ratio=oversample_ratio, downsample_ratio=downsample_ratio, img_size=args.img_size)
            classfied_images = model.outsee_results(loader.get_val_loader())
        '''

        if model_type == 'densenet121-cbam':
            # cbam
            layers_to_visualize = ['features.pool0', 'features.denseblock1.0', 'features.denseblock1.1', 'features.transition1', 'features.denseblock2.0', 'features.denseblock2.1', 'features.transition2', 'features.denseblock3.0', 'features.denseblock3.1', 'features.transition3', 'features.denseblock4']
        elif model_type == 'densenet121':
            layers_to_visualize = ['features.conv0', 'features.denseblock1', 'features.transition1', 'features.denseblock2', 'features.transition2', 'features.denseblock3', 'features.transition3', 'features.denseblock4']
        elif 'cnn' in model_type:
            layers_to_visualize = ['conv1', 'conv2']
        elif model_type == 'mob-cbam':
            layers_to_visualize = ['features.0', 'features.1.block.0', 'features.1.block.1', 'features.1.block.2', 'features.2.block.2.0', 'features.2.block.2.1', 'features.3', 'features.4', 'features.6', 'features.8', 'features.10', 'features.12']
        elif model_type == 'resnet50':
            layers_to_visualize = ['conv1', 'layer1.0.conv1', 'layer1.0.conv2', 'layer1.0.conv3', 'layer1.1', 'layer1.2', 'layer2', 'layer3', 'layer4']
        elif model_type == 'refers':
            layers_to_visualize = ['transformer.embeddings.patch_embeddings']
        elif model_type == 'shuffle_v2':
            layers_to_visualize = ['conv1', 'stage2.0.branch1.0', 'stage2.0.branch1.2', 'stage2.0.branch2', 'stage2.1', 'stage2.2', 'stage2.3', 'stage3', 'stage4', 'conv5']
        
        image_path = args.inf_img_path
        annotation_path = args.inf_anno_path
        image_tensor = visulize(image_path, model.model, layers_to_visualize, args, mask_path=annotation_path, save_dir=f'layer_output/{args.label}', device=device)
        for target_class_index in range(num_classes):
            pred_score, pred = model.predict_img(image_tensor)
            print(f'pred to be class {pred}, view attention map in layer_output/{args.label}')
            generate_gradcam_visualization(image_path, model.model, layers_to_visualize, args, target_class_index, mask_path=annotation_path, pred=pred, pred_score=pred_score, save_dir=f'layer_output/{args.label}', device=device)

        debug = 0
        if debug:
            # 获取图像 HER2 L TN
            image_path000c = 'data_process_examples/roi_example_000_CC.jpg'  # 替换为你的图像路径
            image_path001c = 'data_process_examples/roi_example_001_CC.jpg'  # 替换为你的图像路径
            image_path110c = 'data_process_examples/roi_example_110_CC.jpg'  # 替换为你的图像路径
            image_path000m = 'data_process_examples/roi_example_000_MLO.jpg'  # 替换为你的图像路径
            image_path001m = 'data_process_examples/roi_example_001_MLO.jpg'  # 替换为你的图像路径
            image_path110m = 'data_process_examples/roi_example_110_MLO.jpg'  # 替换为你的图像路径

            # HER2 1（阴） 分型 1（LA） L 1 TN 0
            img_path_rc_110 = 'data/mammography subtype dataset/chaoyang huigu/BAILIANDI RCC/ser97311img00002.dcm' # if args.mask else image_path110c
            annotation_path_rc_110 = 'data/mammography subtype dataset/chaoyang huigu/BAILIANDI RCC/1.nii.gz'
            img_path1_rmlo_110 = 'data/mammography subtype dataset/chaoyang huigu/BAILIANDI RMLO/ser97311img00001.dcm' # if args.mask else image_path110m
            annotation_path1_rmlo_110 = 'data/mammography subtype dataset/chaoyang huigu/BAILIANDI RMLO/1.nii.gz'

            # HER2 0（阳） 分型 3（HER2+HR+） L 0 TN 0
            img_path_rc_000 = 'data/mammography subtype dataset/chaoyang huigu/BAOYINHUA RCC/ser121876img00004.dcm' # if args.mask else image_path000c
            annotation_path_rc_000 = 'data/mammography subtype dataset/chaoyang huigu/BAOYINHUA RCC/1.nii.gz'
            img_path1_rmlo_000 = 'data/mammography subtype dataset/chaoyang huigu/BAOYINHUA RMLO/ser121876img00001.dcm' # if args.mask else image_path000m
            annotation_path1_rmlo_000 = 'data/mammography subtype dataset/chaoyang huigu/BAOYINHUA RMLO/1.nii.gz'

            # HER2 1（阴） 分型 2（LB） L 1 TN 0
            img_path_rc_2 = 'data/mammography subtype dataset/chaoyang huigu/BAOYUGUI RCC/ser59718img00003.dcm' # if args.mask else image_path000c
            annotation_path_rc_2 = 'data/mammography subtype dataset/chaoyang huigu/BAOYUGUI RCC/1.nii.gz'
            img_path1_rmlo_2 = 'data/mammography subtype dataset/chaoyang huigu/BAOYUGUI RMLO/ser59718img00001.dcm' # if args.mask else image_path000m
            annotation_path1_rmlo_2 = 'data/mammography subtype dataset/chaoyang huigu/BAOYUGUI RMLO/1.nii.gz'

            # HER2 0（阳） 分型 4（HER2+HR-） L 0 TN 0
            img_path_lc_4 = 'data/mammography subtype dataset/chaoyang huigu/CAOLIYA LCC/ser88310img00004.dcm' # if args.mask else image_path000c
            annotation_path_lc_4 = 'data/mammography subtype dataset/chaoyang huigu/CAOLIYA LCC/1.nii.gz'
            img_path1_lmlo_4 = 'data/mammography subtype dataset/chaoyang huigu/CAOLIYA LMLO/ser88310img00002.dcm' # if args.mask else image_path000m
            annotation_path1_lmlo_4 = 'data/mammography subtype dataset/chaoyang huigu/CAOLIYA LMLO/1.nii.gz'
            go = 1
            while go:
                image_path = input('image_path:')
                annotation_path = input('annotation_path:')
                label = int(input('label:'))

                image_tensor = visulize(image_path, model.model, layers_to_visualize, args, mask_path=annotation_path, save_dir='layer_output/ms', device=device)
                pred_score, pred = model.predict_img(image_tensor, label=label)
                '''
                if label == 1 and pred == 1:
                    ans = 'TT'
                elif label == 0 and pred == 0:
                    ans = 'TF'
                elif label == 0 and pred == 1:
                    ans = 'FT'
                else:
                    ans = 'FF'
                print(ans, label, pred)
                '''
                if label == pred:
                    print('true result')
                    generate_gradcam_visualization(image_path, model.model, layers_to_visualize, args, label, mask_path=annotation_path, pred=pred, pred_score=pred_score, save_dir=f'layer_output/ms', device=device)

                go = input('go 1, exist 0')

            visulize('data/processed/clip/train_roi/ser30877img00003.jpg', model.model, layers_to_visualize, args, save_dir='layer_output', device=device)
            input('check')

            if args.label == 'ms':
                generate_gradcam_visualization(img_path_rc_000, model.model, layers_to_visualize, args, 0, mask_path=annotation_path_rc_000, save_dir='layer_output/ms_0/cc', device=device)
                generate_gradcam_visualization(img_path_rc_110, model.model, layers_to_visualize, args, 2, mask_path=annotation_path_rc_110, save_dir='layer_output/ms_2/cc', device=device)
                generate_gradcam_visualization(img_path1_rmlo_000, model.model, layers_to_visualize, args, 0, mask_path=annotation_path1_rmlo_000, save_dir='layer_output/ms_0/mlo', device=device)
                generate_gradcam_visualization(img_path1_rmlo_110, model.model, layers_to_visualize, args, 2, mask_path=annotation_path1_rmlo_110, save_dir='layer_output/ms_2/mlo', device=device)
                generate_gradcam_visualization(image_path001c, model.model, layers_to_visualize, args, 4, save_dir='layer_output/ms_4/cc', device=device)
                generate_gradcam_visualization(image_path001m, model.model, layers_to_visualize, args, 4, save_dir='layer_output/ms_4/mlo', device=device)
                generate_gradcam_visualization(img_path_rc_2, model.model, layers_to_visualize, args, 1, mask_path=annotation_path_rc_2, save_dir='layer_output/ms_1/cc', device=device)
                generate_gradcam_visualization(img_path1_rmlo_2, model.model, layers_to_visualize, args, 1, mask_path=annotation_path1_rmlo_2, save_dir='layer_output/ms_1/mlo', device=device)
                generate_gradcam_visualization(img_path_lc_4, model.model, layers_to_visualize, args, 3, mask_path=annotation_path_lc_4, save_dir='layer_output/ms_3/cc', device=device)
                generate_gradcam_visualization(img_path1_lmlo_4, model.model, layers_to_visualize, args, 3, mask_path=annotation_path1_lmlo_4, save_dir='layer_output/ms_3/mlo', device=device)
                
            if args.label == 'HER2':
                generate_gradcam_visualization(img_path_rc_000, model.model, layers_to_visualize, args, 0, mask_path=annotation_path_rc_000, save_dir='layer_output/HER2_0/cc', device=device)
                generate_gradcam_visualization(img_path_rc_110, model.model, layers_to_visualize, args, 1, mask_path=annotation_path_rc_110, save_dir='layer_output/HER2_1/cc', device=device)
                generate_gradcam_visualization(img_path1_rmlo_000, model.model, layers_to_visualize, args, 0, mask_path=annotation_path1_rmlo_000, save_dir='layer_output/HER2_0/mlo', device=device)
                generate_gradcam_visualization(img_path1_rmlo_110, model.model, layers_to_visualize, args, 1, mask_path=annotation_path1_rmlo_110, save_dir='layer_output/HER2_1/mlo', device=device)
                
                visulize(img_path_rc_000, model.model, layers_to_visualize, args, mask_path=annotation_path_rc_000, save_dir='layer_output/HER2_0/cc', device=device)
                visulize(img_path_rc_110, model.model, layers_to_visualize, args, mask_path=annotation_path_rc_110, save_dir='layer_output/HER2_1/cc', device=device)
                visulize(img_path1_rmlo_000, model.model, layers_to_visualize, args, mask_path=annotation_path1_rmlo_000, save_dir='layer_output/HER2_0/mlo', device=device)
                visulize(img_path1_rmlo_110, model.model, layers_to_visualize, args, mask_path=annotation_path1_rmlo_110, save_dir='layer_output/HER2_1/mlo', device=device)
            elif args.label == 'tn':
                generate_gradcam_visualization(image_path000c, model.model, layers_to_visualize, args, 0, save_dir='layer_output/TN_0/cc', device=device)
                generate_gradcam_visualization(image_path001c, model.model, layers_to_visualize, args, 1, save_dir='layer_output/TN_1/cc', device=device)
                generate_gradcam_visualization(image_path000m, model.model, layers_to_visualize, args, 0, save_dir='layer_output/TN_0/mlo', device=device)
                generate_gradcam_visualization(image_path001m, model.model, layers_to_visualize, args, 1, save_dir='layer_output/TN_1/mlo', device=device)

                visulize(image_path000c, model.model, layers_to_visualize, args, save_dir='layer_output/TN_0/cc', device=device)
                visulize(image_path001c, model.model, layers_to_visualize, args, save_dir='layer_output/TN_1/cc', device=device)
                visulize(image_path000m, model.model, layers_to_visualize, args, save_dir='layer_output/TN_0/mlo', device=device)
                visulize(image_path001m, model.model, layers_to_visualize, args, save_dir='layer_output/TN_1/mlo', device=device)
            elif args.label == 'l':
                generate_gradcam_visualization(img_path_rc_000, model.model, layers_to_visualize, args, 0, mask_path=annotation_path_rc_000, save_dir='layer_output/L_0/cc', device=device)
                generate_gradcam_visualization(img_path_rc_110, model.model, layers_to_visualize, args, 1, mask_path=annotation_path_rc_110, save_dir='layer_output/L_1/cc', device=device)
                generate_gradcam_visualization(img_path1_rmlo_000, model.model, layers_to_visualize, args, 0, mask_path=annotation_path1_rmlo_000, save_dir='layer_output/L_0/mlo', device=device)
                generate_gradcam_visualization(img_path1_rmlo_110, model.model, layers_to_visualize, args, 1, mask_path=annotation_path1_rmlo_110, save_dir='layer_output/L_1/mlo', device=device)

                visulize(image_path000c, model.model, layers_to_visualize, args, save_dir='layer_output/L_0/cc', device=device)
                visulize(image_path110c, model.model, layers_to_visualize, args, save_dir='layer_output/L_1/cc', device=device)
                visulize(image_path000m, model.model, layers_to_visualize, args, save_dir='layer_output/L_0/mlo', device=device)
                visulize(image_path110m, model.model, layers_to_visualize, args, save_dir='layer_output/L_1/mlo', device=device)

    elif args.train_mode == 7:
        train_save_path = f'data/processed/train_img_roi_{clip_limit}_{bound_size}_{args.label}.pkl'
        val_save_path = f'data/processed/val_img_roi_{clip_limit}_{bound_size}_{args.label}.pkl'
        test_save_path = f'data/processed/test_img_roi_{clip_limit}_{bound_size}_{args.label}.pkl'

        train_data, train_labels, num_classes, min_size = load_data(train_save_path)
        train_mask = load_mask(train_mask_path) if args.mask else None
        print('train_mask is None?', train_mask is None)
        print(num_classes)
        print("Min size of train data", min_size)

        print("K Fold on train")
        fold_results, val_fold_results = k_fold_cross_validation(train_data, train_labels, args, neg=neg, pos=pos, mask=train_mask, num_classes=num_classes, device=device)
        print('测试集上结果')
        for result in fold_results:
            print(result)
        print('验证集上结果')
        for result in val_fold_results:
            print(result)

        val_data, val_labels, _, _ = load_data(val_save_path)
        print("K Fold on chaoyang huigu, test on chaoyang qianzhan")
        fold_results, val_fold_results = k_fold_cross_validation(train_data, train_labels, args, neg=neg, pos=pos, test_data=val_data, test_labels=val_labels, num_classes=num_classes, device=device)
        print('测试集上结果')
        for result in fold_results:
            print(result)
        print('验证集上结果')
        for result in val_fold_results:
            print(result)
        del val_data, val_labels
        
        test_data, test_labels, _, _ = load_data(test_save_path)
        print("K Fold on chaoyang huigu, test on luhe")
        fold_results, val_fold_results = k_fold_cross_validation(train_data, train_labels, args, neg=neg, pos=pos, test_data=test_data, test_labels=test_labels, num_classes=num_classes, device=device)
        print('测试集上结果')
        for result in fold_results:
            print(result)
        print('验证集上结果')
        for result in val_fold_results:
            print(result)

    elif args.train_mode == 8:
        # TODO. train on hole chaoyang (80% 5折 + 20% test) 分层抽样/随机抽样
        val_data, val_labels, _, _ = load_data(val_save_path)
        train_data.extend(val_data)
        train_labels.extend(val_labels)
        del val_data, val_labels
        # 分层抽样
        X_train_stratified, X_test_stratified, y_train_stratified, y_test_stratified = train_test_split(train_data, train_labels, test_size=0.2, random_state=42, stratify=train_labels)
        # 随机抽样
        X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
        del train_data, train_labels
        '''
        print("K Fold on 朝阳分层抽样80%")
        fold_results = k_fold_cross_validation(X_train_stratified, y_train_stratified, args, neg=neg, num_classes=num_classes, device=device)
        for result in fold_results:
            print(result)
        print("test on 朝阳分层抽样20%")
        fold_results = k_fold_cross_validation(X_train_stratified, y_train_stratified, args, neg=neg, test_data=X_test_stratified, test_labels=y_test_stratified, num_classes=num_classes, device=device)
        for result in fold_results:
            print(result)
        '''
        print("K Fold on 朝阳随机抽样80%")
        fold_results, val_fold_results = k_fold_cross_validation(X_train, y_train, args, neg=neg, pos=pos, num_classes=num_classes, device=device)
        print('测试集上结果')
        for result in fold_results:
            print(result)
        print('验证集上结果')
        for result in val_fold_results:
            print(result)
        print("test on 朝阳随机抽样20%")
        fold_results, val_fold_results = k_fold_cross_validation(X_train, y_train, args, neg=neg, pos=pos, test_data=X_test, test_labels=y_test, num_classes=num_classes, device=device)
        print('测试集上结果')
        for result in fold_results:
            print(result)
        print('验证集上结果')
        for result in val_fold_results:
            print(result)

    elif args.train_mode == 9:
        #tune
        set_seed(args)

        val_data, val_labels, _, _ = load_data(val_save_path)
        '''
        train_data.extend(val_data)
        train_labels.extend(val_labels)
        del val_data, val_labels
        # 随机抽样
        data = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
        del train_data, train_labels
        '''
        data = [train_data, val_data, train_labels, val_labels] # 回顾only

        iterations = 100
        '''
        Tuner(args={'lr':args.lr, 'momentum':args.momentum, 'weight_decay':args.decay,
        'early_stopping_patience':args.early_stopping_patience, 't_max':20, 'eta_min':1e-5, 
        'dropout': dropout, 'batch_size': batch_size, 'oversample_ratio': oversample_ratio, 'downsample_ratio': downsample_ratio})(data=data, args=args, iterations=iterations)
        # or
        mgto_optimization(data=data, args=args, max_iterations=iterations)
        '''
    elif args.train_mode == 10:
        val_data, val_labels, _, _ = load_data(val_save_path)
        train_data.extend(val_data)
        train_labels.extend(val_labels)
        del val_data, val_labels
        # 随机抽样
        X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
        del train_data, train_labels
        print("test on 朝阳随机抽样20%")
        fold_results, val_fold_results = k_fold_cross_validation(X_train, y_train, args, neg=neg, pos=pos, test_data=X_test, test_labels=y_test, num_classes=num_classes, device=device)
        print('测试集上结果')
        for result in fold_results:
            print(result)
        print('验证集上结果')
        for result in val_fold_results:
            print(result)

    elif args.train_mode == 11:
        ## 统计性检验
        from test_auc_acc import n_delong_test_bootstrap, mcnemar_test
        model1 = 'model/densenet121-cbam/molsub_densenet121_l_4.pth'
        models = ['model/molsub_cnn_ms_1.pth', 'model/molsub_mob-cbam_ms_1.pth', 'model/densenet121_pre/molsub_densenet121_ms_1.pth']
        for i in range(4):
            model2 = f'model/densenet121-cbam/molsub_densenet121_l_{i}.pth'

            val_data, val_labels, _, _ = load_data(val_save_path)
            train_data.extend(val_data)
            train_labels.extend(val_labels)
            del val_data, val_labels

            loader = MolSubDataLoader(None, None, train_data, train_labels, neg=neg, batch_size=batch_size, num_workers=num_workers, oversample_ratio=oversample_ratio, downsample_ratio=downsample_ratio, img_size=args.img_size)
            _, _, _, test_preds1, test_labels12, test_scores1 = test(loader.get_val_loader(), model1, args, None, num_classes, device)
            _, _, _, test_preds2, _, test_scores2 = test(loader.get_val_loader(), model2, args, None, num_classes, device)
            n_delong_test_bootstrap(test_labels12, test_scores1, test_scores2)
            mcnemar_test(test_labels12, test_preds1, test_preds2)

            input('检验下一折')
            