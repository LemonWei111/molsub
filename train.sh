#!/bin/bash

LOG_DIR="logging"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo "=== Training Script Started ===" > "$LOGFILE"
echo "Command: $0 $*" >> "$LOGFILE"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOGFILE"
echo "================================" >> "$LOGFILE"
echo "" >> "$LOGFILE"

LABEL="ms"                     # Chose from ['ms', 'l', 'tn', 'HER2']
BOUND_SIZE=100                 # 肿瘤边界扩展大小
CLIP_LIMIT=0.003               # CLAHE 对比度限制
IMG_SIZE=224                   # 输入图像尺寸
INPUT_CHANNEL=1                # 输入通道数（1=灰度图）
COMBINE_DATA=0                 # 是否使用双通道
MASK=0                         # 是否使用掩码增强
OVERSAMPLE_RATIO=0.0           # 过采样比例
DOWNSAMPLE_RATIO=0.0           # 欠采样比例
BASE_SAMPLING=""               # 基础采样策略
GAN_DIR=""                     # GAN生成图像目录（可选）

MODEL_TYPE="densenet121-cbam"  # 可选: densenet121-cbam, mob-cbam, cnn, resnet18, resnet101, densenet121, mobilevit, mil, etc.
PRETRAIN=1                     # 是否使用预训练权重 (1=是, 0=否)
DROPOUT=0.3                    # Dropout 比例
FEATURE=0                      # 是否使用SIFT特征

LOSS_TYPE="ce"                 # 可选: ce, focal, mwnl, sf1, ce+sf1, bce, etc.

LR=0.0001                      # 初始学习率
DECAY=0.005                    # 权重衰减 (L2 正则)
MOMENTUM=0.9                   # SGD 动量
CONFIDENCE_THRESHOLD=0.0       # 置信度阈值（用于动态采样）

TRAIN_MODE=8                   # 训练模式（自定义用途）
K=5                            # 交叉验证折数
BATCH_SIZE=8
NUM_WORKERS=2
NUM_EPOCHS=300
SAVE_EPOCH=10                  # 每 N 个 epoch 保存一次
TRAIN_DIFF=0.0
TRAIN_DIFF_EPOCHS=30           # 特征提取层训练 epoch 数
EARLY_STOPPING_PATIENCE=100    # 早停耐心值

SEED=21                        # 随机种子，确保可复现
TRAIN_DEBUG=0                  # 调试模式 (1=开启，仅训练少量数据)

show_help() {
    cat << 'EOF'
Usage: ./train.sh [OPTIONS]

训练脚本参数说明：

 -l, --label VALUE           标签类型：ms, l, tn, HER2 (默认: ms)
 -b, --bound-size N          肿瘤边界扩展大小 (与预处理的数据保持一致，默认: 100)
 -c, --clip-limit VALUE      CLAHE 对比度限制 (与预处理的数据保持一致，默认: 0.003)
 -i, --img-size N            输入图像尺寸 (与模型输入匹配，默认: 224)
 -C, --input-channel N       输入通道数（1=灰度图）(默认: 1)
 -d, --combine-data 0|1      是否使用双通道 (与预处理的数据保持一致，默认: 0)
 -m, --mask 0|1              是否使用掩码增强 (默认: 0)
 -O, --oversample-ratio V    过采样比例 (默认: 0.0)
 -D, --downsample-ratio V    欠采样比例 (默认: 0.0)
 -B, --base-sampling STR     基础采样策略 (默认: 空)
 -G, --gan-dir PATH          GAN生成图像目录（可选）
 -M, --model-type NAME       模型类型 (默认: densenet121-cbam)
 -P, --pretrain 0|1          是否使用预训练权重 (1=是, 0=否) (默认: 1)
 -r, --dropout VALUE         Dropout 比例 (默认: 0.3)
 -f, --feature 0|1           是否使用SIFT特征 (默认: 0)
 -p, --patch-size STR        MIL patch 尺寸，如 "(16,16)" (默认: None)
 -L, --loss-type NAME        损失函数类型 (默认: ce)
 -R, --lr VALUE              初始学习率 (默认: 0.0001)
 -e, --decay VALUE           权重衰减 (默认: 0.005)
 -o, --momentum VALUE        SGD 动量 (默认: 0.9)
 -T, --confidence-threshold V 置信度阈值 (默认: 0.0)
 -t, --train-mode N          训练模式 (默认: 8)
 -k N                        交叉验证折数 (默认: 5)
 -s, --batch-size N          Batch大小 (默认: 8)
 -w, --num-workers N         数据加载线程数 (默认: 2)
 -E, --num-epochs N          总epoch数 (默认: 300)
 -S, --save-epoch N          每N个epoch保存一次 (默认: 10)
 -a, --train-diff 0|1        是否进行难分类训练 (默认: 0)
 -F, --train-diff-epochs N   难分类训练epoch数 (默认: 30)
 -N, --early-stopping-patience N 早停耐心值 (默认: 100)
 -I, --seed N                随机种子 (默认: 21)
 -x, --train-debug 0|1       调试模式 (1=开启) (默认: 0)

 -h, --help                  显示此帮助信息

示例:
 ./train.sh -l tn -i 256 -M resnet18 -s 16
EOF
}

while getopts "l:b:c:i:C:d:m:O:D:B:G:M:P:r:f:L:R:e:o:T:t:k:s:w:E:S:a:F:N:I:x:h" opt; do
    case $opt in
        l) LABEL="$OPTARG" ;;
        b) BOUND_SIZE="$OPTARG" ;;
        c) CLIP_LIMIT="$OPTARG" ;;
        i) IMG_SIZE="$OPTARG" ;;
        C) INPUT_CHANNEL="$OPTARG" ;;
        d) COMBINE_DATA="$OPTARG" ;;
        m) MASK="$OPTARG" ;;
        O) OVERSAMPLE_RATIO="$OPTARG" ;;
        D) DOWNSAMPLE_RATIO="$OPTARG" ;;
        B) BASE_SAMPLING="$OPTARG" ;;
        G) GAN_DIR="$OPTARG" ;;
        M) MODEL_TYPE="$OPTARG" ;;
        P) PRETRAIN="$OPTARG" ;;
        r) DROPOUT="$OPTARG" ;;
        f) FEATURE="$OPTARG" ;;
        L) LOSS_TYPE="$OPTARG" ;;
        R) LR="$OPTARG" ;;
        e) DECAY="$OPTARG" ;;
        o) MOMENTUM="$OPTARG" ;;
        T) CONFIDENCE_THRESHOLD="$OPTARG" ;;
        t) TRAIN_MODE="$OPTARG" ;;
        k) K="$OPTARG" ;;
        s) BATCH_SIZE="$OPTARG" ;;
        w) NUM_WORKERS="$OPTARG" ;;
        E) NUM_EPOCHS="$OPTARG" ;;
        S) SAVE_EPOCH="$OPTARG" ;;
        a) TRAIN_DIFF="$OPTARG" ;;
        F) TRAIN_DIFF_EPOCHS="$OPTARG" ;;
        N) EARLY_STOPPING_PATIENCE="$OPTARG" ;;
        I) SEED="$OPTARG" ;;
        x) TRAIN_DEBUG="$OPTARG" ;;
        h) show_help; exit 0 ;;
        ?) echo "无效选项: -$OPTARG" >&2; show_help; exit 1 ;;
    esac
done

shift $((OPTIND-1))

if [ -z "${OVERSAMPLE_RATIO}" ] || [ "$OVERSAMPLE_RATIO" = "0.0" ]; then
    case "$LABEL" in
        "tn"|"TN")
            OVERSAMPLE_RATIO=1.7
            ;;
        "l"|"L"|"luminal")
            OVERSAMPLE_RATIO=1.3
            ;;
        *)
            OVERSAMPLE_RATIO=1.5
            ;;
    esac
    echo "📊 根据 LABEL='$LABEL' 自动设置 OVERSAMPLE_RATIO=$OVERSAMPLE_RATIO"
else
    echo "📌 使用手动设置的 OVERSAMPLE_RATIO=$OVERSAMPLE_RATIO"
fi

echo "训练日志保存在: $LOGFILE"
echo "🚀 开始训练模型..."
python train.py \
    --bound_size $BOUND_SIZE \
    --clip_limit $CLIP_LIMIT \
    --label "$LABEL" \
    \
    --train_mode $TRAIN_MODE \
    --k $K \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --num_epochs $NUM_EPOCHS \
    --save_epoch $SAVE_EPOCH \
    --train_diff_epochs $TRAIN_DIFF_EPOCHS \
    --lr $LR \
    --decay $DECAY \
    --momentum $MOMENTUM \
    --pretrain $PRETRAIN \
    --dropout $DROPOUT \
    --model_type $MODEL_TYPE \
    --feature $FEATURE \
    --input_channel $INPUT_CHANNEL \
    --loss_type $LOSS_TYPE \
    --train_diff $TRAIN_DIFF \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    \
    --img_size $IMG_SIZE \
    \
    --pretrained_dir "$PRETRAINED_DIR" \
    --seed $SEED \
    \
    --oversample_ratio $OVERSAMPLE_RATIO \
    --downsample_ratio $DOWNSAMPLE_RATIO \
    --base_sampling "$BASE_SAMPLING" \
    --gan_dir "$GAN_DIR" \
    \
    --combine_data $COMBINE_DATA \
    --mask $MASK \
    --train_debug $TRAIN_DEBUG \
    >> "$LOGFILE" 2>&1

if [ $? -eq 0 ]; then
    echo "" >> "$LOGFILE"
    echo "✅ Training completed successfully."
    echo "✅ Training completed successfully." >> "$LOGFILE"
else
    echo "" >> "$LOGFILE"
    echo "❌ Training failed with exit code $?"
    echo "❌ Training failed with exit code $?" >> "$LOGFILE"
fi

echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOGFILE"
