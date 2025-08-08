#!/bin/bash

LOG_DIR="logging"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/inference_$(date +%Y%m%d_%H%M%S).log"

echo "=== Inferencing Script Started ===" > "$LOGFILE"
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

MODEL_TYPE="densenet121-cbam"  # 可选: densenet121-cbam, mob-cbam, cnn, resnet18, resnet101, densenet121, mobilevit, mil, etc.
PRETRAIN=1                     # 是否使用预训练权重 (1=是, 0=否)
FEATURE=0                      # 是否使用SIFT特征

TRAIN_MODE=6                   # 推理可视化模式（自定义用途）

SEED=21                        # 随机种子，确保可复现

INF_MODEL_PATH="model/prefered_model_for_ms.pth"
INF_IMG_PATH="examples/img1.dcm"
INF_ANNO_PATH="examples/anno1.nii.gz"

show_help() {
    cat << 'EOF'
Usage: ./train.sh [OPTIONS]

推理脚本参数说明：

 -l, --label VALUE           分类任务类型：ms, l, tn, HER2 (默认: ms)
 -b, --bound-size N          肿瘤边界扩展大小 (与预处理的数据保持一致，默认: 100)
 -c, --clip-limit VALUE      CLAHE 对比度限制 (与预处理的数据保持一致，默认: 0.003)
 -i, --img-size N            输入图像尺寸 (与模型输入匹配，默认: 224)
 -C, --input-channel N       输入通道数（1=灰度图）(默认: 1)
 -d, --combine-data 0|1      是否使用双通道 (与预处理的数据保持一致，默认: 0)
 -m, --mask 0|1              是否使用掩码增强 (默认: 0)
 -M, --model-type NAME       模型类型 (默认: densenet121-cbam)
 -P, --pretrain 0|1          是否使用预训练权重 (1=是, 0=否) (默认: 1)
 -f, --feature 0|1           是否使用SIFT特征 (默认: 0)
 -t, --train-mode N          训练模式 (默认: 8)
 -I, --seed N                随机种子 (默认: 21)
 -O inf-model-path PATH
 -G inf-img-path PATH
 -A inf-anno-path PATH

 -h, --help                  显示此帮助信息

示例:
 ./inference.sh -l l -O model/prefered_model_for_l.pth -G examples/img1.dcm -A examples/anno1.nii.gz
EOF
}

while getopts "l:b:c:i:C:d:m:M:P:f:t:I:O:G:A:h" opt; do
    case $opt in
        l) LABEL="$OPTARG" ;;
        b) BOUND_SIZE="$OPTARG" ;;
        c) CLIP_LIMIT="$OPTARG" ;;
        i) IMG_SIZE="$OPTARG" ;;
        C) INPUT_CHANNEL="$OPTARG" ;;
        d) COMBINE_DATA="$OPTARG" ;;
        m) MASK="$OPTARG" ;;
        M) MODEL_TYPE="$OPTARG" ;;
        P) PRETRAIN="$OPTARG" ;;
        f) FEATURE="$OPTARG" ;;
        t) TRAIN_MODE="$OPTARG" ;;
        I) SEED="$OPTARG" ;;
        O) INF_MODEL_PATH="$OPTARG" ;;
        G) INF_IMG_PATH="$OPTARG" ;;
        A) INF_ANNO_PATH="$OPTARG" ;;
        h) show_help; exit 0 ;;
        ?) echo "无效选项: -$OPTARG" >&2; show_help; exit 1 ;;
    esac
done

shift $((OPTIND-1))

echo "推理日志保存在: $LOGFILE"
echo "🚀 开始推理..."
python train.py \
    --bound_size $BOUND_SIZE \
    --clip_limit $CLIP_LIMIT \
    --label "$LABEL" \
    \
    --train_mode $TRAIN_MODE \
    --pretrain $PRETRAIN \
    --model_type $MODEL_TYPE \
    --feature $FEATURE \
    --input_channel $INPUT_CHANNEL \
    \
    --img_size $IMG_SIZE \
    \
    --seed $SEED \
    \
    --combine_data $COMBINE_DATA \
    --mask $MASK \
    \
    --inf_model_path $INF_MODEL_PATH \
    --inf_img_path $INF_IMG_PATH \
    --inf_anno_path $INF_ANNO_PATH \
    >> "$LOGFILE" 2>&1

if [ $? -eq 0 ]; then
    echo "" >> "$LOGFILE"
    echo "✅ Inferencing completed successfully."
    echo "✅ Inferencing completed successfully." >> "$LOGFILE"
else
    echo "" >> "$LOGFILE"
    echo "❌ Inferencing failed with exit code $?"
    echo "❌ Inferencing failed with exit code $?" >> "$LOGFILE"
fi

echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOGFILE"
