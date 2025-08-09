#!/bin/bash

LOG_DIR="logging"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/inference_$(date +%Y%m%d_%H%M%S).log"

echo "=== Inferencing Script Started ===" > "$LOGFILE"
echo "Command: $0 $*" >> "$LOGFILE"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOGFILE"
echo "================================" >> "$LOGFILE"
echo "" >> "$LOGFILE"

LABEL="ms"
BOUND_SIZE=100
CLIP_LIMIT=0.003
IMG_SIZE=224
INPUT_CHANNEL=1
COMBINE_DATA=0
MASK=0

MODEL_TYPE="densenet121-cbam"
PRETRAIN=1
FEATURE=0

TRAIN_MODE=6

SEED=21

INF_MODEL_PATH="model/prefered_model_for_ms.pth"
INF_IMG_PATH="examples/img1.dcm"
INF_ANNO_PATH="examples/anno1.nii.gz"

show_help() {
    cat << 'EOF'
Usage: ./inference.sh [OPTIONS]

Parameter description of Inferencing script:

 -l, --label  VALUE           Classification task types: ms, l, tn, HER2 (default: ms)
 -b, --bound-size  N          Tumor boundary expansion size (consistent with preprocessed data, default: 100)
 -c, --clip-limit  VALUE      CLAHE contrast limit (consistent with preprocessed data, default: 0.003)
 -i, --img-size  N            Input image size (matches model input, default: 224)
 -C, --input-channel  N       Number of input channels (1=grayscale image) (default: 1)
 -d, --combine-data  0|1      Whether to use dual channels (consistent with preprocessed data, default: 0)
 -m, --mask  0|1              Whether to use mask enhancement (default: 0)
 -M, --model-type  NAME       Model type (default: densenet121 cbam)
 -P, --pretrain  0|1          Whether to use pretrained weights (1=Yes, 0=No) (default: 1)
 -f, --feature  0|1           Whether to use SIFT features (default: 0)
 -t, --train-mode  N          Training mode (default: 6)
 -I, --seed  N                Random seeds to ensure reproducibility (default: 21)
 -O, --inf-model-path  PATH   Path of model weights (default: model/prefered_model_for_ms.pth)
 -G, --inf-img-path  PATH     Path of the image that requires inference (default: examples/img1.dcm)
 -A, --inf-anno-path  PATH    Path of the annotation of the image that requires inference (default: examples/anno1.nii.gz)

 -h, --help                   Display this help information

Example:
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
        ?) echo "Invalid Option: -$OPTARG" >&2; show_help; exit 1 ;;
    esac
done

shift $((OPTIND-1))

echo "Inferencing log is saved at: $LOGFILE"
echo "🚀 Start inferencing..."
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
