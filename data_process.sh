#!/bin/bash

LOG_DIR="logging"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/data_process_$(date +%Y%m%d_%H%M%S).log"

echo "=== Data Processing Script Started ===" > "$LOGFILE"
echo "Command: $0 $*" >> "$LOGFILE"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOGFILE"
echo "================================" >> "$LOGFILE"
echo "" >> "$LOGFILE"

DATA_NAME=3
BOUND_SIZE=100  # The optimal value selected for the experiment
CLIP_LIMIT=0.003  # should be more than 0
LABEL="ms"  # Chose from ['ms', 'HER2']

TRAIN_EXCEL_PATH=""
TRAIN_DATA_DIR=""

show_help() {
    cat << 'EOF'
Usage: ./train.sh [OPTIONS]

推理脚本参数说明：

 -b, --bound_size N
 -c, --clip_limit V
 -l, --label VALUE
 -P, --train_excel_path PATH
 -D, --train_data_dir PATH

 -h, --help                  显示此帮助信息

示例:
 ./data_process.sh -P data/mammography\ subtype\ dataset/beiyou\ excel/chaoyang\ retrospective_processed.xlsx -D examples
EOF
}

while getopts "b:c:l:P:D:h" opt; do
    case $opt in
        b) BOUND_SIZE="$OPTARG" ;;
        c) CLIP_LIMIT="$OPTARG" ;;
        l) LABEL="$OPTARG" ;;
        P) TRAIN_EXCEL_PATH="$OPTARG" ;;
        D) TRAIN_DATA_DIR="$OPTARG" ;;
        h) show_help; exit 0 ;;
        ?) echo "无效选项: -$OPTARG" >&2; show_help; exit 1 ;;
    esac
done

shift $((OPTIND-1))

echo "Running your_script.py with the following parameters:"
echo "Bound Size: $BOUND_SIZE"
echo "Clip Limit: $CLIP_LIMIT"
echo "Label: $LABEL"

echo "数据处理日志保存在: $LOGFILE"
echo "🚀 开始处理..."
python data_process.py \
    --data_name $DATA_NAME \
    --bound_size $BOUND_SIZE \
    --clip_limit $CLIP_LIMIT \
    --label "$LABEL" \
    --train_excel_path "$TRAIN_EXCEL_PATH" \
    --train_data_dir "$TRAIN_DATA_DIR" \
    >> "$LOGFILE" 2>&1

if [ $? -eq 0 ]; then
    echo "" >> "$LOGFILE"
    echo "✅ Data processing completed successfully."
    echo "✅ Data processing completed successfully." >> "$LOGFILE"
else
    echo "" >> "$LOGFILE"
    echo "❌ Data processing failed with exit code $?"
    echo "❌ Data processing failed with exit code $?" >> "$LOGFILE"
fi

echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOGFILE"
