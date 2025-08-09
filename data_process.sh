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
BOUND_SIZE=100
CLIP_LIMIT=0.003  
LABEL="ms"

TRAIN_EXCEL_PATH=""
TRAIN_DATA_DIR=""

show_help() {
    cat << 'EOF'
Usage: ./data_process.sh [OPTIONS]

Parameter description of Data Processing script:

 -b, --bound_size  N           Tumor boundary expansion size (default: 100)
 -c, --clip_limit  V           CLAHE contrast limit (should be no less than 0, default: 0.003)
 -l, --label  VALUE            Classification task types: ms, HER2 (default: ms)
 -P, --train_excel_path  PATH  Path of the table of training data
 -D, --train_data_dir  PATH    Path of the training data

 -h, --help                    Display this help information

Example:
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
        ?) echo "Invalid Option: -$OPTARG" >&2; show_help; exit 1 ;;
    esac
done

shift $((OPTIND-1))

echo "Running data_process.sh with the following parameters:"
echo "Bound Size: $BOUND_SIZE"
echo "Clip Limit: $CLIP_LIMIT"
echo "Label: $LABEL"

echo "Data processing log is saved at: $LOGFILE"
echo "🚀 Start processing..."
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
