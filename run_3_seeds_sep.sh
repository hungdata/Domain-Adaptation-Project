#!/bin/bash
# ==============================================================================
# Script chạy tự động 3 thuật toán TÁCH BATCH (DSAN_SEP family) x 3 Seeds
# ==============================================================================

set -euo pipefail

SEEDS=(2026 2027 2028)
METHODS=(
    "train_dsan_sep.py"
    "train_dsane_sep.py"
    "train_dsane_mcc_sep.py"
)

# Thư mục gốc của dataset
DATA_ROOT="/home/ezycloudx-admin/Desktop/paper/uda_fixed_folders/"

mkdir -p results
SUMMARY_FILE="results/final_results_summary_sep.txt"
echo "============================================================" > $SUMMARY_FILE
echo "  BẮT ĐẦU CHẠY THỰC NGHIỆM: 3 METHODS (TÁCH BATCH) × 3 SEEDS" >> $SUMMARY_FILE
echo "  Thời gian bắt đầu: $(date)" >> $SUMMARY_FILE
echo "  Seeds: ${SEEDS[*]}" >> $SUMMARY_FILE
echo "============================================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

NUM_METHODS=${#METHODS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_RUNS=$((NUM_METHODS * NUM_SEEDS))
CURRENT_RUN=0

for METHOD_FILE in "${METHODS[@]}"; do
    METHOD_NAME=$(basename "$METHOD_FILE" .py)
    # Cắt bỏ tiền tố "train_" và viết hoa để tạo tag
    METHOD_TAG=${METHOD_NAME#"train_"}
    METHOD_TAG=$(echo "$METHOD_TAG" | tr '[:lower:]' '[:upper:]')
    
    echo "=========================================================="
    echo "  PHƯƠNG PHÁP: $METHOD_TAG ($METHOD_FILE)"
    echo "=========================================================="

    for SEED in "${SEEDS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        echo "  [$CURRENT_RUN/$TOTAL_RUNS] Đang chạy Seed $SEED..."
        
        LOG_FILE="results/log_${METHOD_TAG}_seed${SEED}.txt"
        echo "    Log file: $LOG_FILE"
        
        python "methods/$METHOD_FILE" \
            --seed "$SEED" \
            --data-root "$DATA_ROOT" \
            --output-dir "working/seed_$SEED" \
            --num-workers 10 \
            --batch-size 64 \
            > "$LOG_FILE" 2>&1
            
        echo "    ✅ Seed $SEED Done!"
        
        # Bóc tách kết quả từ dòng cuối cùng (giả định script in ra dạng: [RESULT] ...)
        RESULT_LINE=$(grep "\[RESULT\]" "$LOG_FILE" | tail -n 1 || true)
        if [ -n "$RESULT_LINE" ]; then
            echo "      Kết quả: $RESULT_LINE"
            echo "[$METHOD_TAG] Seed $SEED -> $RESULT_LINE" >> $SUMMARY_FILE
        else
            echo "      [CẢNH BÁO] Không tìm thấy kết quả cuối cùng trong log!"
            echo "[$METHOD_TAG] Seed $SEED -> ERROR" >> $SUMMARY_FILE
        fi
        echo ""
    done
    echo "" >> $SUMMARY_FILE
done

echo "============================================================"
echo "  HOÀN THÀNH TẤT CẢ!"
echo "  Thời gian kết thúc: $(date)"
echo "  Xem kết quả tổng hợp tại: $SUMMARY_FILE"
echo "============================================================"
echo "Thời gian kết thúc: $(date)" >> $SUMMARY_FILE
