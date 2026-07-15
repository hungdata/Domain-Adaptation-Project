#!/bin/bash

# ==============================================================================
# FINAL RESULTS SCRIPT - 3 SEEDS × 8 METHODS
# ==============================================================================
# Script này chạy toàn bộ 8 thuật toán (Baseline + Đề xuất) trên 3 hạt giống
# ngẫu nhiên (2026, 2027, 2028) để lấy kết quả Trung Bình ± Độ Lệch Chuẩn.
#
# Cấu trúc output:
#   working/seed_2026/ERM_logs/final_test_metrics.json
#   working/seed_2026/DANN_logs/final_test_metrics.json
#   ...
#   working/seed_2027/ERM_logs/final_test_metrics.json
#   ...
# ==============================================================================

set -euo pipefail  # Dừng ngay nếu có lỗi, kể cả trong pipe

SEEDS=(2026 2027 2028)
METHODS=(
    "train_erm.py"
    "train_dann.py"
    "train_cdan.py"
    "train_cdane.py"
    "train_cdanc.py"
    "train_dsan.py"
    "train_dsane.py"
    "train_dsanc.py"
    "train_dsane_mcc.py"
    "train_cdane_mcc.py"
)

# Thư mục gốc của dataset
DATA_ROOT="/home/ezycloudx-admin/Desktop/paper/uda_fixed_folders/"

mkdir -p results
SUMMARY_FILE="results/final_results_summary.txt"
echo "============================================================" > $SUMMARY_FILE
echo "  BẮT ĐẦU CHẠY THỰC NGHIỆM: 8 METHODS × 3 SEEDS" >> $SUMMARY_FILE
echo "  Thời gian bắt đầu: $(date)" >> $SUMMARY_FILE
echo "  Seeds: ${SEEDS[*]}" >> $SUMMARY_FILE
echo "============================================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

NUM_METHODS=${#METHODS[@]}
echo "Tìm thấy $NUM_METHODS thuật toán cần chạy: ${METHODS[*]}"

TOTAL_RUNS=$(( ${#METHODS[@]} * ${#SEEDS[@]} ))
CURRENT_RUN=0

for method_file in "${METHODS[@]}"; do
    # Lấy tên method từ bên trong file Python
    METHOD_NAME=$(grep "^METHOD = " methods/$method_file | cut -d '"' -f 2)
    
    echo ""
    echo "=========================================================="
    echo "  PHƯƠNG PHÁP: $METHOD_NAME ($method_file)"
    echo "=========================================================="
    echo "--- $METHOD_NAME ---" >> $SUMMARY_FILE



    # CHẠY TUẦN TỰ (SEQUENTIAL) - TỐI ƯU NUM_WORKERS = 10
    for seed in "${SEEDS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        OUTPUT_DIR="./working/seed_${seed}"
        LOG_FILE="results/log_${METHOD_NAME}_seed${seed}.txt"

        echo "  [$CURRENT_RUN/$TOTAL_RUNS] Đang chạy Seed $seed..."
        echo "    Log file: $LOG_FILE"
        
        # Chạy file Python tuần tự, tăng tốc CPU với num_workers=10
        python "methods/$method_file" \
            --seed "$seed" \
            --data-root "$DATA_ROOT" \
            --output-dir "$OUTPUT_DIR" \
            --num-workers 10 \
            --batch-size 64 \
            --epochs 150 \
            2>&1 | tee "$LOG_FILE"

        # Đọc kết quả sau khi chạy xong seed này
        JSON_FILE="${OUTPUT_DIR}/${METHOD_NAME}_logs/final_test_metrics.json"
        
        if [ -f "$JSON_FILE" ]; then
            TGT_ACC=$(python3 -c 'import json; d=json.load(open("'$JSON_FILE'")); print("{:.4f}".format(d["target_test_acc"]))')
            SRC_ACC=$(python3 -c 'import json; d=json.load(open("'$JSON_FILE'")); print("{:.4f}".format(d["source_test_acc"]))')
            TGT_F1=$(python3 -c 'import json; d=json.load(open("'$JSON_FILE'")); print("{:.4f}".format(d.get("target_test_f1", 0)))')
            echo "    ✅ Seed $seed Done! Src=$SRC_ACC | Tgt=$TGT_ACC | F1=$TGT_F1"
            echo "  Seed $seed: Tgt_Acc=$TGT_ACC | Src_Acc=$SRC_ACC | Tgt_F1=$TGT_F1" >> $SUMMARY_FILE
        else
            echo "    ❌ ERROR: $JSON_FILE not found cho Seed $seed!"
            echo "  Seed $seed: ERROR" >> $SUMMARY_FILE
        fi
    done
    echo "" >> $SUMMARY_FILE

done

echo "" >> $SUMMARY_FILE
echo "============================================================" >> $SUMMARY_FILE
echo "  HOÀN TẤT: $(date)" >> $SUMMARY_FILE
echo "============================================================" >> $SUMMARY_FILE

echo ""
echo "=========================================================="
echo "  🎉 THỰC NGHIỆM HOÀN TẤT!"
echo "  Kết quả tóm tắt: $SUMMARY_FILE"
echo "  Chạy 'python tools/aggregate_seeds.py' để tính Mean ± Std"
echo "=========================================================="
