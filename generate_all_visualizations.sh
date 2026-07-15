#!/bin/bash
# ==============================================================================
# Script tự động tạo TẤT CẢ biểu đồ (CM, t-SNE, PCA, Grad-CAM) cho TẤT CẢ methods
# ==============================================================================

set -euo pipefail

METHODS=("ERM" "DANN" "CDAN" "CDANE" "CDANC" "CDANE_MCC" "DSAN" "DSANE" "DSANE_MCC")
SEED="2026"
DATA_ROOT="uda_fixed_folders"

OUT_DIR="results/all_visualizations"
mkdir -p "$OUT_DIR"

echo "Bắt đầu khởi tạo bộ sưu tập đồ thị..."
echo "Thư mục lưu trữ: $OUT_DIR"

for method in "${METHODS[@]}"; do
    echo "========================================================="
    echo " Đang xử lý thuật toán: $method"
    echo "========================================================="
    
    # 1. Vẽ Confusion Matrix (siêu nhanh)
    echo "[1/4] Vẽ Confusion Matrix..."
    if [ -f "working/seed_${SEED}/${method}_logs/final_test_metrics.json" ]; then
        python tools/confusion_matrix_plot.py \
          --json "working/seed_${SEED}/${method}_logs/final_test_metrics.json" \
          --output "${OUT_DIR}/cm_${method}.png"
    else
        echo "  -> [SKIP] Không tìm thấy file JSON của $method"
    fi
      
    # 2. Vẽ t-SNE (mất khoảng 30-40s mỗi thuật toán)
    echo "[2/4] Vẽ t-SNE..."
    if [ -f "working/seed_${SEED}/${method}_model/last_model.pth" ]; then
        python tools/visualize_tsne.py \
          --model-path "working/seed_${SEED}/${method}_model/last_model.pth" \
          --output "${OUT_DIR}/tsne_${method}.png"
    else
        echo "  -> [SKIP] Không tìm thấy model của $method"
    fi
      
    # 3. Vẽ PCA
    echo "[3/4] Vẽ PCA..."
    if [ -f "working/seed_${SEED}/${method}_model/last_model.pth" ]; then
        python tools/visualize_pca.py \
          --model-path "working/seed_${SEED}/${method}_model/last_model.pth" \
          --output "${OUT_DIR}/pca_${method}.png"
    fi
      
    # 4. Vẽ Grad-CAM (Rotten & Fresh)
    echo "[4/4] Vẽ Grad-CAM..."
    if [ -f "working/seed_${SEED}/${method}_model/last_model.pth" ]; then
        python tools/visualize_gradcam.py \
          --model-path "working/seed_${SEED}/${method}_model/last_model.pth" \
          --image-dir "${DATA_ROOT}/target/test/rotten" \
          --output "${OUT_DIR}/gradcam_rotten_${method}.png" > /dev/null 2>&1
          
        python tools/visualize_gradcam.py \
          --model-path "working/seed_${SEED}/${method}_model/last_model.pth" \
          --image-dir "${DATA_ROOT}/target/test/fresh" \
          --output "${OUT_DIR}/gradcam_fresh_${method}.png" > /dev/null 2>&1
    fi
      
    echo "-> Hoàn thành $method!"
    echo ""
done

echo "========================================================="
echo " 🎉 TẤT CẢ ĐÃ XONG! TOÀN BỘ ẢNH ĐÃ ĐƯỢC LƯU TẠI: $OUT_DIR"
echo "========================================================="
