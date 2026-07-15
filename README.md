# Fruit Domain Adaptation

Dự án này triển khai 8 phương pháp Domain Adaptation để nhận diện trái cây Fresh/Rotten, tập trung vào Unsupervised Domain Adaptation (UDA).

## 8 Phương pháp (Methods)
1. **ERM**: Empirical Risk Minimization (Source Only Baseline)
2. **DANN**: Domain-Adversarial Training of Neural Networks
3. **CDAN**: Conditional Adversarial Domain Adaptation
4. **CDANE**: CDAN with Entropy Conditioning
5. **DSAN**: Deep Subdomain Adaptation Network
6. **DSANE**: DSAN with Entropy-weighted Pseudo-labels
7. **CDANE_MCC**: CDANE with Minimum Class Confusion
8. **DSANE_MCC**: DSANE with Minimum Class Confusion

## Cấu trúc thư mục
- `core/`: Các hàm tiện ích, DataLoader, Network Architecture (FeatureExtractor, Bottleneck, ClassifierHead)
- `methods/`: Chứa 8 script train cho 8 thuật toán
- `tools/`: Các công cụ đánh giá, visualize (t-SNE, PCA, GradCAM, Confusion Matrix), aggregate seeds
- `working/`: Nơi lưu trữ checkpoint, log CSV, log JSON (tự động tạo ra)
- `results/`: Nơi lưu ảnh visualize và bảng tổng hợp

## Cách chạy (How to run)

Chạy thực nghiệm chuẩn 8 methods × 3 seeds (2026, 2027, 2028):
```bash
bash run_3_seeds.sh
```
Kết quả model sẽ được lưu tại `working/seed_{seed}/{METHOD}_model/last_model.pth` và `best_source_val_model.pth`.

Chạy Visualize tự động:
```bash
python tools/auto_eval_all.py
```
Kết quả ảnh sẽ nằm ở `results/vis/`.
