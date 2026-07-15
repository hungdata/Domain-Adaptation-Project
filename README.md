# 🍎 Fruit Quality Domain Adaptation (Unsupervised Domain Adaptation)

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📌 Vấn đề cốt lõi (The Problem)
Trong thực tế, một mô hình AI phân loại trái cây (tươi/hỏng - Fresh/Rotten) được huấn luyện trên một tập dữ liệu ảnh gốc (Source Domain - ví dụ: ảnh chụp trong điều kiện ánh sáng chuẩn ở phòng thí nghiệm, phông nền sạch) thường bị suy giảm hiệu suất nghiêm trọng khi đem ra áp dụng ở môi trường thực tế (Target Domain - ví dụ: ảnh chụp tại siêu thị, ánh sáng phức tạp, camera thiết bị di động). Nguyên nhân của sự suy giảm này là do hiện tượng **Domain Shift** (Sự chênh lệch phân phối dữ liệu giữa hai môi trường).

**Mục tiêu của dự án này:** Áp dụng các kỹ thuật **Unsupervised Domain Adaptation (UDA)** để giúp mô hình có thể học và tự động thích nghi với môi trường thực tế (Target Domain) mà **không cần bất kỳ dữ liệu có nhãn (labels) nào** từ môi trường mới này. Chúng ta chỉ sử dụng dữ liệu có nhãn từ môi trường gốc (Source Domain) và dữ liệu không nhãn từ Target Domain.

## 📖 Nguồn tham khảo (Reference & Inspiration)
Codebase này được xây dựng theo kiến trúc module hóa chuẩn mực của các framework Domain Adaptation hàng đầu hiện nay (lấy cảm hứng từ [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library)), kết hợp với việc tối ưu hóa cho bài toán nhận diện chất lượng trái cây trên thiết bị di động bằng MobileNetV3. Các thuật toán được triển khai bám sát với lý thuyết từ các bài báo khoa học gốc (như DANN của Ganin et al., CDAN của Long et al.).

## 🚀 Các phương pháp được triển khai (Implemented Methods)
Dự án cung cấp 8 phương pháp từ cơ bản đến nâng cao:
1. **ERM (Baseline)**: Empirical Risk Minimization (Chỉ huấn luyện trên Source Domain, không thích nghi).
2. **DANN**: Domain-Adversarial Training of Neural Networks.
3. **CDAN**: Conditional Adversarial Domain Adaptation.
4. **CDANE**: CDAN with Entropy Conditioning.
5. **DSAN**: Deep Subdomain Adaptation Network.
6. **DSANE**: DSAN with Entropy-weighted Pseudo-labels.
7. **CDANE_MCC**: CDANE with Minimum Class Confusion.
8. **DSANE_MCC**: DSANE with Minimum Class Confusion.

## 📊 Kết quả thực nghiệm tiêu biểu (Highlight Results)
Dưới đây là một ví dụ minh chứng cho hiệu quả của Domain Adaptation trên tập dữ liệu này (kết quả trích xuất thực tế từ Seed 2026):

| Phương pháp | Source Test Accuracy | Target Test Accuracy | Domain Gap | Cải thiện trên Target |
|:---:|:---:|:---:|:---:|:---:|
| **ERM (Baseline)** | 99.91% | 81.12% | 18.79% | - |
| **DANN (UDA)** | 99.72% | **88.05%** | **11.67%** | **+ 6.93%** |

*Nhận xét:* So với mô hình gốc (ERM) chỉ đạt 81.12% trên tập thực tế, việc áp dụng phương pháp UDA (DANN) để đồng bộ hóa đặc trưng (feature alignment) giữa hai miền dữ liệu đã giúp độ chính xác trên Target Domain **tăng xấp xỉ 7%** (lên 88.05%). Đồng thời, mô hình đã thu hẹp đáng kể khoảng cách rủi ro miền (Domain Gap từ 18.79% xuống còn 11.67%).

## 📂 Cấu trúc thư mục (Repository Structure)
- `core/`: Chứa các thành phần cốt lõi (DataLoader, Utils, Network Architecture bao gồm Feature Extractor, Bottleneck, ClassifierHead).
- `methods/`: Các kịch bản huấn luyện (training scripts) độc lập cho từng thuật toán.
- `tools/`: Các công cụ đánh giá và trực quan hóa mạnh mẽ (t-SNE, PCA, GradCAM, Confusion Matrix, tự động tổng hợp kết quả nhiều seeds).
- `configs/`: Các file cấu hình YAML (Hyperparameters) quản lý linh hoạt cho từng phương pháp.
- `working/`: Nơi tự động lưu trữ logs quá trình huấn luyện, metrics (JSON/CSV) và checkpoints.
- `results/`: Nơi xuất và lưu trữ các biểu đồ, hình ảnh trực quan hóa cuối cùng.

## 💻 Hướng dẫn sử dụng (How to Run)

**1. Cài đặt môi trường:**
```bash
pip install -r requirements.txt
```

**2. Huấn luyện toàn bộ mô hình (Chạy thực nghiệm với 3 Seeds chuẩn: 2026, 2027, 2028):**
```bash
bash run_3_seeds.sh
```
Kết quả model và log sẽ được lưu tự động tại `working/seed_{seed}/{METHOD}_model/` và `working/seed_{seed}/{METHOD}_logs/`.

**3. Đánh giá và trực quan hóa tự động:**
Sau khi quá trình huấn luyện hoàn tất, bạn có thể tự động tạo tất cả các biểu đồ đánh giá (so sánh Acc/F1, Confusion Matrix, t-SNE) bằng lệnh:
```bash
bash generate_all_visualizations.sh
```
Hoặc chạy trực tiếp qua Python:
```bash
python tools/auto_eval_all.py
```
Toàn bộ biểu đồ báo cáo sẽ được lưu tại thư mục `results/vis/`.
