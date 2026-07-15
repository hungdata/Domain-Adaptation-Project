#!/usr/bin/env python3
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

matplotlib.use("Agg")

def main():
    csv_path = "results/method_comparison.csv"
    if not os.path.exists(csv_path):
        print(f"Không tìm thấy {csv_path}. Vui lòng chạy lệnh 'python tools/compare_methods.py' trước để tạo bảng kết quả.")
        return

    methods = []
    target_accs = []
    
    # Đọc dữ liệu từ CSV
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Tìm index của cột Target Acc Mean
        try:
            target_acc_idx = header.index("Target Acc (%) Mean")
        except ValueError:
            print("Không tìm thấy cột 'Target Acc (%) Mean' trong file CSV.")
            return
            
        for row in reader:
            method = row[0]
            acc_str = row[target_acc_idx]
            if acc_str == "N/A":
                continue # Bỏ qua các method chưa chạy xong
            methods.append(method)
            target_accs.append(float(acc_str))

    if not methods:
        print("Không có đủ dữ liệu hợp lệ để vẽ.")
        return

    # Sắp xếp theo thứ tự quen thuộc (tùy chọn) hoặc giữ nguyên từ CSV
    # Ở đây ta giữ nguyên vì compare_methods đã xếp theo METHOD_ORDER
    
    # Thiết lập biểu đồ
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Tô màu: CDANC nổi bật, các baseline màu khác
    colors = []
    for m in methods:
        if m == "CDANC":
            colors.append("#e74c3c") # Đỏ nổi bật
        elif m.startswith("DSAN") or m.startswith("CDAN"):
            colors.append("#3498db") # Xanh lam
        else:
            colors.append("#95a5a6") # Xám (ERM, DANN)
            
    bars = ax.bar(methods, target_accs, color=colors, edgecolor='black', linewidth=1.2)
    
    # Hiển thị số liệu trên đỉnh mỗi cột
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}%", 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
                
    # Giới hạn trục Y (cắt bớt phần dưới để sự chênh lệch rõ ràng hơn)
    min_acc = min(target_accs)
    max_acc = max(target_accs)
    ax.set_ylim(max(0, min_acc - 5), min(100, max_acc + 5))
    
    # Trang trí
    ax.set_ylabel("Target Accuracy (%)", fontsize=12, fontweight='bold')
    ax.set_title("Ablation Study: Target Accuracy Comparison", fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Đường kẻ ngang tham chiếu (kẻ ngang từ đỉnh ERM và CDANC)
    if "ERM" in methods:
        erm_val = target_accs[methods.index("ERM")]
        ax.axhline(erm_val, color='gray', linestyle=':', alpha=0.8)
    if "CDANC" in methods:
        cdanc_val = target_accs[methods.index("CDANC")]
        ax.axhline(cdanc_val, color='#e74c3c', linestyle=':', alpha=0.5)

    plt.xticks(rotation=30, ha="right", fontsize=11)
    plt.tight_layout()
    
    out_path = "results/ablation_bar_chart.png"
    plt.savefig(out_path, dpi=300)
    print(f"✅ Đã vẽ thành công Biểu đồ Cột Ablation Study tại: {out_path}")

if __name__ == "__main__":
    main()
