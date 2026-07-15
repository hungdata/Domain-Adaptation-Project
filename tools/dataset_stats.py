import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def count_images_recursive(dir_path):
    count = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                count += 1
    return count

def count_classes(dir_path):
    if not os.path.exists(dir_path):
        return {}
    
    counts = {}
    for class_name in ["fresh", "rotten"]:
        class_path = os.path.join(dir_path, class_name)
        if os.path.exists(class_path):
            counts[class_name] = count_images_recursive(class_path)
        else:
            counts[class_name] = 0
    return counts

def count_unlabeled(dir_path):
    if not os.path.exists(dir_path):
        return 0
    return count_images_recursive(dir_path)

def main():
    global DATA_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./uda_fixed_folders")
    args = parser.parse_args()
    DATA_ROOT = args.data_root

    stats = {}
    stats["Source Train"] = count_classes(os.path.join(DATA_ROOT, "source", "train"))
    stats["Source Val"] = count_classes(os.path.join(DATA_ROOT, "source", "val"))
    stats["Source Test"] = count_classes(os.path.join(DATA_ROOT, "source", "test"))
    
    stats["Target Test"] = count_classes(os.path.join(DATA_ROOT, "target", "test"))
    
    # Target Train is unlabeled in UDA
    target_train_count = count_unlabeled(os.path.join(DATA_ROOT, "target", "train_unlabeled", "images"))
    stats["Target Train"] = {"unlabeled": target_train_count}
    
    print("=" * 50)
    print("THỐNG KÊ SỐ LƯỢNG ẢNH TRONG DATASET")
    print("=" * 50)
    
    total_source = sum(stats["Source Train"].values()) + sum(stats["Source Val"].values()) + sum(stats["Source Test"].values())
    total_target = sum(stats["Target Train"].values()) + sum(stats["Target Test"].values())
    
    print("[TỈ LỆ CHIA DATASET - SPLIT RATIOS]")
    if total_source > 0:
        print(f"Tổng tập Source: {total_source} ảnh")
        print(f"   - Train : {sum(stats['Source Train'].values())/total_source*100:.1f}%")
        print(f"   - Val   : {sum(stats['Source Val'].values())/total_source*100:.1f}%")
        print(f"   - Test  : {sum(stats['Source Test'].values())/total_source*100:.1f}%")
    if total_target > 0:
        print(f"Tổng tập Target: {total_target} ảnh")
        print(f"   - Train (Unlabeled) : {sum(stats['Target Train'].values())/total_target*100:.1f}%")
        print(f"   - Test  (Labeled)   : {sum(stats['Target Test'].values())/total_target*100:.1f}%")
    print("-" * 50)
    
    for split, counts in stats.items():
        total = sum(counts.values())
        print(f"[{split.upper()}] - Tổng cộng: {total} ảnh")
        if total > 0:
            for c_name, c_count in counts.items():
                percent = (c_count / total) * 100
                print(f"   - {c_name.capitalize()}: {c_count} ({percent:.2f}%)")
        print("-" * 50)
        
    # Vẽ biểu đồ Bar Chart
    splits = ["Source Train", "Source Val", "Source Test", "Target Test"]
    fresh_counts = [stats[s].get("fresh", 0) for s in splits]
    rotten_counts = [stats[s].get("rotten", 0) for s in splits]
    
    x = np.arange(len(splits))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, fresh_counts, width, label='Fresh', color='#2ecc71', edgecolor='black')
    rects2 = ax.bar(x + width/2, rotten_counts, width, label='Rotten', color='#e74c3c', edgecolor='black')
    
    # Hàm ghi label
    def autolabel(rects, total_list):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            if height > 0:
                total = total_list[i]
                percent = (height / total) * 100
                ax.annotate(f'{height}\n({percent:.1f}%)',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

    total_counts = [f + r for f, r in zip(fresh_counts, rotten_counts)]
    autolabel(rects1, total_counts)
    autolabel(rects2, total_counts)
    
    ax.set_ylabel('Số lượng ảnh (Images)', fontsize=12)
    ax.set_title('Thống kê phân bổ dữ liệu (Data Distribution) theo Class', fontsize=14, fontweight='bold', pad=20)
    
    # Xử lý riêng cột Target Train (vì nó không có label)
    if target_train_count > 0:
        splits_extended = splits + ["Target Train\n(Unlabeled)"]
        x_extended = np.arange(len(splits_extended))
        
        rect3 = ax.bar(x_extended[-1], target_train_count, width, label='Unlabeled', color='#95a5a6', edgecolor='black')
        
        ax.annotate(f'{target_train_count}\n(100.0%)',
                    xy=(rect3[0].get_x() + rect3[0].get_width() / 2, target_train_count),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(x_extended)
        ax.set_xticklabels(splits_extended, fontsize=11)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(splits, fontsize=11)
        
    ax.legend(fontsize=11)
    
    # Thêm đường kẻ grid cho dễ nhìn
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Mở rộng giới hạn Y một chút để text không bị cắt
    max_height = max(max(fresh_counts + rotten_counts), target_train_count)
    ax.set_ylim(0, max_height * 1.15)
    
    fig.tight_layout()
    
    output_path = "working/dataset_statistics.png"
    os.makedirs("working", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"\n[SUCCESS] Đã vẽ và lưu biểu đồ Bar Chart tại: {output_path}")

if __name__ == "__main__":
    main()
