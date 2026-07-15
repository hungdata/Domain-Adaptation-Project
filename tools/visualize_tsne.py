import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.uda_utils import (
    FeatureExtractor,
    Bottleneck,
    get_device,
    get_source_test_loader,
    get_target_test_loader
)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize t-SNE of Bottleneck Features")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the checkpoint (.pth)")
    parser.add_argument("--data-root", type=str, default="./uda_fixed_folders", help="Path to dataset")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples per class per domain")
    parser.add_argument("--output", type=str, default="tsne_bottleneck.png", help="Output image path")
    return parser.parse_args()

def extract_features(F_ext, bottleneck, loader, device, max_samples_per_class):
    features = []
    labels = []
    
    counts = {0: 0, 1: 0}
    
    F_ext.eval()
    bottleneck.eval()
    
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            lbls = lbls.cpu().numpy()
            
            # Forward
            feats = bottleneck(F_ext(imgs)).cpu().numpy()
            
            for i in range(len(lbls)):
                label = int(lbls[i])
                if counts.get(label, 0) < max_samples_per_class:
                    features.append(feats[i])
                    labels.append(label)
                    counts[label] = counts.get(label, 0) + 1
                    
            if counts.get(0, 0) >= max_samples_per_class and counts.get(1, 0) >= max_samples_per_class:
                break
                
    return np.array(features), np.array(labels)

def main():
    args = parse_args()
    device = get_device()
    
    print(f"Loading model from {args.model_path}...")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    
    F_ext = FeatureExtractor().to(device)
    bottleneck = Bottleneck().to(device)
    
    F_ext.load_state_dict(ckpt["model_state_dict"]["F_ext"])
    bottleneck.load_state_dict(ckpt["model_state_dict"]["bottleneck"])
    
    print("Loading data...")
    src_loader = get_source_test_loader(args.data_root, batch_size=32, num_workers=4)
    tgt_loader = get_target_test_loader(args.data_root, batch_size=32, num_workers=4)
    
    print(f"Extracting features (max {args.num_samples} per class)...")
    src_feats, src_labels = extract_features(F_ext, bottleneck, src_loader, device, args.num_samples)
    tgt_feats, tgt_labels = extract_features(F_ext, bottleneck, tgt_loader, device, args.num_samples)
    
    print(f"Source features: {src_feats.shape}, Target features: {tgt_feats.shape}")
    
    all_feats = np.concatenate([src_feats, tgt_feats], axis=0)
    all_labels = np.concatenate([src_labels, tgt_labels], axis=0)
    
    # Domain labels: 0 for Source, 1 for Target
    domains = np.concatenate([np.zeros(len(src_labels)), np.ones(len(tgt_labels))], axis=0)
    
    print("Running t-SNE (this might take a minute)...")
    tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=30, max_iter=2000)
    tsne_results = tsne.fit_transform(all_feats)
    
    print("Plotting...")
    plt.figure(figsize=(10, 8))
    
    # Source Fresh
    idx_src_fresh = (domains == 0) & (all_labels == 0)
    plt.scatter(tsne_results[idx_src_fresh, 0], tsne_results[idx_src_fresh, 1], 
                c='#32CD32', marker='o', alpha=0.7, label=f'Source Fresh ({sum(idx_src_fresh)})')
                
    # Source Rotten
    idx_src_rotten = (domains == 0) & (all_labels == 1)
    plt.scatter(tsne_results[idx_src_rotten, 0], tsne_results[idx_src_rotten, 1], 
                c='#FF4500', marker='o', alpha=0.7, label=f'Source Rotten ({sum(idx_src_rotten)})')
                
    # Target Fresh
    idx_tgt_fresh = (domains == 1) & (all_labels == 0)
    plt.scatter(tsne_results[idx_tgt_fresh, 0], tsne_results[idx_tgt_fresh, 1], 
                c='#00FF00', marker='^', alpha=0.7, label=f'Target Fresh ({sum(idx_tgt_fresh)})')
                
    # Target Rotten
    idx_tgt_rotten = (domains == 1) & (all_labels == 1)
    plt.scatter(tsne_results[idx_tgt_rotten, 0], tsne_results[idx_tgt_rotten, 1], 
                c='#DC143C', marker='^', alpha=0.7, label=f'Target Rotten ({sum(idx_tgt_rotten)})')
                
    plt.title(f"t-SNE on Bottleneck Features ({all_feats.shape[1]}-dim)")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved plot to {args.output}")

if __name__ == '__main__':
    main()
