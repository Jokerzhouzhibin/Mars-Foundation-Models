"""
注意力可视化定量评估
====================
对所有 6 个模型 × 候选图 × 2 种方法，从三个维度定量评估注意力图质量：

1. 统计指标 (Statistics):
   - peak, mean, contrast(std), sparsity, top5_ratio, norm_entropy

2. 语义对齐 (Semantic Alignment):
   - edge_corr: 注意力与边缘图的 Pearson 相关系数
   - high_capture / low_leakage / discriminability

3. 空间连贯性 (Spatial Coherence):
   - largest_blob_ratio, n_blobs, compactness, spatial_variance
   - combined_score = edge_corr × largest_blob_ratio

用法:
    python eval_attention.py                # 运行全部三项评估
    python eval_attention.py --mode stats   # 仅统计指标
    python eval_attention.py --mode semantic # 仅语义对齐
    python eval_attention.py --mode coherence # 仅空间连贯性
"""
import argparse
import gc
import glob
import os

import cv2
import numpy as np
import torch
from scipy.ndimage import label as nd_label
from scipy.stats import entropy as sp_entropy
from scipy.stats import pearsonr

from loader import load_model
from visualize import (
    method_patch_cosine, method_mid_layer, preprocess_image, _get_patch_size
)


# ==========================================
# 共享配置
# ==========================================
ALL_MODELS = ["dinov3_lvd", "dinov3_lvd_1cha", "dinov3_sat", "dinov3_sat_1cha", "mars_mae", "dinov1_b8"]

REF_POINTS = {
    "label11_sample0.png": (4, 13),
    "label13_sample.png":  (8, 12),
    "label13_sample0.png": (8, 6),
    "original_0.png":      (7, 7),
}

IMG_DIR = "./candidate_images/domars16k_shared"


def _get_ref_rel_pos(basename):
    """获取参考点的归一化坐标。"""
    if basename in REF_POINTS:
        row, col = REF_POINTS[basename]
        return (row / 16.0, col / 16.0)
    return (0.5, 0.5)


def _compute_edge_grid(img_path, res, grid_h, grid_w):
    """计算图像边缘强度并下采样到 patch 网格。"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (res, res))
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    cell_h, cell_w = res // grid_h, res // grid_w
    grid = np.zeros((grid_h, grid_w))
    for r in range(grid_h):
        for c in range(grid_w):
            grid[r, c] = mag[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w].mean()
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    return grid


# ==========================================
# 评估 1: 统计指标
# ==========================================
def compute_heatmap_metrics(hmap):
    """计算热力图的统计指标。"""
    flat = hmap.flatten()
    mean = flat.mean()
    std = flat.std()
    peak = flat.max()

    threshold = mean + std
    sparsity = (flat > threshold).sum() / len(flat)

    k = max(1, int(len(flat) * 0.05))
    top_k = np.sort(flat)[-k:]
    top5_ratio = top_k.sum() / (flat.sum() + 1e-10)

    p = flat / (flat.sum() + 1e-10)
    ent = sp_entropy(p)
    max_ent = np.log(len(p))
    norm_entropy = ent / (max_ent + 1e-10)

    return {
        'peak': peak,
        'mean': mean,
        'contrast': std,
        'sparsity': sparsity,
        'top5_ratio': top5_ratio,
        'norm_entropy': norm_entropy,
    }


def run_stats(img_paths, device):
    """运行统计指标评估。"""
    methods = {'patch_cosine': 'Patch Cosine Similarity', 'mid_layer_cls': 'Mid-Layer CLS Attention'}
    results = {m: {name: [] for name in ALL_MODELS} for m in methods}

    for model_name in ALL_MODELS:
        print(f"\n>>> [Stats] Processing {model_name}")
        model = load_model(model_name, device)
        ps = _get_patch_size(model_name)

        for img_path in img_paths:
            basename = os.path.basename(img_path)
            tensor, _ = preprocess_image(img_path, model_name)
            ref_rel_pos = _get_ref_rel_pos(basename)

            hmap_cos, _ = method_patch_cosine(model, tensor, patch_size=ps, ref_rel_pos=ref_rel_pos)
            results['patch_cosine'][model_name].append(compute_heatmap_metrics(hmap_cos))

            hmap_mid = method_mid_layer(model, tensor, patch_size=ps)
            results['mid_layer_cls'][model_name].append(compute_heatmap_metrics(hmap_mid))

        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 打印
    for method_key, method_name in methods.items():
        print(f"\n{'='*80}")
        print(f"  {method_name} -- Statistics")
        print(f"{'='*80}")
        print(f"{'Model':22s} | {'Peak':>6s} | {'Mean':>6s} | {'Contrast':>8s} | {'Sparsity':>8s} | {'Top5%':>6s} | {'Entropy':>7s}")
        print("-" * 80)

        model_scores = {}
        for model_name in ALL_MODELS:
            metrics_list = results[method_key][model_name]
            avg = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
            model_scores[model_name] = avg
            print(f"{model_name:22s} | {avg['peak']:6.4f} | {avg['mean']:6.4f} | "
                  f"{avg['contrast']:8.4f} | {avg['sparsity']:8.4f} | "
                  f"{avg['top5_ratio']:6.4f} | {avg['norm_entropy']:7.4f}")

        print(f"\n  Ranking by contrast (higher = more discriminative attention):")
        for i, (name, sc) in enumerate(sorted(model_scores.items(), key=lambda x: x[1]['contrast'], reverse=True)):
            print(f"    {i+1}. {name:22s}  contrast={sc['contrast']:.4f}  top5={sc['top5_ratio']:.4f}")

        print(f"\n  Ranking by top5_ratio (higher = more focused attention):")
        for i, (name, sc) in enumerate(sorted(model_scores.items(), key=lambda x: x[1]['top5_ratio'], reverse=True)):
            print(f"    {i+1}. {name:22s}  top5={sc['top5_ratio']:.4f}  entropy={sc['norm_entropy']:.4f}")


# ==========================================
# 评估 2: 语义对齐
# ==========================================
def compute_semantic_metrics(heatmap, edge_grid):
    """计算语义对齐指标。"""
    h_flat = heatmap.flatten()
    e_flat = edge_grid.flatten()

    if h_flat.std() < 1e-8 or e_flat.std() < 1e-8:
        corr = 0.0
    else:
        corr, _ = pearsonr(h_flat, e_flat)

    threshold_high = np.percentile(e_flat, 80)
    threshold_low = np.percentile(e_flat, 20)
    high_grad_mask = e_flat >= threshold_high
    low_grad_mask = e_flat <= threshold_low

    high_grad_capture = h_flat[high_grad_mask].mean() if high_grad_mask.sum() > 0 else 0
    low_grad_leakage = h_flat[low_grad_mask].mean() if low_grad_mask.sum() > 0 else 0
    discriminability = high_grad_capture / (low_grad_leakage + 1e-8)

    return {
        'edge_corr': corr,
        'high_capture': high_grad_capture,
        'low_leakage': low_grad_leakage,
        'discriminability': discriminability,
    }


def run_semantic(img_paths, device):
    """运行语义对齐评估。"""
    results_cos = {name: [] for name in ALL_MODELS}
    results_mid = {name: [] for name in ALL_MODELS}

    for model_name in ALL_MODELS:
        print(f"\n>>> [Semantic] Processing {model_name}")
        model = load_model(model_name, device)
        ps = _get_patch_size(model_name)

        for img_path in img_paths:
            basename = os.path.basename(img_path)
            tensor, _ = preprocess_image(img_path, model_name)
            res = tensor.shape[2]
            grid_h, grid_w = res // ps, res // ps
            edge_grid = _compute_edge_grid(img_path, res, grid_h, grid_w)
            ref_rel_pos = _get_ref_rel_pos(basename)

            hmap_cos, _ = method_patch_cosine(model, tensor, patch_size=ps, ref_rel_pos=ref_rel_pos)
            results_cos[model_name].append(compute_semantic_metrics(hmap_cos, edge_grid))

            hmap_mid = method_mid_layer(model, tensor, patch_size=ps)
            results_mid[model_name].append(compute_semantic_metrics(hmap_mid, edge_grid))

        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    for method_name, results in [("Patch Cosine Similarity", results_cos),
                                  ("Mid-Layer CLS Attention", results_mid)]:
        print(f"\n{'='*85}")
        print(f"  {method_name} -- Semantic Alignment")
        print(f"{'='*85}")
        print(f"{'Model':22s} | {'EdgeCorr':>8s} | {'HiCapture':>9s} | {'LoLeak':>7s} | {'Discrim':>7s}")
        print("-" * 85)

        scores = {}
        for model_name in ALL_MODELS:
            metrics_list = results[model_name]
            avg = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
            scores[model_name] = avg
            print(f"{model_name:22s} | {avg['edge_corr']:8.4f} | {avg['high_capture']:9.4f} | "
                  f"{avg['low_leakage']:7.4f} | {avg['discriminability']:7.2f}")

        print(f"\n  Ranking by edge_corr:")
        for i, (name, sc) in enumerate(sorted(scores.items(), key=lambda x: x[1]['edge_corr'], reverse=True)):
            print(f"    {i+1}. {name:22s}  corr={sc['edge_corr']:.4f}  discrim={sc['discriminability']:.2f}")

        print(f"\n  Ranking by discriminability:")
        for i, (name, sc) in enumerate(sorted(scores.items(), key=lambda x: x[1]['discriminability'], reverse=True)):
            print(f"    {i+1}. {name:22s}  discrim={sc['discriminability']:.2f}  corr={sc['edge_corr']:.4f}")


# ==========================================
# 评估 3: 空间连贯性
# ==========================================
def compute_coherence_metrics(heatmap, edge_grid):
    """计算空间连贯性指标。"""
    h, w = heatmap.shape

    hmap_uint8 = (heatmap * 255).astype(np.uint8)
    _, binary = cv2.threshold(hmap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_bool = binary > 0

    labeled, n_blobs = nd_label(binary_bool)
    total_highlight = binary_bool.sum()

    if n_blobs == 0 or total_highlight == 0:
        return {
            'largest_blob_ratio': 0, 'n_blobs': 0, 'compactness': 0,
            'spatial_variance': 1.0, 'edge_corr': 0, 'combined_score': 0,
        }

    blob_sizes = [(labeled == i).sum() for i in range(1, n_blobs + 1)]
    largest_size = max(blob_sizes)
    largest_blob_ratio = largest_size / total_highlight

    largest_id = blob_sizes.index(largest_size) + 1
    largest_mask = (labeled == largest_id).astype(np.uint8)
    largest_up = cv2.resize(largest_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(largest_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        blob_area = cv2.contourArea(contours[0])
        compactness = blob_area / (hull_area + 1e-8)
    else:
        compactness = 0

    ys, xs = np.where(binary_bool)
    spatial_variance = (ys / h).var() + (xs / w).var()

    h_flat = heatmap.flatten()
    e_flat = edge_grid.flatten()
    if h_flat.std() < 1e-8 or e_flat.std() < 1e-8:
        edge_corr = 0.0
    else:
        edge_corr, _ = pearsonr(h_flat, e_flat)

    combined_score = edge_corr * largest_blob_ratio

    return {
        'largest_blob_ratio': largest_blob_ratio,
        'n_blobs': n_blobs,
        'compactness': compactness,
        'spatial_variance': spatial_variance,
        'edge_corr': edge_corr,
        'combined_score': combined_score,
    }


def run_coherence(img_paths, device):
    """运行空间连贯性评估。"""
    results_cos = {name: [] for name in ALL_MODELS}
    results_mid = {name: [] for name in ALL_MODELS}

    for model_name in ALL_MODELS:
        print(f"\n>>> [Coherence] Processing {model_name}")
        model = load_model(model_name, device)
        ps = _get_patch_size(model_name)

        for img_path in img_paths:
            basename = os.path.basename(img_path)
            tensor, _ = preprocess_image(img_path, model_name)
            res = tensor.shape[2]
            grid_h, grid_w = res // ps, res // ps
            edge_grid = _compute_edge_grid(img_path, res, grid_h, grid_w)
            ref_rel_pos = _get_ref_rel_pos(basename)

            hmap_cos, _ = method_patch_cosine(model, tensor, patch_size=ps, ref_rel_pos=ref_rel_pos)
            results_cos[model_name].append(compute_coherence_metrics(hmap_cos, edge_grid))

            hmap_mid = method_mid_layer(model, tensor, patch_size=ps)
            results_mid[model_name].append(compute_coherence_metrics(hmap_mid, edge_grid))

        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    for method_name, results in [("Patch Cosine Similarity", results_cos),
                                  ("Mid-Layer CLS Attention", results_mid)]:
        print(f"\n{'='*95}")
        print(f"  {method_name} -- Spatial Coherence & Semantic Alignment")
        print(f"{'='*95}")
        print(f"{'Model':22s} | {'BlobRatio':>9s} | {'#Blobs':>6s} | {'Compact':>7s} | "
              f"{'SpatVar':>7s} | {'EdgeCorr':>8s} | {'Combined':>8s}")
        print("-" * 95)

        scores = {}
        for model_name in ALL_MODELS:
            ml = results[model_name]
            avg = {key: np.mean([m[key] for m in ml]) for key in ml[0]}
            scores[model_name] = avg
            print(f"{model_name:22s} | {avg['largest_blob_ratio']:9.4f} | {avg['n_blobs']:6.1f} | "
                  f"{avg['compactness']:7.4f} | {avg['spatial_variance']:7.4f} | "
                  f"{avg['edge_corr']:8.4f} | {avg['combined_score']:8.4f}")

        print(f"\n  Ranking by combined_score (edge_corr * blob_ratio):")
        for i, (name, sc) in enumerate(sorted(scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)):
            print(f"    {i+1}. {name:22s}  combined={sc['combined_score']:.4f}  "
                  f"blob={sc['largest_blob_ratio']:.4f}  edge={sc['edge_corr']:.4f}  "
                  f"blobs={sc['n_blobs']:.1f}")


# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="注意力可视化定量评估")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "stats", "semantic", "coherence"],
                        help="评估模式: all=全部, stats=统计指标, semantic=语义对齐, coherence=空间连贯性")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))

    if not img_paths:
        print(f"[Error] No images found in {IMG_DIR}")
        exit(1)

    print(f"[System] Found {len(img_paths)} images in {IMG_DIR}")
    print(f"[System] Models: {ALL_MODELS}")
    print(f"[System] Mode: {args.mode}\n")

    if args.mode in ("all", "stats"):
        run_stats(img_paths, device)

    if args.mode in ("all", "semantic"):
        run_semantic(img_paths, device)

    if args.mode in ("all", "coherence"):
        run_coherence(img_paths, device)

    print("\n✅ Evaluation complete.")
