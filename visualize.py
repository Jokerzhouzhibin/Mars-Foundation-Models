"""
Attention & Feature 可视化脚本
==============================
两种可视化方法：
  1. Patch Cosine Similarity  — Patch Token 间余弦相似度热力图
  2. Mid-Layer CLS Attention  — 中间层 CLS->patch 注意力权重

三个对照组（共用同一组图，均来自 domars16k）：
  A: DINOv3 LVD (3ch) vs DINOv3 LVD (1ch)   — lvd 的 1-channel 聚焦效应
  B: DINOv3 SAT (3ch) vs DINOv3 SAT (1ch)   — sat 的 1-channel 聚焦效应（与 A 对称）
  C: Mars-MAE vs DINOv1 (ViT-B/8)           — 同参数量级的域内 vs 通用对比
"""
import os
import glob
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from config_mars import MODEL_CONFIGS
from loader import load_model


# ==========================================
# 1. 多层 QKV Hook 引擎
# ==========================================
class MultiLayerQKVHook:
    """Hook 指定 block 的 attn.qkv 层，截获 QKV 输出后手动计算 attention weights。"""

    def __init__(self, model, layer_indices=None):
        self.raw_model = model.model if hasattr(model, 'model') else model
        self.qkv_outputs = {}
        self.num_heads = None
        self.hooks = []

        qkv_layers = []
        for name, module in self.raw_model.named_modules():
            if name.endswith('.attn.qkv') and isinstance(module, nn.Linear):
                qkv_layers.append((name, module))
                if self.num_heads is None:
                    attn_name = name.rsplit('.qkv', 1)[0]
                    for n2, m2 in self.raw_model.named_modules():
                        if n2 == attn_name and hasattr(m2, 'num_heads'):
                            self.num_heads = m2.num_heads
                            break

        total = len(qkv_layers)
        if total == 0:
            return

        if layer_indices is None:
            layer_indices = [total - 1]

        for idx in layer_indices:
            if 0 <= idx < total:
                name, module = qkv_layers[idx]
                handle = module.register_forward_hook(self._make_hook(idx))
                self.hooks.append(handle)

    def _make_hook(self, idx):
        def fn(module, input, output):
            self.qkv_outputs[idx] = output.detach().cpu()
        return fn

    def compute_attention(self, idx):
        qkv = self.qkv_outputs[idx]
        B, N, dim3 = qkv.shape
        C = dim3 // 3
        head_dim = C // self.num_heads
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim)
        q, k, _ = qkv.unbind(2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        return attn.softmax(dim=-1)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.qkv_outputs.clear()


# ==========================================
# 2. 辅助函数
# ==========================================
def _get_model_info(model):
    raw = model.model if hasattr(model, 'model') else model
    n_extra = getattr(raw, 'n_storage_tokens', 0) or getattr(raw, 'num_register_tokens', 0)
    n_prefix = 1 + n_extra
    n_blocks = sum(1 for name, _ in raw.named_modules() if name.endswith('.attn.qkv'))
    return raw, n_prefix, n_blocks


def _normalize_heatmap(arr):
    arr = arr.astype(np.float32)
    normalized = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    # 自适应 gamma 增强：当热力图稀疏（均值低）时，用较小的 gamma 提升可见性
    mean_val = normalized.mean()
    if mean_val < 0.25:
        gamma = max(0.3, mean_val / 0.25)
        normalized = np.power(normalized, gamma)
    return normalized


def _get_patch_size(model_name):
    """根据模型名称返回对应的 patch_size。"""
    if model_name == "dinov1_b8":
        return 8
    return 16


def _get_qkv(hook: MultiLayerQKVHook, idx: int):
    """
    从 MultiLayerQKVHook 中提取 Q, K, V 张量。
    返回 (Q, K, V)，各 shape (B, num_heads, N, head_dim)。
    """
    qkv = hook.qkv_outputs[idx]
    B, N, dim3 = qkv.shape
    C = dim3 // 3
    head_dim = C // hook.num_heads
    qkv = qkv.reshape(B, N, 3, hook.num_heads, head_dim)
    q, k, v = qkv.unbind(2)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    return q, k, v


# ==========================================
# 3. 核心可视化方法（函数式接口）
# ==========================================
def method_patch_cosine(model, img_tensor, patch_size=16, ref_rel_pos=None):
    """
    Patch Cosine Similarity:
    取 forward_features 的 patch token 特征，计算与参考 patch 的余弦相似度。

    Args:
        ref_rel_pos: (row_ratio, col_ratio) 归一化参考点坐标。
                     None 时自动选择注意力峰值 patch。

    Returns:
        heatmap: (h, w) 归一化热力图
        chosen_rel_pos: (row_ratio, col_ratio) 实际使用的参考点相对坐标
    """
    raw, n_prefix, n_blocks = _get_model_info(model)
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    model.eval()
    with torch.no_grad():
        if hasattr(raw, 'forward_features'):
            output = raw.forward_features(img_tensor)
        else:
            output = raw(img_tensor)

    # 提取 patch tokens
    if isinstance(output, dict):
        patch_tokens = output.get('x_norm_patchtokens', None)
        if patch_tokens is None:
            for v in output.values():
                if isinstance(v, torch.Tensor) and v.ndim == 3:
                    patch_tokens = v[:, n_prefix:]
                    break
    elif isinstance(output, (tuple, list)):
        if len(output) >= 2 and output[1].ndim == 3:
            patch_tokens = output[1]
        elif output[0].ndim == 3:
            patch_tokens = output[0][:, n_prefix:]
        else:
            patch_tokens = None
    elif isinstance(output, torch.Tensor) and output.ndim == 3:
        patch_tokens = output[:, n_prefix:]
    else:
        patch_tokens = None

    h = img_tensor.shape[-2] // patch_size
    w = img_tensor.shape[-1] // patch_size
    n_patches = h * w

    if patch_tokens is None:
        return np.zeros((h, w), dtype=np.float32), (0.5, 0.5)

    feats = patch_tokens[0][:n_patches]

    if ref_rel_pos is not None:
        ref_row = min(int(ref_rel_pos[0] * h), h - 1)
        ref_col = min(int(ref_rel_pos[1] * w), w - 1)
        ref_idx = ref_row * w + ref_col
    else:
        hook = MultiLayerQKVHook(model, layer_indices=[n_blocks - 1])
        with torch.no_grad():
            _ = raw(img_tensor)
        attn = hook.compute_attention(n_blocks - 1)[0]
        hook.remove()
        cls_attn = attn[:, 0, n_prefix:].mean(dim=0).numpy()[:n_patches]
        ref_idx = int(np.argmax(cls_attn))

    ref_row = ref_idx // w
    ref_col = ref_idx % w
    chosen_rel_pos = (ref_row / h, ref_col / w)

    ref_feat = feats[ref_idx]
    feats_norm = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
    ref_norm = ref_feat / (ref_feat.norm() + 1e-8)
    cos_sim = (feats_norm @ ref_norm).cpu().numpy()

    return _normalize_heatmap(cos_sim.reshape(h, w)), chosen_rel_pos


def method_mid_layer(model, img_tensor, patch_size=16):
    """
    Mid-Layer CLS Attention:
    约 3/4 深度处的 CLS→patch 注意力权重。返回 (h, w) heatmap。
    """
    raw, n_prefix, n_blocks = _get_model_info(model)
    mid_idx = max(0, n_blocks - n_blocks // 4 - 1)

    hook = MultiLayerQKVHook(model, layer_indices=[mid_idx])
    with torch.no_grad():
        _ = raw(img_tensor.to(next(model.parameters()).device))
    attn = hook.compute_attention(mid_idx)[0]
    hook.remove()

    cls_attn = attn[:, 0, n_prefix:].mean(dim=0).numpy()
    h = img_tensor.shape[-2] // patch_size
    w = img_tensor.shape[-1] // patch_size
    return _normalize_heatmap(cls_attn.reshape(h, w))


# ==========================================
# 4. 图像预处理与叠加
# ==========================================
def preprocess_image(img_path, model_name):
    cfg = MODEL_CONFIGS[model_name]
    target_size = cfg['res']
    mean, std = list(cfg['mean']), list(cfg['std'])
    is_1cha = cfg.get('in_chans', 3) == 1

    img_pil = Image.open(img_path)
    img_pil = img_pil.convert('L') if is_1cha else img_pil.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    original_numpy = np.array(img_pil.resize((target_size, target_size)))
    return transform(img_pil).unsqueeze(0), original_numpy


def overlay_heatmap(original_img, heatmap_2d):
    """将热力图叠加到原图上。支持灰度图自动转 RGB。"""
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    heatmap_resized = cv2.resize(heatmap_2d, (original_img.shape[1], original_img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(original_img, 0.5, heatmap_color, 0.5, 0)


# ==========================================
# 5. 图片提取
# ==========================================
def extract_images_by_labels(parquet_path, labels, output_dir, n_per_label=1):
    """从 parquet 中按 label 列表各随机取 n 张，每次运行结果不同。"""
    import pandas as pd
    import io
    import shutil

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(parquet_path))
    if not files:
        base_dir = os.path.dirname(parquet_path)
        base_name = os.path.basename(parquet_path)
        prefix = base_name.split("-")[0] if "-" in base_name else base_name.replace(".parquet", "")
        files = sorted(glob.glob(os.path.join(base_dir, f"{prefix}-*.parquet")))
    if not files:
        print(f"[Warning] 未找到: {parquet_path}")
        return []

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    saved_paths = []
    for label in labels:
        subset = df[df['label'] == label]
        sampled = subset.sample(n=min(n_per_label, len(subset)))
        for i, (_, row) in enumerate(sampled.iterrows()):
            img_data = row['image']
            byte_data = img_data['bytes'] if isinstance(img_data, dict) else img_data
            img = Image.open(io.BytesIO(byte_data))
            fname = f"label{label}_sample{i}.png"
            fpath = os.path.join(output_dir, fname)
            img.save(fpath)
            saved_paths.append(fpath)

    print(f"[*] 随机提取 {len(saved_paths)} 张图片 -> {output_dir}")
    return saved_paths


# ==========================================
# 6. 统一方法类接口
# ==========================================

@dataclass
class VizResult:
    """可视化方法的统一返回结构"""
    heatmap: Optional[np.ndarray] = None
    rgb_map: Optional[np.ndarray] = None
    matrix: Optional[np.ndarray] = None
    cluster_map: Optional[np.ndarray] = None
    multi_heatmaps: Optional[list] = None
    overlay_mode: str = "heatmap"


class VisualizationMethod(ABC):
    """所有可视化方法的基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def short_name(self) -> str:
        ...

    @abstractmethod
    def compute(self, model: torch.nn.Module, img_tensor: torch.Tensor, patch_size: int = 16) -> VizResult:
        ...


class PatchCosineSimilarity(VisualizationMethod):
    """封装 method_patch_cosine 为统一接口，所有模型使用手动指定的参考点。"""

    IMAGE_REF_POINTS = {
        "label11_sample0.png": (4, 13),
        "label13_sample.png":  (8, 12),
        "label13_sample0.png": (8, 6),
        "original_0.png":      (7, 7),
    }

    @property
    def name(self) -> str:
        return "Patch Cosine Sim"

    @property
    def short_name(self) -> str:
        return "patch_cosine"

    def reset_ref(self):
        pass

    def compute(self, model, img_tensor, patch_size=16, img_path=None):
        ref_rel_pos = (0.5, 0.5)
        if img_path is not None:
            basename = os.path.basename(img_path)
            if basename in self.IMAGE_REF_POINTS:
                row, col = self.IMAGE_REF_POINTS[basename]
                ref_rel_pos = (row / 16.0, col / 16.0)

        heatmap, _ = method_patch_cosine(model, img_tensor, patch_size=patch_size, ref_rel_pos=ref_rel_pos)
        return VizResult(heatmap=heatmap, overlay_mode="heatmap")


class MidLayerCLSAttention(VisualizationMethod):
    """封装 method_mid_layer 为统一接口。"""

    @property
    def name(self) -> str:
        return "Mid-Layer CLS Attn"

    @property
    def short_name(self) -> str:
        return "mid_layer_cls"

    def compute(self, model, img_tensor, patch_size=16):
        heatmap = method_mid_layer(model, img_tensor, patch_size=patch_size)
        return VizResult(heatmap=heatmap, overlay_mode="heatmap")


ALL_METHODS = [
    PatchCosineSimilarity(),
    MidLayerCLSAttention(),
]


# ==========================================
# 7. 对照组排版输出
# ==========================================

@dataclass
class ExperimentConfig:
    """对照组实验配置"""
    name: str
    title: str
    model_names: list
    display_labels: list


class ImageProvider:
    """图像提取与预处理的统一接口"""

    @staticmethod
    def extract_images(parquet_path, labels, output_dir, n_per_label=1):
        return extract_images_by_labels(parquet_path, labels, output_dir, n_per_label)

    @staticmethod
    def preprocess(img_path, model_name):
        return preprocess_image(img_path, model_name)


class ComparisonPlotter:
    """对照组排版输出"""

    def __init__(self, output_root="./all_checkpoints/output_vis"):
        self.output_root = output_root

    def _render_result(self, original_img, viz_result):
        """根据 VizResult 的 overlay_mode 渲染结果图像。"""
        if viz_result.overlay_mode == "heatmap" and viz_result.heatmap is not None:
            return overlay_heatmap(original_img, viz_result.heatmap)
        elif viz_result.overlay_mode == "rgb" and viz_result.rgb_map is not None:
            rgb = viz_result.rgb_map
            rgb_resized = cv2.resize(rgb, (original_img.shape[1], original_img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            return rgb_resized
        elif viz_result.overlay_mode == "matrix":
            if viz_result.cluster_map is not None:
                cmap = viz_result.cluster_map.astype(np.float32)
                cmap = (cmap - cmap.min()) / (cmap.max() - cmap.min() + 1e-8)
                cmap_resized = cv2.resize(cmap, (original_img.shape[1], original_img.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
                cmap_color = cv2.applyColorMap(np.uint8(255 * cmap_resized), cv2.COLORMAP_JET)
                return cmap_color
            return original_img
        elif viz_result.overlay_mode == "multi" and viz_result.multi_heatmaps is not None:
            if len(viz_result.multi_heatmaps) > 0:
                _, hmap = viz_result.multi_heatmaps[-1]
                return overlay_heatmap(original_img, hmap)
            return original_img
        return original_img

    def plot_single_method(self, experiment, method, img_paths, models, model_names, save_dir):
        """为单个方法生成对照组图表。"""
        n_images = len(img_paths)
        n_models = len(models)

        fig, axes = plt.subplots(n_images, 1 + n_models,
                                  figsize=(4 * (1 + n_models), 4 * n_images))
        if n_images == 1:
            axes = axes[np.newaxis, :]

        for row, img_path in enumerate(img_paths):
            if hasattr(method, 'reset_ref'):
                method.reset_ref()

            _, orig_np = preprocess_image(img_path, model_names[0])
            display_orig = orig_np if len(orig_np.shape) == 3 else cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)
            axes[row, 0].imshow(display_orig)
            axes[row, 0].axis('off')

            for col, (mdl, mn, dlabel) in enumerate(zip(models, model_names, experiment.display_labels)):
                tensor, orig = preprocess_image(img_path, mn)
                ps = _get_patch_size(mn)
                if hasattr(method, 'compute') and 'img_path' in method.compute.__code__.co_varnames:
                    result = method.compute(mdl, tensor, patch_size=ps, img_path=img_path)
                else:
                    result = method.compute(mdl, tensor, patch_size=ps)
                rendered = self._render_result(orig, result)

                if len(rendered.shape) == 3 and rendered.shape[2] == 3:
                    axes[row, 1 + col].imshow(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
                else:
                    axes[row, 1 + col].imshow(rendered, cmap='gray')
                axes[row, 1 + col].axis('off')

        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{method.short_name}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

    def plot_all_methods(self, experiment, methods, img_paths, models, model_names):
        """为所有方法生成对照组图表。"""
        save_dir = os.path.join(self.output_root, experiment.name)
        os.makedirs(save_dir, exist_ok=True)

        for method in methods:
            print(f"  Plotting {method.name}...", end=" ", flush=True)
            self.plot_single_method(experiment, method, img_paths, models, model_names, save_dir)
            print("done")


# ==========================================
# 8. 对照组配置
# ==========================================
EXPERIMENTS = [
    ExperimentConfig(
        name="ExpA_1ch_lvd",
        title="Exp A: 1-Channel Focus Effect (LVD)",
        model_names=["dinov3_lvd", "dinov3_lvd_1cha"],
        display_labels=["DINOv3 LVD (3ch)", "DINOv3 LVD (1ch)"],
    ),
    ExperimentConfig(
        name="ExpB_1ch_sat",
        title="Exp B: 1-Channel Focus Effect (SAT)",
        model_names=["dinov3_sat", "dinov3_sat_1cha"],
        display_labels=["DINOv3 SAT (3ch)", "DINOv3 SAT (1ch)"],
    ),
    ExperimentConfig(
        name="ExpC_Domain_vs_General",
        title="Exp C: Mars-MAE vs DINOv1 (Same-scale Comparison)",
        model_names=["mars_mae", "dinov1_b8"],
        display_labels=["Mars-MAE (ViT-B, 86M)", "DINOv1 (ViT-B/8, 86M)"],
    ),
]


# ==========================================
# 9. 主程序
# ==========================================
def main(extract_new=False):
    """
    主程序。
    Args:
        extract_new: 是否从 parquet 重新随机提取图片。
                     False (默认) = 直接使用 candidate_images 目录下已有的图片。
                     True = 从 parquet 随机提取新图片（会覆盖已有图片）。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vis_root = "./all_checkpoints/output_vis"
    os.makedirs(vis_root, exist_ok=True)

    # ---- 图片准备 ----
    img_dir = "./candidate_images/domars16k_shared"

    if extract_new:
        selected_labels = [13, 3, 7, 11, 8]
        extract_images_by_labels(
            "dataset/mb-domars16k/data/test-00000-of-00001.parquet",
            selected_labels, img_dir, n_per_label=1
        )
    else:
        os.makedirs(img_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not img_paths:
        print(f"[Error] No images in {img_dir}. Use --extract to sample from parquet.")
        return
    print(f"[System] {len(img_paths)} images ({'freshly sampled' if extract_new else 'existing files'})")

    # ---- 加载模型 ----
    print("\n[System] Loading models...")
    loaded_models = {}
    all_needed = set()
    for exp in EXPERIMENTS:
        all_needed.update(exp.model_names)
    for name in sorted(all_needed):
        loaded_models[name] = load_model(name, device)
    print(f"[System] {len(loaded_models)} models loaded.\n")

    # ---- 使用统一接口跑所有对照组 ----
    plotter = ComparisonPlotter(output_root=vis_root)

    for exp in EXPERIMENTS:
        print(f"\n{'='*70}")
        print(f"  {exp.title}")
        print(f"{'='*70}")

        models = [loaded_models[name] for name in exp.model_names]
        plotter.plot_all_methods(exp, ALL_METHODS, img_paths, models, exp.model_names)

    print(f"\n{'='*70}")
    print(f"  Done! Results saved to {vis_root}/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', action='store_true',
                        help='从 parquet 随机提取新图片（默认使用已有图片）')
    args = parser.parse_args()
    main(extract_new=args.extract)
