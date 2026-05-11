import os

# 限制底层 C 库线程，防止多进程数据加载时线程锁死
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
import glob
import io
import sys
import concurrent.futures

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from sklearn.metrics import accuracy_score, f1_score

# 引入你的基础设施
from config_mars import MODEL_CONFIGS
from loader import load_model

# ===========================
# 0. 日志与全局设置
# ===========================
LOG_FILE = "benchmark_results.log"
K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15]
MAX_K = max(K_VALUES)

# 指定使用的 GPU 列表 (假设为 0, 1, 2, 3)
GPU_IDS = [0, 1, 2, 3]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ===========================
# 1. 核心：自适应通道变换
# ===========================
class AdaptiveChannelTransform(torch.nn.Module):
    def __init__(self, target_chans, model_name):
        super().__init__()
        self.target_chans = target_chans
        self.model_name = model_name

    def forward(self, img):
        current_mode = img.mode
        if self.target_chans == 1:
            return img.convert('L') if current_mode != 'L' else img
        elif self.target_chans == 3:
            return img.convert('RGB') if current_mode != 'RGB' else img
        return img


def get_transforms(model_name):
    cfg = MODEL_CONFIGS[model_name]
    target_chans = cfg.get('in_chans', 3)
    logger.info(f"   ⚙️ [{model_name}] Target Chans: {target_chans} | Stats: {cfg['mean']}")

    return v2.Compose([
        AdaptiveChannelTransform(target_chans, model_name),
        v2.Resize((cfg['res'], cfg['res']), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=cfg['mean'], std=cfg['std'])
    ])


# ===========================
# 2. 增强版数据加载器 (支持多核并发读取)
# ===========================
def _read_single_parquet(f):
    try:
        df = pd.read_parquet(f)
        if 'label' not in df.columns and 'labels' in df.columns:
            df.rename(columns={'labels': 'label'}, inplace=True)
        return df
    except Exception as e:
        logger.warning(f"   ⚠️ Failed to read {f}: {e}")
        return None


class UniversalParquetDataset(Dataset):
    def __init__(self, data_root, split="train", transform=None):
        self.transform = transform
        search_pattern = os.path.join(data_root, f"{split}-*.parquet")
        files = sorted(glob.glob(search_pattern))

        if not files:
            fallback_pattern = os.path.join(data_root, f"{split}.parquet")
            if os.path.exists(fallback_pattern):
                files = [fallback_pattern]
            else:
                raise FileNotFoundError(f"❌ No parquet files found for '{split}' in {data_root}")

        print(f"   📂 [{split.upper()}] Parallel loading {len(files)} file(s) from {data_root}...")

        # 🟢 优化：使用多线程并发读取 Parquet 文件
        dfs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(files), 16)) as executor:
            results = list(executor.map(_read_single_parquet, files))
            dfs = [df for df in results if df is not None]

        if not dfs:
            raise ValueError(f"Failed to load any data for {split}")

        self.df = pd.concat(dfs, ignore_index=True)
        print(f"   ✅ Loaded {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_data = row['image']

        if isinstance(img_data, dict) and 'bytes' in img_data:
            byte_data = img_data['bytes']
        elif isinstance(img_data, bytes):
            byte_data = img_data
        else:
            byte_data = img_data['bytes']

        image = Image.open(io.BytesIO(byte_data))
        label = row['label']

        if self.transform:
            image = self.transform(image)
        return image, label


def get_datasets(dataset_dir_name, model_name):
    transform = get_transforms(model_name)
    data_root = os.path.join("./dataset", dataset_dir_name, "data")
    if not os.path.exists(data_root):
        data_root = os.path.join("./dataset", dataset_dir_name)

    train_ds = UniversalParquetDataset(data_root, "train", transform)
    test_ds = UniversalParquetDataset(data_root, "test", transform)
    return train_ds, test_ds


# ===========================
# 3. 工具与 GPU KNN 函数
# ===========================
def gem_pooling(x, p=3, eps=1e-6):
    x = x.clamp(min=eps)
    return x.pow(p).mean(dim=1).pow(1.0 / p)


def get_pooled_features(cls_token, patch_tokens, mode):
    if mode == 'cls':
        return cls_token
    elif mode == 'mean':
        return patch_tokens.mean(dim=1)
    elif mode == 'max':
        return patch_tokens.max(dim=1)[0]
    elif mode == 'gem':
        return gem_pooling(patch_tokens, p=3)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def extract_all_features(model, loader, device):
    features = {'cls': [], 'patch': []}
    labels = []

    model.eval()
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Extracting", leave=False):
            imgs = imgs.to(device, non_blocking=True)

            # 🟢 多卡前向传播
            cls_tok, patch_tok = model(imgs)

            # 先转移到 CPU 存入内存，防止 OOM
            features['cls'].append(cls_tok.cpu())
            features['patch'].append(patch_tok.cpu())

            if isinstance(targets, list):
                targets = np.array(targets)
            elif isinstance(targets, torch.Tensor):
                targets = targets.cpu().numpy()
            labels.append(targets)

    features['cls'] = torch.cat(features['cls'], dim=0)
    features['patch'] = torch.cat(features['patch'], dim=0)

    try:
        labels = np.concatenate(labels, axis=0)
    except:
        labels = np.array([item for sublist in labels for item in sublist])

    return features, labels


# 🟢 优化：纯 GPU 批处理 KNN，一次计算，多次评估
def fast_gpu_knn(X_train, y_train, X_test, y_test, k_values, device='cuda', batch_size=4096):
    max_k = max(k_values)

    # 将标签转为 tensor 放入 GPU
    y_train_t = torch.tensor(y_train, device=device)
    X_train_t = X_train.to(device)

    all_preds = {k: [] for k in k_values}

    # 测试集特征分批处理，避免测试集太大导致 GPU OOM
    for i in range(0, X_test.size(0), batch_size):
        X_test_batch = X_test[i:i + batch_size].to(device)

        # 计算余弦相似度矩阵: [test_batch, num_train]
        # (因为特征已经 L2 归一化，内积就是余弦相似度，相似度最大即距离最近)
        sim = torch.mm(X_test_batch, X_train_t.T)

        # 找出最相似的 top K_MAX 个索引
        _, topk_indices = sim.topk(max_k, dim=1)  # [test_batch, max_k]
        topk_labels = y_train_t[topk_indices]  # [test_batch, max_k]

        # 一次性计算出各个 K 值的多数投票结果
        for k in k_values:
            k_labels = topk_labels[:, :k]
            # torch.mode 返回 (values, indices)，我们只需要 values
            preds, _ = torch.mode(k_labels, dim=1)
            all_preds[k].append(preds.cpu().numpy())

    # 计算评估指标
    results = {}
    for k in k_values:
        preds = np.concatenate(all_preds[k])
        acc = accuracy_score(y_test, preds)
        f1_micro = f1_score(y_test, preds, average='micro', zero_division=0)
        f1_macro = f1_score(y_test, preds, average='macro', zero_division=0)
        results[k] = (acc, f1_micro, f1_macro)

    return results


# ===========================
# 4. 主流程
# ===========================
def run_benchmark():
    DATASETS = [
        "mb-atmospheric_dust_cls_rdr",
        "mb-change_cls_ctx",
        "mb-change_cls_hirise",
        "mb-domars16k",
        "mb-frost_cls",
        "mb-landmark_cls"
    ]

    MODELS = [
        "vit_l",
        "swin_v2",
        "satmae",
        "mars_mae",
        "siglip2",
        "openclip",
        "dinov1_b8",
        "dinov2_l14",
        "dinov3_lvd",
        "dinov3_sat",
        "dinov3_lvd_1cha",
        "dinov3_sat_1cha",
    ]
    POOLING_MODES = ["cls", "mean", "max", "gem"]

    # 设置主设备
    device = torch.device(f"cuda:{GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    logger.info("=" * 60)
    logger.info("🚀 Running ACCELERATED UNIVERSAL KNN Benchmark")
    logger.info(f"   Datasets: {len(DATASETS)}")
    logger.info(f"   Models:   {len(MODELS)}")
    logger.info(f"   GPUs:     {num_gpus} detected, using IDs {GPU_IDS}")
    logger.info("=" * 60)
    logger.info("Dataset,Model,Pooling,K,Accuracy,F1_Micro,F1_Macro")

    for ds_name in DATASETS:
        logger.info(f"\n📂 Dataset: {ds_name}")

        for model_name in MODELS:
            logger.info(f"   👉 Model: {model_name}")

            try:
                # 1. 加载数据
                train_ds, test_ds = get_datasets(ds_name, model_name)

                # 🟢 优化: 根据 GPU 数量动态扩大 Batch Size，开启 pin_memory 和 prefetch
                base_bs = 64
                total_bs = base_bs * len(GPU_IDS) if num_gpus > 1 else base_bs

                train_loader = DataLoader(
                    train_ds, batch_size=total_bs, shuffle=False,
                    num_workers=8, pin_memory=True, prefetch_factor=4
                )
                test_loader = DataLoader(
                    test_ds, batch_size=total_bs, shuffle=False,
                    num_workers=8, pin_memory=True, prefetch_factor=4
                )

                # 2. 加载模型
                model = load_model(model_name, device)

                # 🟢 优化: 多 GPU 包装 (DataParallel)
                if num_gpus > 1:
                    model = torch.nn.DataParallel(model, device_ids=GPU_IDS)
                    model.to(device)

            except Exception as e:
                logger.error(f"   ❌ Skipped {ds_name}/{model_name}: {e}")
                continue

            # 3. 提取特征
            try:
                train_feats_raw, y_train = extract_all_features(model, train_loader, device)
                test_feats_raw, y_test = extract_all_features(model, test_loader, device)
            except Exception as e:
                logger.error(f"   ❌ Feature extraction failed: {e}")
                del model
                torch.cuda.empty_cache()
                continue

            # 释放模型显存，留给接下来的矩阵乘法
            del model
            torch.cuda.empty_cache()

            is_multilabel = len(y_train.shape) > 1 and y_train.shape[1] > 1
            if is_multilabel:
                logger.warning(f"   ⚠️ Multilabel dataset detected. KNN accuracy will be strict exact match.")

            # 4. 极速 KNN 验证
            for pool_mode in POOLING_MODES:
                X_train_t = get_pooled_features(train_feats_raw['cls'], train_feats_raw['patch'], pool_mode)
                X_test_t = get_pooled_features(test_feats_raw['cls'], test_feats_raw['patch'], pool_mode)

                # L2 归一化后准备传入自写的 GPU KNN
                X_train_t = F.normalize(X_train_t, p=2, dim=1)
                X_test_t = F.normalize(X_test_t, p=2, dim=1)

                # 🟢 优化: 批量调用手写的 GPU KNN，秒出结果
                knn_results = fast_gpu_knn(X_train_t, y_train, X_test_t, y_test, K_VALUES, device=device)

                for k in K_VALUES:
                    acc, f1_micro, f1_macro = knn_results[k]
                    result_line = f"{ds_name},{model_name},{pool_mode},{k},{acc:.4f},{f1_micro:.4f},{f1_macro:.4f}"
                    logger.info(result_line)


if __name__ == "__main__":
    run_benchmark()