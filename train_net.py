import os
import argparse
import glob
import io
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from sklearn.metrics import accuracy_score, f1_score

# 引入你的基础设施
from config_mars import MODEL_CONFIGS
from loader import load_model

# 🟢 1. L40 优化: 启用 TF32
torch.set_float32_matmul_precision('high')


# ===========================
# 0. 基础设施：双向日志记录器
# ===========================
class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()


# ===========================
# 1. 配置中心 (🔥 修改：加入 GSD 参数)
# ===========================
DATASET_META = {
    # --- CTX (5m) ---
    "mb-change_cls_ctx": {"nc": 2, "multi": False},
    "mb-domars16k": {"nc": 15, "multi": False},

    # --- HiRISE (0.3m) ---
    "mb-change_cls_hirise": {"nc": 2, "multi": False},
    "mb-frost_cls": {"nc": 2, "multi": False},
    "mb-landmark_cls": {"nc": 8, "multi": False},
    "mb-atmospheric_dust_cls_rdr": {"nc": 2, "multi": False},

    # --- Rover (1m Baseline) ---
    "mb-surface_cls": {"nc": 36, "multi": False},
    "mb-surface_multi_label_cls": {"nc": 25, "multi": True},
}


# ===========================
# 2. 通用 Transform (自适应通道)
# ===========================
class AdaptiveChannelTransform(torch.nn.Module):
    def __init__(self, target_chans):
        super().__init__()
        self.target_chans = target_chans

    def forward(self, img):
        current_mode = img.mode
        if self.target_chans == 1:
            if current_mode == 'RGB':
                return img.convert('L')
            elif current_mode == 'L':
                return img
            else:
                return img.convert('L')
        elif self.target_chans == 3:
            if current_mode == 'L':
                return img.convert('RGB')
            elif current_mode == 'RGB':
                return img
            else:
                return img.convert('RGB')
        return img


def get_transforms(model_name, dataset_name, is_train=True):
    cfg = MODEL_CONFIGS[model_name]
    target_chans = cfg.get('in_chans', 3)

    ORBITAL_DATASETS = [
        "mb-domars16k",
        "mb-landmark_cls",
        "mb-frost_cls",
        "mb-change_cls_ctx",
        "mb-change_cls_hirise",
        "mb-atmospheric_dust_cls_rdr"
    ]

    is_orbital = any(x in dataset_name for x in ORBITAL_DATASETS)

    ops = []
    # 1. 通道适配
    ops.append(AdaptiveChannelTransform(target_chans))

    if is_train:
        # A. 几何增强
        ops.append(v2.RandomResizedCrop((cfg['res'], cfg['res']), scale=(0.2, 1.0), antialias=True))

        ops.append(v2.RandomHorizontalFlip(p=0.5))

        # if is_orbital:
        #     ops.append(v2.RandomHorizontalFlip(p=0.5))
        #     ops.append(v2.RandomVerticalFlip(p=0.5))
        #     ops.append(v2.RandomRotation(degrees=180))
        # else:
        #     ops.append(v2.RandomHorizontalFlip(p=0.5))
        #     ops.append(v2.RandomRotation(degrees=15))

        # B. 光度增强
        ops.append(v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.0))
    else:
        # 验证集
        ops.append(v2.Resize((cfg['res'], cfg['res']), antialias=True))

    # 3. 归一化
    ops.append(v2.ToImage())
    ops.append(v2.ToDtype(torch.float32, scale=True))
    ops.append(v2.Normalize(mean=cfg['mean'], std=cfg['std']))

    return v2.Compose(ops)


# ===========================
# 3. 数据集类
# ===========================
class MarsParquetDataset(Dataset):
    def __init__(self, data_root, split="train", transform=None, is_multi_label=False, num_classes=2, data_fraction="1.0x"):
        self.transform = transform
        self.is_multi_label = is_multi_label
        self.num_classes = num_classes

        if split == "train" and data_fraction and data_fraction != "1.0x":
            search_pattern = os.path.join(data_root, f"partition_train_{data_fraction}_*.parquet")
        else:
            search_pattern = os.path.join(data_root, f"{split}-*.parquet")
            
        files = sorted(glob.glob(search_pattern))

        if not files:
            fallback = os.path.join(data_root, f"{split}.parquet")
            if os.path.exists(fallback): files = [fallback]

        if not files:
            if split == 'test':
                fallback_val = os.path.join(data_root, "val.parquet")
                if os.path.exists(fallback_val):
                    files = [fallback_val]
                    print(f"   ⚠️ 'test' not found, using 'val' instead.")

            if not files:
                raise FileNotFoundError(f"❌ No {split} parquet files found in {data_root}")

        print(f"   📂 [{split.upper()}] Loading {len(files)} shards...")
        dfs = []
        for f in files:
            try:
                df_shard = pd.read_parquet(f)
                if 'label' not in df_shard.columns and 'labels' in df_shard.columns:
                    df_shard.rename(columns={'labels': 'label'}, inplace=True)
                dfs.append(df_shard)
            except Exception as e:
                print(f"   ⚠️ Error reading shard {f}: {e}")

        self.df = pd.concat(dfs, ignore_index=True)
        print(f"   ✅ Loaded {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_data = row['image']
        byte_data = img_data['bytes'] if isinstance(img_data, dict) else img_data
        image = Image.open(io.BytesIO(byte_data))

        if self.transform:
            image = self.transform(image)

        raw_label = row['label']

        if self.is_multi_label:
            if hasattr(raw_label, '__len__') and len(raw_label) == self.num_classes:
                label = torch.tensor(raw_label, dtype=torch.float32)
            elif hasattr(raw_label, '__len__'):
                target = torch.zeros(self.num_classes, dtype=torch.float32)
                for l in raw_label:
                    if l < self.num_classes: target[int(l)] = 1.0
                label = target
            else:
                target = torch.zeros(self.num_classes, dtype=torch.float32)
                target[int(raw_label)] = 1.0
                label = target
        else:
            label = int(raw_label)

        return image, label


# ===========================
# 4. 模型包装器 (🔥 修改：结构自检 & 多策略支持)
# ===========================
class MarsClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes, device, mode="linear", dataset_name=None, strategy_override=None):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.backbone = load_model(backbone_name, device)

        # 1. 识别模型特性
        self.is_swin = 'swin' in self.backbone_name

        # 3. 策略定义 (外部传入优先)
        # 选项: "original" (按旧逻辑), "concat_2" (CLS+Mean), "concat_3" (CLS+Mean+Max)
        self.target_strategy = strategy_override if strategy_override else "original"

        # 4. 动态计算维度 & 结构自检 (CRITICAL STEP)
        cfg = MODEL_CONFIGS[backbone_name]
        in_chans = cfg.get('in_chans', 3)
        dummy = torch.randn(1, in_chans, cfg['res'], cfg['res']).to(device)
        kwargs = {}

        print(f"\n   🔍 [Structure Check] {backbone_name} Output Analysis:")
        with torch.no_grad():
            output = self.backbone(dummy, **kwargs)

            # --- 打印输出结构 ---
            if isinstance(output, tuple):
                print(f"      Type: Tuple (len={len(output)})")
                for i, item in enumerate(output):
                    if isinstance(item, torch.Tensor):
                        print(f"      Item {i}: Shape {tuple(item.shape)}")
                    else:
                        print(f"      Item {i}: Type {type(item)}")
            elif isinstance(output, torch.Tensor):
                print(f"      Type: Tensor | Shape {tuple(output.shape)}")
            elif isinstance(output, dict):
                print(f"      Type: Dict | Keys {output.keys()}")
            else:
                print(f"      Type: {type(output)}")

            # --- 维度计算逻辑 ---
            self.has_cls = False
            self.has_patch = False
            cls_dim = 0
            patch_dim = 0

            # 情况 A: 标准 ViT/DINO/MAE (Tuple: cls, patches)
            if isinstance(output, tuple) and len(output) >= 2:
                # 假设 [0] 是 CLS, [1] 是 Patch
                # 校验: CLS 应该是 (B, D), Patch 应该是 (B, N, D)
                item0 = output[0]
                item1 = output[1]

                if item0.ndim == 2:  # (B, D)
                    self.has_cls = True
                    cls_dim = item0.shape[-1]

                if item1.ndim == 3:  # (B, N, D)
                    self.has_patch = True
                    patch_dim = item1.shape[-1]

                print(f"      ✅ Identified: CLS={self.has_cls}, Patch={self.has_patch}")

            # 情况 B: Swin / 特殊模型 (Tensor: B, C, H, W or B, L, C)
            elif isinstance(output, torch.Tensor):
                self.has_cls = False  # Swin 通常没有独立的 CLS
                self.has_patch = True
                patch_dim = output.shape[-1] if output.ndim == 3 else output.shape[1]  # Swin (B, C, H, W) -> C is dim
                print(f"      ✅ Identified: Pure Feature Map (No CLS token)")

            # --- 决定最终维度 ---
            if self.target_strategy == "concat_3":
                # CLS + Mean + Max
                dim = 0
                if self.has_cls: dim += cls_dim
                if self.has_patch: dim += (patch_dim * 2)  # Mean + Max
                self.embed_dim = dim

            elif self.target_strategy == "concat_2":
                # CLS + Mean
                dim = 0
                if self.has_cls: dim += cls_dim
                if self.has_patch: dim += patch_dim  # Mean
                self.embed_dim = dim

            else:
                # Original logic (fallback)
                # DINO/ViT -> CLS
                # MAE/Swin -> Mean
                # CLIP -> CLS
                is_mae = any(k in self.backbone_name for k in ['mae', 'sat', 'scale'])
                is_clip = any(k in self.backbone_name for k in ['clip', 'siglip'])

                if self.is_swin:
                    self.embed_dim = patch_dim
                    self.target_strategy = "GAP_only"
                elif is_mae:
                    self.embed_dim = patch_dim
                    self.target_strategy = "GAP_only"
                elif is_clip:
                    self.embed_dim = cls_dim
                    self.target_strategy = "CLS_only"
                else:  # DINO/ViT
                    self.embed_dim = cls_dim
                    self.target_strategy = "CLS_only"

        print(f"      👉 Final Strategy: {self.target_strategy} | Input Dim: {self.embed_dim}")

        # 5. 初始化分类头
        self.classifier = nn.Linear(self.embed_dim, num_classes).to(device)

        # 6. 冻结逻辑
        if mode == "linear":
            print(f"   ❄️  Mode: Linear Probe (Backbone Frozen)")
            for param in self.backbone.parameters(): param.requires_grad = False
        elif mode == "full":
            print(f"   🔥 Mode: Full Fine-tuning (Backbone Unfrozen)")
            for param in self.backbone.parameters(): param.requires_grad = True

    def forward(self, x):
        # 1. Backbone Forward
        outputs = self.backbone(x)

        # 3. 特征提取与融合
        features_list = []

        # --- 提取原始组件 ---
        cls_token = None
        patch_tokens = None

        if isinstance(outputs, tuple) and len(outputs) >= 2:
            # 标准 (CLS, Patch) 结构
            if self.has_cls: cls_token = outputs[0]
            if self.has_patch: patch_tokens = outputs[1]
        elif isinstance(outputs, torch.Tensor):
            # Swin 结构
            if self.is_swin and outputs.ndim == 4:  # (B, C, H, W)
                # 拉平为 (B, H*W, C) 以便统一处理
                patch_tokens = outputs.flatten(2).transpose(1, 2)
            else:
                patch_tokens = outputs

        # --- 执行融合策略 ---
        # 1. CLS Token (如果策略需要且模型有)
        if "concat" in self.target_strategy or "CLS" in self.target_strategy:
            if self.has_cls and cls_token is not None:
                features_list.append(cls_token)

        # 2. Patch Mean (如果策略需要且模型有)
        if "concat" in self.target_strategy or "GAP" in self.target_strategy or "Mean" in self.target_strategy:
            if self.has_patch and patch_tokens is not None:
                patch_mean = patch_tokens.mean(dim=1)
                features_list.append(patch_mean)

        # 3. Patch Max (仅在 concat_3 策略下启用)
        if self.target_strategy == "concat_3":
            if self.has_patch and patch_tokens is not None:
                patch_max = patch_tokens.max(dim=1)[0]  # max 返回 (values, indices)
                features_list.append(patch_max)

        # --- 兜底逻辑 ---
        # 如果什么都没加进去 (比如 Swin 没有 CLS 但策略是 CLS_only)，就加个 Mean 防止报错
        if not features_list:
            if patch_tokens is not None:
                features_list.append(patch_tokens.mean(dim=1))
            elif cls_token is not None:
                features_list.append(cls_token)

        # 4. 拼接
        final_feature = torch.cat(features_list, dim=-1)

        # 5. 分类
        logits = self.classifier(final_feature)
        return logits


# ===========================
# 5. 验证函数
# ===========================
def evaluate(model, loader, device, is_multi, amp_dtype):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # 验证时也使用 AMP
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = model(imgs)

            if is_multi:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
            else:
                preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    if is_multi:
        score = f1_score(all_targets, all_preds, average='micro', zero_division=0) * 100
        metric_name = "Micro-F1"
    else:
        score = accuracy_score(all_targets, all_preds) * 100
        metric_name = "Accuracy"

    return score, metric_name


# ===========================
# 6. 训练主流程
# ===========================
def train(args):
    log_dir = f"./logs/{args.mode}_records"
    os.makedirs(log_dir, exist_ok=True)

    # 建议修改为包含 data_fraction
    log_filename = f"{args.dataset}_{args.model}_frac{args.data_fraction}_lr{args.lr}_{args.strategy}.log"
    log_path = os.path.join(log_dir, log_filename)

    sys.stdout = DualLogger(log_path)

    print(f"\n{'=' * 60}")
    print(f"📄 Log File: {log_path}")
    print(f"⏰ Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training {args.model} on {args.dataset} | Mode: {args.mode.upper()} | LR: {args.lr}")

    # 1. 元数据
    if args.dataset not in DATASET_META:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    meta = DATASET_META[args.dataset]
    num_classes = meta['nc']
    is_multi = meta['multi']

    data_root = os.path.join("./dataset", args.dataset, "data")
    if not os.path.exists(data_root): data_root = os.path.join("./dataset", args.dataset)

    # 2. 数据准备
    train_tf = get_transforms(args.model, dataset_name=args.dataset, is_train=True)
    val_tf = get_transforms(args.model, dataset_name=args.dataset, is_train=False)

    train_ds = MarsParquetDataset(data_root, "train", train_tf, is_multi, num_classes, data_fraction=args.data_fraction)

    try:
        val_ds = MarsParquetDataset(data_root, "test", val_tf, is_multi, num_classes)
    except:
        print("   ⚠️ Test set not found, splitting Train set (80/20)...")
        train_size = int(0.8 * len(train_ds))
        val_size = len(train_ds) - train_size
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

    # 动态计算 Workers
    is_small_data = len(train_ds) < 500
    optimal_workers = 0 if is_small_data else 4
    real_batch_size = min(args.batch_size, len(train_ds))

    print(f"   ⚙️  Config: Workers={optimal_workers}, BS={real_batch_size}, SmallData={is_small_data}")

    train_loader = DataLoader(
        train_ds,
        batch_size=real_batch_size,
        shuffle=True,
        num_workers=optimal_workers,
        pin_memory=True,
        drop_last=len(train_ds) > real_batch_size,
        persistent_workers=(optimal_workers > 0)
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=optimal_workers, pin_memory=True)

    # 3. 初始化模型 (🔥 修改：传入 dataset_name 以便获取 GSD)
    # model = MarsClassifier(args.model, num_classes, device, mode=args.mode, dataset_name=args.dataset)
    model = MarsClassifier(
        args.model,
        num_classes,
        device,
        mode=args.mode,
        dataset_name=args.dataset,
        strategy_override=args.strategy  # 传入 concat_2 或 concat_3
    )

    # 4. 优化器
    if args.mode == "linear":
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr)
    else:
        backbone_lr = args.lr * 0.1
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': backbone_lr},
            {'params': model.classifier.parameters(), 'lr': args.lr}
        ], weight_decay=0.05)

    criterion = nn.BCEWithLogitsLoss() if is_multi else nn.CrossEntropyLoss()

    # AMP Config
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"   ⚡ AMP Precision: {amp_dtype}")

    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))

    target_bs = 64
    accum_steps = max(1, target_bs // real_batch_size)

    # 5. Loop Config
    best_score = 0.0
    patience_counter = 0

    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_path = os.path.join(ckpt_dir, f"{args.mode}_{args.dataset}_{args.model}_lr{args.lr}_{args.strategy}.pth")

    print(f"   🔄 Training Start... (Epochs: {args.epochs}, Patience: {args.patience})")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False, file=sys.stderr)

        for i, (imgs, labels) in enumerate(loop):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = model(imgs)
                if not is_multi: labels = labels.long()
                loss = criterion(logits, labels)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()

            train_loss += loss.item() * accum_steps
            loop.set_postfix(loss=loss.item() * accum_steps)

        # --- Validation ---
        val_score, metric = evaluate(model, val_loader, device, is_multi, amp_dtype)

        avg_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        print(f"   📅 Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Val {metric}: {val_score:.2f}%")

        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            # torch.save(model.state_dict(), best_model_path)  # Commented out to save disk space
            print(f"      ⭐ New Best! (Model saving disabled to save space)")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\n   🛑 Early Stopping triggered after {args.patience} epochs.")
            break

    print(f"\n{'=' * 40}")
    print(f"🏆 Best Val {metric}: {best_score:.2f}%")
    print(f"💾 Best Checkpoint: {best_model_path}")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    print("\n[DEBUG] 1. Script is running! Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, default="linear", choices=["linear", "full"])
    parser.add_argument("--lr", type=float, required=True, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--strategy", type=str, default="original",
                        choices=["concat_2", "concat_3", "original"],
                        help="Feature fusion strategy: concat_2 (CLS+Mean) or concat_3 (CLS+Mean+Max)")
    parser.add_argument("--data_fraction", type=str, default="1.0x", help="Dataset fraction to use, e.g., 0.01x, 0.10x, 1.0x")
    args = parser.parse_args()

    if args.mode == 'full' and args.patience == 5:
        args.patience = 8

    train(args)
