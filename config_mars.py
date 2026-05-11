import os

# ===========================
# 1. 统计常量定义
# ===========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMAGENET_MEAN_1cha = (0.449,)
IMAGENET_STD_1cha  = (0.226,)

CLIP_MEAN     = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD      = (0.26862954, 0.26130258, 0.27577711)

# SatMAE++ 专用
SATMAE_MEAN   = (0.41820073, 0.42147991, 0.39912757)
SATMAE_STD    = (0.28774282, 0.27541765, 0.27640175)

# DINOv3 SAT 专用
DINO_SAT_MEAN = (0.430, 0.411, 0.296)
DINO_SAT_STD  = (0.213, 0.156, 0.143)

DINO_SAT_MEAN_1cha = (0.379,)
DINO_SAT_STD_1cha = (0.1707,)

# Mars-DINO 专用
MARS_DINO_MEAN = (0.5034,)
MARS_DINO_STD  = (0.1986,)

SIGLIP_MEAN   = (0.5, 0.5, 0.5)
SIGLIP_STD    = (0.5, 0.5, 0.5)

# 基础路径 (当前目录)
BASE_PATH = os.getcwd()

# ===========================
# 2. 模型配置表
# ===========================
MODEL_CONFIGS = {
    "vit_l": {
        "type": "torchvision_vit",
        "arch": "vit_l_16",
        "path": "model_weights/vit_l_16_imagenet.pth",
        "res": 224,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD
    },
    "swin_v2": {
        "type": "torchvision_swin",
        "arch": "swin_v2_b",
        "path": "model_weights/swin_v2_b_imagenet.pth",
        "res": 256,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD
    },
    "satmae": {
        "type": "timm_mae",
        "arch": "vit_large_patch16_224",
        "path": "model_weights/SatMAE++.pth",
        "res": 224,
        "mean": SATMAE_MEAN,
        "std": SATMAE_STD
    },
    "mars_mae": {
        "type": "timm_mae",
        "arch": "vit_base_patch16_224",
        "path": "model_weights/mars-mae.pth",
        "res": 224,
        "mean": (0.5,),  # ⚠️ 修改：单通道均值
        "std": (0.5,),  # ⚠️ 修改：单通道方差
        "in_chans": 1  # ⚠️ 新增：标记为单通道
    },
    "siglip2": {
        "type": "transformers",
        "path": "model_weights/siglip2", # 文件夹名
        "res": 256,
        "mean": SIGLIP_MEAN,
        "std": SIGLIP_STD
    },
    "openclip": {
        "type": "transformers_clip",
        "path": "model_weights/openclip", # 文件夹名
        "res": 224,
        "mean": CLIP_MEAN,
        "std": CLIP_STD
    },
    "dinov1_b8": {
        "type": "dinov1",
        "path": "model_weights/dinov1_b8.pth",
        "res": 224,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD
    },
    "dinov2_l14": {
        "type": "dinov2_hub",
        "path": "model_weights/dinov2_l14.pth",
        "res": 224,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD
    },
    "dinov3_lvd": {
        "type": "dinov3_hub",
        "variant": "lvd",
        "path": "model_weights/dinov3_lvd.pth",
        "res": 256,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD
    },
    "dinov3_sat": {
        "type": "dinov3_hub",
        "variant": "sat",
        "path": "model_weights/dinov3_sat.pth",
        "res": 256,
        "mean": DINO_SAT_MEAN,
        "std": DINO_SAT_STD
    },
    "dinov3_lvd_1cha": {
        "type": "dinov3_hub",
        "variant": "lvd",
        "path": "model_weights/dinov3_lvd_1cha.pth",
        "res": 256,
        "mean": IMAGENET_MEAN_1cha,
        "std": IMAGENET_STD_1cha,
        "in_chans": 1  # ⚠️ 新增：标记为单通道
    },
    "dinov3_sat_1cha": {
        "type": "dinov3_hub",
        "variant": "sat",
        "path": "model_weights/dinov3_sat_1cha.pth",
        "res": 256,
        "mean": DINO_SAT_MEAN_1cha,
        "std": DINO_SAT_STD_1cha,
        "in_chans": 1  # ⚠️ 新增：标记为单通道
    },
}