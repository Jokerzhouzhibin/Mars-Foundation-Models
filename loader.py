import os
import torch
import timm
from torchvision.models import vit_l_16, swin_v2_b
from transformers import AutoModel, CLIPVisionModel
from config_mars import BASE_PATH, MODEL_CONFIGS

# ==========================================
# 核心工具：加载审计函数 (保持不变)
# ==========================================
def smart_load_and_report(model, state_dict, model_name):
    print(f"   🔍 Auditing weights for [{model_name}]...")
    new_ckpt = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_ckpt[k[7:]] = v
        else:
            new_ckpt[k] = v
    state_dict = new_ckpt

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    all_keys = list(model.state_dict().keys())
    total_params = len(all_keys)
    loaded_params = total_params - len(missing_keys)
    match_rate = (loaded_params / total_params) * 100

    ignore_keywords = ['head', 'fc', 'classifier', 'prediction', 'decoder']
    real_missing = [k for k in missing_keys if not any(x in k for x in ignore_keywords)]

    if len(real_missing) == 0 and match_rate > 99:
        print(f"   ✅ [PERFECT MATCH] {loaded_params}/{total_params} layers loaded ({match_rate:.1f}%)")
    else:
        print(f"   ⚠️ [PARTIAL MATCH] {loaded_params}/{total_params} layers loaded ({match_rate:.1f}%)")
        input_layer_missing = any('patch_embed' in k or 'conv1' in k for k in real_missing)
        if input_layer_missing:
            print(f"\n   🚨🚨🚨 CRITICAL WARNING 🚨🚨🚨")
            print(f"   Input layers (patch_embed/conv1) are MISSING! The model sees NOISE.")
        elif len(real_missing) > 0:
            print(f"   👉 Missing (First 5): {real_missing[:5]}")

    return model


# ==========================================
# 模型包装类 (🔥 修改：支持 kwargs 传参)
# ==========================================
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, model_type):
        super().__init__()
        self.model = model
        self.type = model_type

    # 🔥 修改：增加 **kwargs 以接收额外参数
    def forward(self, x, **kwargs):

        if self.type == "torchvision_vit":
            x = self.model._process_input(x)
            n = x.shape[0]
            batch_class_token = self.model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.model.encoder(x)
            return x[:, 0], x[:, 1:]

        elif self.type == "torchvision_swin":
            x = self.model.features(x)
            x = self.model.norm(x)
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)
            cls_token = x.mean(dim=1)
            return cls_token, x

        elif self.type == "timm_mae":
            x = self.model.forward_features(x)
            return x[:, 0], x[:, 1:]

        elif self.type == "transformers":  # SigLIP
            out = self.model(x)
            last_hidden = out.last_hidden_state
            if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                cls_token = out.pooler_output
            else:
                cls_token = last_hidden[:, 0]
            return cls_token, last_hidden

        elif self.type == "transformers_clip":  # OpenCLIP
            out = self.model(x, output_hidden_states=True)
            last_hidden = out.last_hidden_state
            return out.pooler_output, last_hidden[:, 1:]

        elif self.type == "dinov1":
            x = self.model.get_intermediate_layers(x, n=1)[0]
            return x[:, 0], x[:, 1:]

        elif self.type == "dinov2_hub":
            x = self.model.forward_features(x)
            return x['x_norm_clstoken'], x['x_norm_patchtokens']

        elif self.type == "dinov3_hub":
            x = self.model.forward_features(x)
            if isinstance(x, dict):
                return x['x_norm_clstoken'], x['x_norm_patchtokens']
            else:
                return x[:, 0], x[:, 1:]

        return None, None


# ==========================================
# 加载主函数 (🔥 修改：增加 scalemae 分支)
# ==========================================
def load_model(name, device='cuda'):
    cfg = MODEL_CONFIGS[name]
    print(f"\n>>> Loading Model: {name} ({cfg['type']})")

    model = None
    if name != "dinov1_b8":
        full_path = os.path.join(BASE_PATH, cfg['path'])

    # ---------------------------
    # 🔥 新增：ScaleMAE 官方加载逻辑
    # ---------------------------
    if cfg['type'] == "scalemae":
        if not HAS_SCALEMAE:
            raise ImportError("Cannot load ScaleMAE: 'scalemae_vit.py' missing.")

        # 使用官方架构初始化 (通常不需要 num_classes，因为只取特征)
        m = mae_vit_large_patch16(img_size=cfg['res'])
        ckpt = torch.load(full_path, map_location='cpu', weights_only=False)
        if 'model' in ckpt: ckpt = ckpt['model']

        # 审计并加载
        m = smart_load_and_report(m, ckpt, name)
        model = ModelWrapper(m, cfg['type'])

    # ---------------------------
    # 1. Torchvision Models
    # ---------------------------
    elif cfg['type'] == "torchvision_vit":
        m = vit_l_16(weights=None)
        ckpt = torch.load(full_path, map_location='cpu', weights_only=False)
        m = smart_load_and_report(m, ckpt, name)
        model = ModelWrapper(m, cfg['type'])

    elif cfg['type'] == "torchvision_swin":
        m = swin_v2_b(weights=None)
        ckpt = torch.load(full_path, map_location='cpu', weights_only=False)
        m = smart_load_and_report(m, ckpt, name)
        model = ModelWrapper(m, cfg['type'])

    # ---------------------------
    # 2. Timm Models (MAE / SatMAE)
    # ---------------------------
    elif cfg['type'] == "timm_mae":
        in_chans = cfg.get('in_chans', 3)
        print(f"   ℹ️  Config: in_chans={in_chans}")
        m = timm.create_model(cfg['arch'], pretrained=False, num_classes=0, in_chans=in_chans)
        ckpt = torch.load(full_path, map_location='cpu', weights_only=False)
        if 'model' in ckpt:
            ckpt = ckpt['model']
        elif 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        m = smart_load_and_report(m, ckpt, name)
        model = ModelWrapper(m, cfg['type'])

    # ---------------------------
    # 3. Transformers (HuggingFace)
    # ---------------------------
    elif cfg['type'] in ["transformers", "transformers_clip"]:
        loader_cls = CLIPVisionModel if "clip" in cfg['type'] else AutoModel
        m, loading_info = loader_cls.from_pretrained(full_path, local_files_only=True, output_loading_info=True)
        # ... (保持原样) ...
        if hasattr(m, "vision_model"): m = m.vision_model
        model = ModelWrapper(m, cfg['type'])

    # ---------------------------
    # DINO v1/v2/v3
    # ---------------------------
    elif cfg['type'] == "dinov1":
        # 本地加载 DINOv1 ViT-Base/8：用 timm 创建架构，加载本地权重
        m = timm.create_model('vit_base_patch8_224', pretrained=False, num_classes=0)
        full_path = os.path.join(BASE_PATH, cfg['path'])
        ckpt = torch.load(full_path, map_location='cpu', weights_only=False)
        if 'teacher' in ckpt:
            ckpt = ckpt['teacher']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        m = smart_load_and_report(m, ckpt, name)
        model = ModelWrapper(m, "timm_mae")  # 复用 timm_mae 的 forward 逻辑

    elif cfg['type'] == "dinov2_hub":
        repo_dir = os.path.join(BASE_PATH, "model_weights/dinov2")
        m = torch.hub.load(repo_dir, 'dinov2_vitl14', source='local', pretrained=False)
        ckpt = torch.load(full_path, map_location='cpu')
        if "teacher" in ckpt:
            ckpt = ckpt["teacher"]
        elif "model" in ckpt:
            ckpt = ckpt["model"]
        m = smart_load_and_report(m, ckpt, name)
        model = ModelWrapper(m, cfg['type'])

    elif cfg['type'] == "dinov3_hub":
        repo_dir = os.path.join(BASE_PATH, "model_weights/dinov3")
        # 🔥 新增：从 config 中获取通道数，默认是 3
        in_chans = cfg.get('in_chans', 3)
        print(f"  ℹ️  DINOv3 Config: in_chans={in_chans}")
        # 🔥 修改：将 in_chans 作为 kwargs 传给本地的 hub 代码
        m = torch.hub.load(repo_dir, 'dinov3_vitl16', source='local', pretrained=False, in_chans=in_chans)
        st = torch.load(full_path, map_location='cpu')
        if "teacher" in st:
            st = st["teacher"]
        elif "model" in st:
            st = st["model"]
        m = smart_load_and_report(m, st, name)
        model = ModelWrapper(m, cfg['type'])

    return model.to(device).eval()
