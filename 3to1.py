import torch
import sys


def convert_to_one_channel(src_path, dst_path):
    print(f"Loading weights from: {src_path}")
    # 加载权重
    checkpoint = torch.load(src_path, map_location='cpu')

    # 兼容不同的 checkpoint 格式 (有些是 'model', 'teacher', 有些直接是 state_dict)
    keys_to_check = ['model', 'teacher', 'student']
    processed_keys = []

    # 如果 checkpoint 本身就是 state_dict (没有嵌套结构)
    if not any(k in checkpoint for k in keys_to_check) and 'patch_embed.proj.weight' in checkpoint:
        # 封装一下以便统一处理
        checkpoint = {'model': checkpoint}
        keys_to_check = ['model']

    for key in checkpoint.keys():
        # 只处理包含网络权重的字典
        if isinstance(checkpoint[key], dict) and 'patch_embed.proj.weight' in checkpoint[key]:
            print(f"Processing key: [{key}]")
            state_dict = checkpoint[key]

            # 定位第一层卷积权重
            weight_name = 'patch_embed.proj.weight'
            w = state_dict[weight_name]  # shape: [Embed_Dim, 3, Patch, Patch]

            if w.shape[1] == 3:
                print(f"  - Found RGB weights shape: {w.shape}")
                # 🔥 核心操作：在通道维度 (dim=1) 求和
                new_w = w.sum(dim=1, keepdim=True)
                state_dict[weight_name] = new_w
                print(f"  - Converted to 1-channel shape: {new_w.shape}")
                processed_keys.append(key)
            else:
                print(f"  - Skipping {weight_name}, already shape {w.shape}")

    if not processed_keys:
        print("❌ Warning: No RGB weights were converted! Check key names.")
    else:
        print(f"✅ Success! Saving to: {dst_path}")
        torch.save(checkpoint, dst_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert 3-channel patch_embed weights to 1-channel via sum.")
    parser.add_argument("--src", type=str, nargs="+", help="Source weight file(s)")
    parser.add_argument("--dst", type=str, nargs="+", help="Destination weight file(s)")
    args = parser.parse_args()

    # Default: convert both lvd and sat
    if args.src is None or args.dst is None:
        pairs = [
            ('model_weights/dinov3_lvd.pth', 'model_weights/dinov3_lvd_1cha.pth'),
            ('model_weights/dinov3_sat.pth', 'model_weights/dinov3_sat_1cha.pth'),
        ]
    else:
        pairs = list(zip(args.src, args.dst))

    for src, dst in pairs:
        convert_to_one_channel(src, dst)

