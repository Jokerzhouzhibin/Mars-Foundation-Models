import os
import re
import pandas as pd

DATASETS = [
    "mb-atmospheric_dust_cls_rdr", "mb-change_cls_ctx", "mb-change_cls_hirise",
    "mb-domars16k", "mb-frost_cls", "mb-landmark_cls",
    "mb-surface_cls", "mb-surface_multi_label_cls"
]

MODELS = [
    "mars_mae", "satmae", "vit_l", "swin_v2",
    "siglip2", "openclip", "dinov3_lvd", "dinov3_sat",
    "dinov2_l14", "dinov1_b8", "dinov3_lvd_1cha", "dinov3_sat_1cha",
]


def process_directory(dir_path, output_suffix):
    results = []

    # --- 正则升级 ---
    # 1. 指标正则：新增对 Best Val Micro-F1 的兼容
    # 捕获组1: 指标名称 (Accuracy, mAP, 或 Micro-F1)
    # 捕获组2: 数值 ([\d\.]+ 会自动提取 85.09，忽略后面的 %)
    metric_pattern = re.compile(r'(Best Val Accuracy|Best mAP|Best Accuracy|Best Val Micro-F1)[:\s]+([\d\.]+)')

    # 2. 学习率正则：严格匹配浮点数或科学计数法，不匹配末尾的 .log
    lr_extract_pattern = re.compile(r'_lr(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)')

    print(f"\n🔍 正在扫描: {dir_path}")

    if not os.path.exists(dir_path):
        print(f"⚠️ 目录不存在: {dir_path}")
        return

    # 按长度排序，优先匹配长名字 (解决了 multi_label 被 surface_cls 截胡的风险)
    sorted_datasets = sorted(DATASETS, key=len, reverse=True)

    # 统计计数器
    count_matched_name = 0
    count_extracted_data = 0

    for filename in os.listdir(dir_path):
        full_path = os.path.join(dir_path, filename)

        if not os.path.isfile(full_path) or "_lr" not in filename:
            continue

        matched_ds = None
        matched_md = None

        # A. 匹配数据集
        for ds in sorted_datasets:
            if filename.startswith(ds):
                matched_ds = ds
                break

        # B. 匹配模型
        if matched_ds:
            remaining = filename[len(matched_ds):]
            # 同样按长度倒序排序，避免短名字截胡长名字
            sorted_models = sorted(MODELS, key=len, reverse=True)
            for md in sorted_models:
                if md in remaining:
                    matched_md = md
                    break

        # C. 数据提取与调试
        if matched_ds and matched_md:
            count_matched_name += 1
            try:
                # 提取 LR
                lr_match = lr_extract_pattern.search(filename)
                # 如果正则还是匹配到了末尾的点，这里再次 strip 保证干净
                lr = lr_match.group(1).rstrip('.') if lr_match else "unknown"

                # 提取特征结构 (cls, concat2, concat3)
                feature_structure = "unknown"
                struct_match = re.search(r'_(original|concat_?2|concat_?3)\.log$', filename)
                if struct_match:
                    feature_structure = struct_match.group(1).replace('_', '')

                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    # 查找指标
                    metric_match = metric_pattern.search(content)

                    if metric_match:
                        metric_name = metric_match.group(1)  # 比如 'Best Val Micro-F1'
                        score = float(metric_match.group(2))  # 对于 85.09%，这里会提取出 85.09

                        results.append({
                            'Dataset': matched_ds,
                            'Model': matched_md,
                            'Feature Structure': feature_structure,
                            'LR': lr,
                            'Score': score,  # 统一叫 Score
                            'Metric': metric_name,  # 记录是 Acc, mAP 还是 Micro-F1
                            'Filename': filename
                        })
                        count_extracted_data += 1
                    else:
                        # ⚠️ 关键调试：更新未找到指标时的报错提示
                        if "change_cls_ctx" in matched_ds or "multi_label" in matched_ds:
                            print(f"  ❌ 文件名匹配但无结果: {filename}")
                            print(f"     -> 原因: 内容中未找到 'Best Val Accuracy', 'Best mAP' 或 'Best Val Micro-F1'")

            except Exception as e:
                print(f"⚠️ 读取报错 {filename}: {e}")
        else:
            # 如果是 ctx 或 multi_label 相关文件但没匹配上名字，打印出来
            if "ctx" in filename or "multi_label" in filename:
                if "_lr" in filename:  # 只关心相关日志
                    print(f"  ❓ 未识别文件名规则: {filename}")

    # --- 结果汇总 ---
    if not results:
        print(f"❌ {output_suffix} 目录下未提取到任何有效数据。")
        return

    df = pd.DataFrame(results)

    # 保存全量表
    df.to_csv(f'all_results_{output_suffix}.csv', index=False)

    # 生成 Best 表 (每个模型在每个数据集中的最好表现)
    best_df = df.loc[df.groupby(['Dataset', 'Model'])['Score'].idxmax()].copy()
    best_df = best_df.sort_values(['Dataset', 'Model'])

    # 保存精简版 Best 表
    best_csv = f'best_results_{output_suffix}.csv'
    best_df[['Dataset', 'Model', 'Feature Structure', 'LR', 'Score', 'Metric']].to_csv(best_csv, index=False)

    print(f"✅ 处理完成 (匹配文件名: {count_matched_name} -> 提取数据: {count_extracted_data})")
    print(f"📊 最优结果预览 ({output_suffix}):")
    # 格式化打印，保留 Metric 列以便确认是否提取到了 mAP 或 Micro-F1
    print("-" * 115)
    print(f"{'DATASET':<32} | {'MODEL':<18} | {'FEATURE':<9} | {'LR':<9} | {'SCORE':<7} | {'METRIC'}")
    print("-" * 115)
    for _, row in best_df.iterrows():
        print(f"{row['Dataset']:<32} | {row['Model']:<18} | {row['Feature Structure']:<9} | {row['LR']:<9} | {row['Score']:<7.4f} | {row['Metric']}")
    print("-" * 115)


if __name__ == "__main__":
    process_directory('full_records', 'full')
    process_directory('linear_records', 'linear')
