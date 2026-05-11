import pandas as pd
import csv
import re


def extract_best_performance(input_csv, output_csv):
    print(f"📂 正在读取数据: {input_csv}...")

    # 1. 读取 CSV 数据
    # 如果文件极大（GB级别），可以添加 low_memory=False
    df = pd.read_csv(input_csv)

    # 2. 核心逻辑：寻找每个 Dataset 和 Model 组合下的最高 Accuracy
    # idxmax() 返回最大值所在的行索引，这样能保留 Pooling 和 K 的原始信息
    idx = df.groupby(['Dataset', 'Model'])['Accuracy'].idxmax()
    best_df = df.loc[idx]

    # 3. 整理输出格式
    # 按数据集名称和准确率排序，让 SOTA 模型排在前面
    best_df = best_df.sort_values(by=['Dataset', 'Accuracy'], ascending=[True, False])

    # 4. 打印预览 (前 20 条数据)
    print("\n🚀 提取完成！每个模型在各数据集的最佳表现预览：")
    print(best_df.head(20).to_string(index=False))

    # 5. 保存到新的 CSV
    best_df.to_csv(output_csv, index=False)
    print(f"\n✅ 结果已保存至: {output_csv}")


def process_mars_logs(input_file, output_file):
    # 定义匹配 CSV 数据的正则表达式（匹配逗号分隔的行）
    # 规律是：时间戳 | 数据1,数据2,数据3...
    data_pattern = re.compile(r'^.*? \| (.*)$')

    extracted_data = []
    header = None

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和装饰线
            if not line or '===' in line or 'Datasets:' in line or 'Models:' in line:
                continue

            match = data_pattern.match(line)
            if match:
                content = match.group(1)
                # 检查是否是标题行或数据行（通过逗号数量判断）
                parts = [p.strip() for p in content.split(',')]

                if len(parts) == 7:
                    if parts[0] == "Dataset":
                        header = parts
                    else:
                        extracted_data.append(parts)

    # 写入 CSV 文件
    if extracted_data:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
            writer.writerows(extracted_data)
        print(f"✅ 处理完成！成功提取 {len(extracted_data)} 条数据，已保存至: {output_file}")
    else:
        print("❌ 未能在日志中找到有效数据。")

    return output_file


# 执行脚本
if __name__ == "__main__":
    process_mars_logs('benchmark_results_dino3_1cha.log', 'knn_results_dino3_1cha.csv')
    extract_best_performance('knn_results_dino3_1cha.csv', 'best_summary_1cha.csv')
