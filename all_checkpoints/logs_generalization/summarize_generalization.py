import os
import re
import pandas as pd
import shutil

# Define the datasets and models to match
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

# Define the source and target directories for file copying
SOURCE_DIR = '../train_all_logs/linear_records'
TARGET_DIR = './linear_records'

# Copy files ending with 'concat_3.log' from source to target
def copy_concat_3_logs(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith('concat_3.log'):
            shutil.copy(os.path.join(source_dir, filename), os.path.join(target_dir, filename))

    print(f"✅ Files ending with 'concat_3.log' have been copied from {source_dir} to {target_dir}.")

# Update the process_generalization_logs function to handle files without datafraction
def process_generalization_logs(dir_path, output_file):
    results = []

    # Regular expressions for extracting data
    metric_pattern = re.compile(r'(Best Val Accuracy|Best Accuracy)[:\s]+([\d\.]+)')
    datafraction_pattern = re.compile(r'_frac([\d\.]+)x')

    print(f"\n🔍 Scanning directory: {dir_path}")

    if not os.path.exists(dir_path):
        print(f"⚠️ Directory does not exist: {dir_path}")
        return

    sorted_datasets = sorted(DATASETS, key=len, reverse=True)

    for filename in os.listdir(dir_path):
        full_path = os.path.join(dir_path, filename)

        if not os.path.isfile(full_path):
            continue

        matched_ds = None
        matched_md = None

        # Match dataset
        for ds in sorted_datasets:
            if filename.startswith(ds):
                matched_ds = ds
                break

        # Match model
        if matched_ds:
            remaining = filename[len(matched_ds):]
            sorted_models = sorted(MODELS, key=len, reverse=True)
            for md in sorted_models:
                if md in remaining:
                    matched_md = md
                    break

        # Extract data if both dataset and model are matched
        if matched_ds and matched_md:
            try:
                # Extract datafraction or set to 'full' if not present
                datafraction_match = datafraction_pattern.search(filename)
                datafraction = datafraction_match.group(1) if datafraction_match else "full"

                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    # Extract metric
                    metric_match = metric_pattern.search(content)

                    if metric_match:
                        score = float(metric_match.group(2))

                        results.append({
                            'Dataset': matched_ds,
                            'Model': matched_md,
                            'Datafraction': datafraction,
                            'Score': score,
                            'Filename': filename
                        })
            except Exception as e:
                print(f"⚠️ Error reading {filename}: {e}")

    # Summarize results
    if not results:
        print(f"❌ No valid data extracted from {dir_path}.")
        return

    df = pd.DataFrame(results)

    # Save full results
    df.to_csv(f'{output_file}_all.csv', index=False)

    # Find the best model for each dataset and datafraction
    best_df = df.loc[df.groupby(['Dataset', 'Datafraction'])['Score'].idxmax()].copy()
    best_df = best_df.sort_values(['Dataset', 'Datafraction'])

    # Save best results
    best_csv = f'{output_file}_best.csv'
    best_df[['Dataset', 'Datafraction', 'Model', 'Score']].to_csv(best_csv, index=False)

    print(f"✅ Processing complete. Results saved to {output_file}_all.csv and {output_file}_best.csv")

if __name__ == "__main__":
    # copy_concat_3_logs(SOURCE_DIR, TARGET_DIR)
    process_generalization_logs(TARGET_DIR, 'generalization_results')
