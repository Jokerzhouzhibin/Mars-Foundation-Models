# Mars Foundation Models Benchmark

Systematic evaluation of foundation models for Mars terrain classification using single-channel orbital imagery.

This repository accompanies the thesis: *Cross-Domain Feature Extraction and Adaptive Foundation Models for Mars Terrain Classification*.

## Overview

We benchmark **12 pretrained vision models** (including 2 channel-fused variants) across **6 Mars classification datasets** from [MarsBench](https://huggingface.co/collections/BAAI/marsbench), using four evaluation protocols:

- **KNN** — frozen feature clustering quality
- **Linear Probing** — frozen backbone + trainable linear head
- **Full Fine-tuning** — all parameters updated
- **Data Efficiency** — linear probing at 1%/5%/20%/50%/100% data fractions

### Model Matrix

| Model | Type | Params | Input |
|-------|------|--------|-------|
| ViT-L/16 | Supervised (ImageNet) | ~307M | 3ch, 224² |
| Swin-V2-B | Supervised (ImageNet) | ~88M | 3ch, 256² |
| SatMAE++ | SSL (Earth satellite) | ~307M | 3ch, 224² |
| Mars-MAE | SSL (Mars CTX, 4M images) | ~86M | 1ch, 224² |
| OpenCLIP | VLM (LAION) | ~307M | 3ch, 224² |
| SigLIP2 | VLM (WebLI) | ~400M | 3ch, 256² |
| DINOv1-B/8 | SSL (ImageNet) | ~86M | 3ch, 224² |
| DINOv2-L/14 | SSL (LVD-142M) | ~307M | 3ch, 224² |
| DINOv3-LVD | SSL (LVD) | ~307M | 3ch, 256² |
| DINOv3-SAT | SSL (Earth satellite) | ~307M | 3ch, 256² |
| DINOv3-LVD-1ch | SSL (LVD, channel-fused) | ~307M | 1ch, 256² |
| DINOv3-SAT-1ch | SSL (Earth satellite, channel-fused) | ~307M | 1ch, 256² |

### Datasets

| Dataset | Classes | Instrument | Task |
|---------|---------|------------|------|
| mb-atmospheric_dust_cls_rdr | 2 | HiRISE | Dust detection |
| mb-domars16k | 15 | CTX | Terrain classification |
| mb-frost_cls | 2 | HiRISE | Frost detection |
| mb-change_cls_hirise | 2 | HiRISE | Change detection |
| mb-change_cls_ctx | 2 | CTX | Change detection |
| mb-landmark_cls | 8 | HiRISE | Landmark classification |

---

## Quick Start (End-to-End Workflow)

```
1. Clone repo & install deps
2. Download datasets (MarsBench from HuggingFace)
3. Download model weights (from official sources)
4. Run 3to1.py to generate 1-channel variants
5. Run evaluations:
   - eval_knn.py          → KNN benchmark
   - train_net.py         → Linear Probe / Full Fine-tuning
   - eval_generalization.py → Data efficiency curves
   - visualize.py         → Attention visualization
   - eval_attention.py    → Quantitative attention analysis
6. Plot figures:
   - all_checkpoints/plot_results.py → Publication figures
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Jokerzhouzhibin/Mars-Foundation-Models.git
cd Mars-Foundation-Models
```

### 2. Install dependencies

Requires Python >= 3.10.

```bash
pip install -r requirements.txt
```

> **Note:** For GPU support, install PyTorch with CUDA first following [pytorch.org](https://pytorch.org/get-started/locally/), then install the remaining dependencies.

### 3. Download datasets

Download the 6 classification subsets from [MarsBench on HuggingFace](https://huggingface.co/collections/BAAI/marsbench):

```
dataset/
├── mb-atmospheric_dust_cls_rdr/data/   ← place parquet files here
├── mb-change_cls_ctx/data/
├── mb-change_cls_hirise/data/
├── mb-domars16k/data/
├── mb-frost_cls/data/
└── mb-landmark_cls/data/
```

Each dataset's `data/` folder should contain the train/test parquet shards (e.g., `train-00000-of-00001.parquet`, `test-00000-of-00001.parquet`).

### 4. Download model weights

Place all weights under `model_weights/`:

| Model | Source | Filename |
|-------|--------|----------|
| ViT-L/16 | [torchvision](https://download.pytorch.org/models/vit_l_16-852ce7e3.pth) | `vit_l_16_imagenet.pth` |
| Swin-V2-B | [torchvision](https://download.pytorch.org/models/swin_v2_b-781e5279.pth) | `swin_v2_b_imagenet.pth` |
| SatMAE++ | [GitHub](https://github.com/techmn/satmae_pp) | `SatMAE++.pth` |
| Mars-MAE | [HuggingFace](https://huggingface.co/BAAI/Mars-MAE) | `mars-mae.pth` |
| SigLIP2 | [HuggingFace](https://huggingface.co/google/siglip2-base-patch16-256) | `siglip2/` (folder) |
| OpenCLIP | [HuggingFace](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K) | `openclip/` (folder) |
| DINOv1-B/8 | [GitHub](https://github.com/facebookresearch/dino) | `dinov1_b8.pth` |
| DINOv2-L/14 | [GitHub](https://github.com/facebookresearch/dinov2) | `dinov2_l14.pth` |
| DINOv3-LVD | [GitHub](https://github.com/facebookresearch/dinov3) | `dinov3_lvd.pth` |
| DINOv3-SAT | [GitHub](https://github.com/facebookresearch/dinov3) | `dinov3_sat.pth` |

Additionally, clone the DINOv2 hub code for local model loading:

```bash
git clone https://github.com/facebookresearch/dinov2.git model_weights/dinov2
```

The DINOv3 hub code is already included at `model_weights/dinov3/`.

### 5. Generate 1-channel variants

Convert DINOv3 weights from 3-channel to 1-channel input via patch embedding weight summation:

```bash
python 3to1.py
```

This generates `dinov3_lvd_1cha.pth` and `dinov3_sat_1cha.pth` in `model_weights/`.

You can also specify custom paths:

```bash
python 3to1.py --src model_weights/dinov3_lvd.pth --dst model_weights/dinov3_lvd_1cha.pth
```

---

## Usage

### Training (Linear Probe / Full Fine-tuning)

```bash
# Linear Probe
python train_net.py --dataset mb-domars16k --model dinov1_b8 --mode linear --lr 0.001 --strategy concat_3

# Full Fine-tuning
python train_net.py --dataset mb-domars16k --model dinov1_b8 --mode full --lr 0.0001 --strategy concat_3
```

**Arguments:**

| Arg | Description | Options |
|-----|-------------|---------|
| `--dataset` | Dataset name | `mb-domars16k`, `mb-frost_cls`, etc. |
| `--model` | Model key | `dinov1_b8`, `mars_mae`, `dinov3_lvd`, etc. |
| `--mode` | Training mode | `linear`, `full` |
| `--lr` | Learning rate | e.g., `0.001`, `0.0001`, `1e-5` |
| `--strategy` | Feature fusion | `concat_3` (CLS+Mean+Max), `concat_2` (CLS+Mean), `original` |
| `--epochs` | Max epochs | default: 20 |
| `--patience` | Early stopping | default: 5 (linear), 8 (full) |
| `--batch_size` | Batch size | default: 64 |
| `--data_fraction` | Data subset | `1.0x`, `0.50x`, `0.20x`, `0.05x`, `0.01x` |

Training logs are saved to `./logs/{mode}_records/`.

### KNN Evaluation

```bash
python eval_knn.py
```

Runs KNN (K=1,3,5,7,9,11,13,15) with 4 pooling modes (cls, mean, max, gem) across all 12 models × 6 datasets. Requires multi-GPU for speed (configurable via `GPU_IDS` in the script). Results are logged to `benchmark_results.log`.

### Data Efficiency Evaluation

```bash
python eval_generalization.py --gpus 0,1,2,3 --mode linear --epochs 50
```

Runs linear probing at 4 data fractions (1%, 5%, 20%, 50%) × 3 learning rates × all models × all datasets in parallel across GPUs.

### Attention Visualization

Generate comparison figures for the three experimental groups:

```bash
# Use existing candidate images
python visualize.py

# Or extract fresh samples from parquet
python visualize.py --extract
```

Output saved to `all_checkpoints/output_vis/`.

### Quantitative Attention Evaluation

```bash
# Run all three evaluation dimensions
python eval_attention.py

# Or run individually
python eval_attention.py --mode stats       # Heatmap statistics
python eval_attention.py --mode semantic    # Edge alignment
python eval_attention.py --mode coherence   # Spatial coherence
```

### Plot Publication Figures

```bash
cd all_checkpoints
python plot_results.py
```

Generates PDF + PNG figures in `all_checkpoints/output_figures/`.

---

## Results

Pre-computed results are included in `all_checkpoints/`:

- `knn_checkpoints/best_summary_knn.csv` — Best KNN F1 per model per dataset
- `train_all_logs/best_results_linear.csv` — Best linear probe accuracy
- `train_all_logs/best_results_full.csv` — Best full fine-tuning accuracy
- `logs_generalization/generalization_results_best.csv` — Best data efficiency results

---

## Project Structure

```
├── config_mars.py              # Model configurations and normalization constants
├── loader.py                   # Unified model loading (torchvision/timm/HuggingFace/hub)
├── train_net.py                # Training pipeline (linear probe & full fine-tuning)
├── eval_knn.py                 # KNN evaluation (GPU-accelerated)
├── eval_generalization.py      # Data efficiency evaluation (multi-GPU parallel)
├── eval_attention.py           # Quantitative attention evaluation
├── visualize.py                # Attention visualization & comparison plots
├── 3to1.py                     # 3-channel to 1-channel weight conversion
├── requirements.txt            # Python dependencies
├── all_checkpoints/            # Results, logs, and figures
│   ├── knn_checkpoints/        # KNN results
│   ├── train_all_logs/         # Training logs and summaries
│   ├── logs_generalization/    # Data efficiency logs
│   ├── output_figures/         # Publication figures
│   ├── output_vis/             # Visualization comparison images
│   └── plot_results.py         # Figure generation script
├── candidate_images/           # Sample images for visualization
├── dataset/                    # Dataset metadata (parquet data not included)
└── model_weights/
    └── dinov3/                 # DINOv3 hub code (for local loading)
```

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@thesis{zhou2026mars,
  title={Cross-Domain Feature Extraction and Adaptive Foundation Models for Mars Terrain Classification},
  author={Zhou, Zhibin},
  year={2026},
  school={Southern University of Science and Technology}
}
```

## License

This project is for academic research purposes. Model weights are subject to their respective licenses (Meta for DINO family, Google for SigLIP2, LAION for OpenCLIP, BAAI for Mars-MAE/MarsBench).
