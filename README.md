# Mars-Foundation-Models

## Introduction

<!-- TODO: Add an introduction to the project, its goals, and research context. -->

This repository contains code and resources for the graduating thesis:
"Systematic Evaluation of Vision Foundation Models for Mars Terrain Classification".
It evaluates self-supervised models (DINO, MAE) and VLMs on the Mars-Bench dataset,
and explores cross-domain transfer learning from Earth to Mars.

---

## Dataset

<!-- TODO: Describe the Mars-Bench dataset, its structure, classes, and how to obtain/prepare it. -->

Place the Mars-Bench dataset files inside the `data/` directory.

---

## Models

<!-- TODO: List and describe the Vision Transformer models (e.g. DINO, MAE) used in this project. -->

Pre-trained model weights should be placed inside the `models/` directory.

---

## Requirements

Install all Python dependencies with:

```bash
pip install -r requirements.txt
```

Key dependencies:

- `torch` — deep learning framework
- `torchvision` — computer vision utilities for PyTorch
- `transformers` — Hugging Face model hub and Vision Transformers
- `scikit-learn` — evaluation metrics and classical ML tools
- `numpy` — numerical computing
- `pandas` — data manipulation
- `matplotlib` — visualisation

