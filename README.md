# XAI-Guided Deep Learning for Interpretable Rice Leaf Disease Classification

Code, data split and complete results for the manuscript

> Jain, S., Agrawal, K., Gharibian, L., & Yogarayan, S. *Advancing Food Security: XAI-Guided Deep Learning for Interpretable Rice Leaf Disease Classification*. Journal of Computational and Cognitive Engineering (revised manuscript, R1).

The study trains seven CNNs on the Paddy Doctor training release, explains ResNet50 with Grad-CAM, and tests Top Losses data cleaning against a random-removal control under a split that keeps exact and near-duplicate images out of the test set.

## Data

The images are **not** included in this repository. Download them from the Kaggle competition
**Paddy Doctor: Paddy Disease Classification**:
https://www.kaggle.com/competitions/paddy-disease-classification/data

Only the training release is used (`train_images/`, 10,407 images, 10 classes). The file `results/split_manifest.csv` lists every image used, with its class, partition (train / val / test), MD5 hash and near-duplicate group, so the exact split can be rebuilt.

## Two notebooks

| Notebook | Split | Results folder | Used in the paper for |
|---|---|---|---|
| `Rice_XAI_Revision_Experiments.ipynb` (v2, final) | Group-aware: exact and near-duplicate images kept within one partition | `results/` | All reported results |
| `Rice_XAI_Experiments_v1_image_level_split.ipynb` (v1) | Image-level stratified split after exact de-duplication; near-duplicates only reported | `results_image_level_split/` | Leakage comparison only (Section 4.3, Table 8) |

Apart from the split, the two notebooks use the same models, settings and seeds.

## How to reproduce

1. Open a new Kaggle notebook and import `Rice_XAI_Revision_Experiments.ipynb` (or the v1 notebook for the leakage comparison).
2. Settings: Accelerator **GPU T4**, Internet **ON**.
3. Add Input → Competitions → *Paddy Doctor: Paddy Disease Classification* (join the competition first).
4. Optional smoke test: set `QUICK_TEST = True` in the first code cell and run all (about 10 minutes).
5. Full run: set `QUICK_TEST = False` and use *Save Version → Save & Run All (Commit)*. The run takes about 2.5 hours on one T4.
6. Download `results_pack.zip` from the Output tab.

All settings (seeds, epochs, batch size, learning rate, number of removed images) are in the first code cell. GPU runs are seeded but not bit-exact, so repeated runs can differ by a few test images.

## Repository contents

| Path | Content |
|---|---|
| `Rice_XAI_Revision_Experiments.ipynb` | Final pipeline (v2): provenance, de-duplication, group-aware split, training of seven models, Top Losses and random-removal experiment, statistics, Grad-CAM, figures |
| `Rice_XAI_Experiments_v1_image_level_split.ipynb` | Same pipeline with an image-level split (v1) |
| `results/` | Results reported in the paper (group-aware split) |
| `results/split_manifest.csv` | Image identifier, class, partition, MD5, dHash and near-duplicate group of all 10,333 unique images |
| `results/duplicates_found.csv` | The 72 groups of exact duplicates (74 removed files) |
| `results/near_duplicate_pairs.csv` | The 2,451 near-duplicate pairs (dHash distance ≤ 4), including the 107 pairs with conflicting labels |
| `results/table*.csv` | Tables of the paper (model comparison, latency, paired tests, per-class metrics, refinement arms) |
| `results/top_losses_removed_images.csv` | The images removed by Top Losses in every seed |
| `results/runs/` | Per-run predictions on every test image, confusion matrices (`.npy`), per-class metrics and training histories |
| `results/figures/` | All figures as PNG |
| `results/run_summary.json` | Key numbers, configuration and software versions |
| `results_image_level_split/` | Output of the v1 notebook (image-level split; 1,019 near-duplicate pairs crossed partitions), used only for the leakage comparison in Section 4.3 / Table 8 |

## Main results (test set, n = 1,550)

| ResNet50 arm (5 seeds) | Mean correct | Accuracy |
|---|---|---|
| Baseline | 1,471.8 | 94.95% ± 0.73% |
| Top Losses removal (k = 5) | 1,482.4 | 95.64% ± 0.38% |
| Random removal (k = 5) | 1,479.0 | 95.42% ± 0.47% |

Top Losses removal did not outperform random removal (Wilcoxon p = 0.50). See the paper for the full analysis.

## Environment

Kaggle, NVIDIA Tesla T4, Python 3.13, PyTorch 2.11, torchvision 0.26, timm 1.0.30 (see `requirements.txt`).

## License

Code: MIT (see `LICENSE`). The images remain subject to the terms of the Kaggle competition and the Paddy Doctor dataset.
