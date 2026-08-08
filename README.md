# Weed Detection in Smart Agriculture
### A Comparative Analysis of Machine Learning, CNNs, and Vision Transformers

**Course:** ML/DL Research Paper Assignment — Bells University of Technology
**Team:** [Add names here]

---

## Problem Statement

Weed identification is a bottleneck for automated/precision weed control in smart agriculture. This
project trains and rigorously compares three modeling paradigms — a classical ML baseline, a CNN
baseline, and a custom Vision Transformer / hybrid CNN-ViT model — on multi-species weed
classification, and evaluates how well each paradigm generalizes to a second, more species-diverse
dataset it was never trained on.

## Datasets

| Dataset | Role | Species | Size |
|---|---|---|---|
| [DeepWeeds](https://github.com/AlexOlsen/DeepWeeds) | Primary (train/val/test) | 8 weed species + negative | 17,509 images |
| [MFWD](https://doi.org/10.5281/zenodo.7598372) | Secondary (cross-dataset eval only, no training) | 28 weed species | 94,321 images |

Data is **not** committed to this repo (see `.gitignore`). Run `src/data/download.py` to fetch both
datasets into `data/` (script to be filled in during Phase 2).

## Repository Structure

```
weed-detection-project/
├── configs/              # YAML configs — one per experiment, hyperparameter search spaces
├── data/                 # Downloaded datasets (gitignored, not committed)
├── notebooks/            # EDA and exploratory analysis only — no model training here
├── results/
│   ├── figures/          # All plots (loss curves, confusion matrices, comparison charts)
│   └── tables/           # All result tables as CSV
├── src/
│   ├── data/              # Download, preprocessing, augmentation, dataset splitting
│   ├── models/
│   │   ├── classical_ml/  # Feature extraction + RF/SVM (Person B)
│   │   ├── cnn/            # ResNet/EfficientNet baseline (Person B)
│   │   └── vit/             # Custom ViT / hybrid CNN-ViT (Person C)
│   ├── utils/              # Shared metrics, CV protocol, plotting helpers
│   ├── train.py            # Single entry point — trains whichever model the config specifies
│   └── evaluate.py         # Runs Phase 7 comparison + cross-dataset generalization test
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing Results

Every experiment runs from a single command, per the assignment's reproducibility requirement:

```bash
# Classical ML baseline
python -m src.train --config configs/classical_ml.yaml

# CNN baseline
python -m src.train --config configs/cnn.yaml

# Custom ViT / hybrid model
python -m src.train --config configs/vit_custom.yaml

# Full comparison + cross-dataset generalization test (Phase 7)
python -m src.evaluate --config configs/evaluate.yaml
```

## Team Roles & Contribution

| Member | Role | Responsible for |
|---|---|---|
| [Name A] | Team Lead / Literature Reviewer | Intro, Related Work, project coordination |
| [Name B] | Implementation Lead | `src/models/classical_ml/`, `src/models/cnn/`, Methodology, Results |
| [Name C] | Custom Model / Visualisation | `src/models/vit/`, ablations, `results/figures/`, Discussion |

See `docs/literature_assignment_plan.md` and `docs/project_execution_algorithm.md` for the full
reading list and phase-by-phase execution plan.

## Status

- [x] Phase 0: Task type + datasets decided
- [ ] Phase 1: Literature review
- [ ] Phase 2: Data acquisition & EDA
- [ ] Phase 3: Preprocessing pipeline
- [ ] Phase 4-6: Baseline + custom models
- [ ] Phase 7-8: Comparison, statistical tests, ablations
- [ ] Phase 9-12: Visualization, write-up, packaging, submission

## Academic Integrity Note

AI tools were used for project planning and code scaffolding (per Section 8, Rule 6 of the assignment
handbook). All model implementation, experimentation, and analysis are original team work. Full
disclosure will be included in the final manuscript's Methodology section.
