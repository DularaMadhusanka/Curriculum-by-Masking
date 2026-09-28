# CBM: Curriculum by Masking (Unofficial PyTorch Implementation)

[![Paper](https://img.shields.io/badge/arXiv-2407.05193-B31B1B.svg?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2407.05193)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

An unofficial PyTorch reproduction of the research paper **"CBM: Curriculum by Masking"** (*Jarcă et al., 2024*).

This repository implements a Curriculum Learning strategy designed to improve model generalization. It builds an "easy-to-hard" training progression by dynamically masking image patches based on their spatial saliency (gradient magnitude).

---

## 📖 Paper Summary

**Curriculum by Masking (CBM)** enhances image classifier training through two core mechanisms:

1. **Gradient-Based Saliency Masking:** Calculates local image gradients to locate salient features (e.g., key edges, object boundaries) and probabilistically masks high-saliency patches during training.
2. **Linear Repeat Schedule:** Gradually increases the masking ratio using a Fibonacci-inspired sawtooth pattern. This progressive difficulty prevents catastrophic forgetting while pushing the network to learn robust contextual features.

---

## ✨ Key Features

- [x] **Saliency Pre-Computation:** Fast patch-importance generation using Sobel gradient filtering.
- [x] **CBM ResNet Wrapper:** Custom `ResNet-18` architecture with integrated probabilistic masking layers in the forward pass.
- [x] **Sawtooth Curriculum Scheduler:** Modular Fibonacci-based linear repeat schedule logic.
- [x] **Automated Evaluation Suite:** Generates training curves, confusion matrices, and multi-class ROC/AUC plots automatically post-training.

---

## 📂 Repository Structure

```text
.
├── main.py                 # Primary entry point to launch experiments
├── arguments.py            # CLI argument parsing and schedule injection
├── runs.py                 # Registry mapping model architectures to datasets
├── resnet_experiments.py   # Experiment coordination, hyperparameter setup, and metrics
├── resnet_train.py        # Trainer class handling the training loop and logging
├── data_handlers.py        # Dataset wrappers with pre-computed gradient saliency
├── fibonacci.py            # Helper module for generating Linear Repeat schedules
├── test.py                 # Independent evaluation script
└── models/
    └── resnet.py           # ResNet-18 model integrated with CBM masking logic
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DularaMadhusanka/Curriculum-by-Masking.git
   cd Curriculum-by-Masking
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision numpy opencv-python matplotlib seaborn scikit-learn tqdm einops
   ```

---

## 🚀 Quickstart & Usage

Run the baseline CIFAR-10 training run with ResNet-18:

```bash
python main.py
```

### Execution Workflow
1. Downloads the CIFAR-10 dataset (if not available locally).
2. Pre-computes Sobel gradient probabilities across the training dataset.
3. Trains `ResNet-18` for 100 epochs using the CBM Linear Repeat schedule.
4. Saves figures to `plots/` and weights to `saved_models/`.

### Custom Arguments
Extend or override hyperparameters via command line arguments:

```bash
python main.py --model_name resnet18 --dataset cifar10
```

---

## 📊 Results & Artifacts

Upon completion, all diagnostic plots are automatically saved to the `plots/` directory:

| Artifact | Description |
| :--- | :--- |
| `plots/training_curves.png` | Epoch-wise train/validation loss and accuracy trajectories. |
| `plots/confusion_matrix.png` | Normalized heatmaps for true vs. predicted class distributions. |
| `plots/roc_auc_curves.png` | One-vs-Rest multi-class ROC curves and area-under-curve metrics. |
| `saved_models/r18_cif10_100ep.pth` | Checkpoint containing model weights and optimizer state. |

---

## 📜 Citation

If you use this reproduction or reference the original paper, please cite:

```bibtex
@article{jarca2024cbm,
  title   = {CBM: Curriculum by Masking},
  author  = {Jarc{\u{a}}, Andrei and Croitoru, Florinel-Alin and Ionescu, Radu Tudor},
  journal = {arXiv preprint arXiv:2407.05193},
  year    = {2024}
}
```
