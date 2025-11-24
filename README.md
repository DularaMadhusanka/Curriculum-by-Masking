CBM: Curriculum by Masking (Unofficial PyTorch Implementation)

This repository contains a PyTorch reproduction of the research paper "CBM: Curriculum by Masking" (Jarcă et al., 2024).

It implements a novel Curriculum Learning strategy that creates an "easy-to-hard" training schedule by masking image patches based on their Gradient Magnitude (Salience).

📄 Paper Summary

Curriculum by Masking (CBM) improves model generalization by:

Gradient-Based Masking: Calculating image gradients to identify salient regions (e.g., edges, objects) and masking them probabilistically.

Curriculum Schedule: Gradually increasing the masking ratio using a "Linear Repeat" (sawtooth) schedule to prevent catastrophic forgetting.

🚀 Features implemented

[x] Gradient Saliency Calculation: Pre-computes patch importance using Sobel filters.

[x] CBM Masking Engine: Custom ResNet-18 wrapper that applies probabilistic masking during the forward pass.

[x] Linear Repeat Schedule: Implements the Fibonacci-based sawtooth scheduling logic.

[x] Comprehensive Evaluation: Generates Training Curves, Confusion Matrices, and ROC/AUC plots automatically.

📂 Project Structure

.
├── main.py                # Entry point for the experiment runner
├── arguments.py           # Argument parsing and Schedule injection
├── runs.py                # Configuration registry mapping models to datasets
├── resnet_experiments.py  # Main experiment logic (Hyperparameters & Metrics)
├── resnet_train.py        # The Trainer class (Training loop & Logging)
├── data_handlers.py       # CIFAR dataset class with Gradient Calculation
├── fibonacci.py           # Helper for generating the schedule
├── test.py                # Evaluation loop
└── models/
    └── resnet.py          # ResNet-18 wrapper with CBM Masking Logic


🛠️ Installation

Clone the repository:

git clone [https://github.com/your-username/cbm-reproduction.git](https://github.com/your-username/cbm-reproduction.git)
cd cbm-reproduction


Install the required dependencies:

pip install torch torchvision numpy opencv-python matplotlib seaborn scikit-learn tqdm einops


🏃 Usage

To run the reproduction experiment on CIFAR-10 with ResNet-18:

python main.py


This command will:

Download CIFAR-10 (if not present).

Pre-compute gradient probabilities for the training set.

Train ResNet-18 for 100 epochs using the Linear Repeat schedule.

Generate evaluation plots in the plots/ directory.

Custom Arguments

You can modify hyperparameters in arguments.py or pass them via command line (if extended):

# Example (if arguments are exposed in main.py)
python main.py --model_name resnet18 --dataset cifar10


📊 Results & Outputs

After training, the script generates the following visualizations in the plots/ folder:

training_curves.png: Loss and Accuracy over epochs.

confusion_matrix.png: Heatmap of predicted vs. true classes.

roc_auc_curves.png: Multi-class ROC curves with AUC scores.

The best model checkpoint is saved to saved_models/r18_cif10_100ep.pth.

📜 Citation

This code is a reproduction based on the original work:

@article{jarca2024cbm,
  title={CBM: Curriculum by Masking},
  author={Jarcă, Andrei and Croitoru, Florinel-Alin and Ionescu, Radu Tudor},
  journal={arXiv preprint arXiv:2407.05193},
  year={2024}
}
