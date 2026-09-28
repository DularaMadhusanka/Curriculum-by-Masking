import copy
import math
import os
import numpy as np
import torch
import torchvision
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize

import arguments
from data_handlers import get_train_val_loaders_cifar, get_test_loader_cifar
from models.resnet import ResNet18
from resnet_train import Trainer
from fibonacci import fibonacci

# --- 1. User Defined Optimizers & Schedules ---

def build_optimizer_resnet(model):
    return optim.SGD(model.parameters(), lr=0.1,
                               weight_decay=5e-4,
                               momentum=0.9)

def build_optimizer_resnet_tin(model):
    return optim.SGD(model.parameters(), lr=0.1,
                               weight_decay=1e-4,
                               momentum=0.9)

def baseline(num_epochs):
    return [0] * num_epochs

def lin_repeat():
    """Fibonacci-based Linear Repeat Schedule"""
    v = fibonacci(length=7)
    for i in range(1, len(v)):
        v[i] = math.log(v[i]) / (math.log(v[6]) / 0.4036067977500615)
    v = v[2:]
    v[0] = 0.07
    return v

def lin_repeat_tin():
    v = fibonacci(length=7)
    for i in range(1, len(v)):
        v[i] = math.log(v[i]) / (math.log(v[6]) / 0.6036067977500615)
    v = v[2:]
    v[0] = 0.1
    return v

# --- 2. Visualization & Evaluation Functions ---

def plot_training_curves(trainer, save_path="plots"):
    """Plots Loss and Accuracy curves from the Trainer history."""
    os.makedirs(save_path, exist_ok=True)
    
    # Ensure we have data to plot
    if not trainer.train_losses:
        print("No training history found. Skipping plots.")
        return

    epochs = range(1, len(trainer.train_losses) + 1)

    plt.figure(figsize=(14, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, trainer.train_losses, 'b-o', label='Training Loss')
    plt.title('Training Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, trainer.train_accuracies, 'g-o', label='Train Acc')
    plt.plot(epochs, trainer.val_accuracies, 'r--s', label='Val Acc')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_curves.png"))
    print(f"Training curves saved to {save_path}/training_curves.png")
    plt.close()

def evaluate_performance(model, device, test_loader, classes):
    """
    Computes detailed metrics: Confusion Matrix, ROC/AUC, and Classification Report.
    """
    print("\n--- Starting Final Evaluation ---")
    model.eval()
    y_true = []
    y_pred = []
    y_score = []  # Probabilities for AUC

    with torch.no_grad():
        for data in test_loader:
            # Handle CBM loader (images, labels, probs) vs Standard (images, labels)
            if len(data) == 3:
                images, labels, _ = data
            else:
                images, labels = data
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) # Inference (No masking)
            
            # Get probabilities via Softmax
            probs = F.softmax(outputs, dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    # 1. Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/confusion_matrix.png")
    print("Confusion matrix saved to plots/confusion_matrix.png")
    plt.close()

    # 3. ROC Curve & AUC
    plot_multiclass_roc(y_true, y_score, classes)

def plot_multiclass_roc(y_true, y_score, classes, save_path="plots"):
    """Plots ROC curves for each class."""
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    # Use a distinct color map
    colors = plt.cm.get_cmap('tab10', n_classes)

    for i in range(n_classes):
        color = colors(i)
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{classes[i]} (AUC = {roc_auc[i]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_path, "roc_auc_curves.png"))
    print(f"ROC curves saved to {save_path}/roc_auc_curves.png")
    plt.close()

# --- 3. Main Experiment Function ---

def train_cifar10(args):
    # User Config
    args.num_classes = 10
    args.decay_epoch = 32
    args.decay_step = 30
    args.stop_decay_epoch = 93
    args.num_epochs = 100
    args.model_name = 'r18_cif10_100ep'
    
    # Schedule Generation
    curriculum = lin_repeat()
    # Pattern length is 5. 100 epochs / 5 = 20 repeats.
    args.percent = curriculum * 20
    
    print("Linear repeat schedule for CIFAR10:")
    print(args.percent[:15]) # Print first few to verify
    print(f"Schedule length: {len(args.percent)}")
    
    # Model & Data Setup
    resnet18 = ResNet18(num_classes=args.num_classes)
    
    # Use larger validation set (5000) for more reliable accuracy
    train_loader, val_loader = get_train_val_loaders_cifar(val_size=5000, dataset=torchvision.datasets.CIFAR10)
    test_loader = get_test_loader_cifar(dataset=torchvision.datasets.CIFAR10)
    
    args.testlo = test_loader
    
    # Training
    trainer = Trainer(resnet18, train_loader, val_loader, args, build_optimizer_resnet)
    trainer.train()
    
    # --- Final Evaluation ---
    print("\nTraining Complete. Generating plots...")
    
    # 1. Plot Loss and Accuracy over time
    plot_training_curves(trainer)
    
    # 2. Detailed Performance Analysis
    cifar_classes = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    evaluate_performance(trainer.model, trainer.device, test_loader, cifar_classes)