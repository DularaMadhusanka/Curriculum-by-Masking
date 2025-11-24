import argparse
import torch
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="CBM Training Arguments")
    
    # --- Paths & Selection ---
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
    parser.add_argument('--model_name', type=str, default='resnet18', choices=['resnet18'])
    
    # --- Hyperparameters (Required by Trainer) ---
    parser.add_argument('--lr', type=float, default=0.01) # High initial LR for SGD often used in papers
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=200, help="Total training epochs")
    
    # --- Learning Rate Decay (Required by Trainer logic) ---
    parser.add_argument('--decay_epoch', type=int, default=50, help="Start decaying LR here")
    parser.add_argument('--stop_decay_epoch', type=int, default=150, help="Stop decaying LR here")
    parser.add_argument('--decay_step', type=int, default=50, help="Decay LR every X epochs")
    
    # --- CBM Specifics ---
    parser.add_argument('--max_mask_ratio', type=float, default=0.4, help="Max difficulty (0.4 = 40% masked)")
    parser.add_argument('--schedule_type', type=str, default='linear_repeat', choices=['linear', 'linear_repeat', 'constant'])

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # INJECT THE SCHEDULE (args.percent)
    # The Trainer expects a list args.percent where percent[epoch] = difficulty
    # ---------------------------------------------------------
    args.percent = []
    
    if args.schedule_type == 'constant':
        # Baseline: Always constant masking (e.g. 0.4)
        args.percent = [args.max_mask_ratio] * args.num_epochs

    elif args.schedule_type == 'linear':
        # Standard Linear: 0 -> 0.4 over all epochs
        for epoch in range(args.num_epochs):
            ratio = (epoch / args.num_epochs) * args.max_mask_ratio
            args.percent.append(ratio)

    elif args.schedule_type == 'linear_repeat':
        # THE WINNER: Sawtooth Pattern (0 -> 0.4, reset every 20 epochs)
        repeat_interval = 20
        for epoch in range(args.num_epochs):
            # Modulo math creates the repeating cycle
            progress_in_cycle = (epoch % repeat_interval) / repeat_interval
            ratio = progress_in_cycle * args.max_mask_ratio
            args.percent.append(ratio)

    return args

# Quick test to verify
if __name__ == "__main__":
    args = get_args()
    print(f"Device: {args.device}")
    print(f"Total Epochs: {args.num_epochs}")
    print(f"Schedule Length: {len(args.percent)}")
    print(f"Sample Schedule (First 25 epochs): { [round(x, 2) for x in args.percent[:25]] }")