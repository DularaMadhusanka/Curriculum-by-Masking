import resnet_experiments

# Configuration dictionary mapping model names and datasets to functions
# New experiments can be added here easily
RUNS = {
    'resnet18': {
        'cifar10': resnet_experiments.train_cifar10
    }
}