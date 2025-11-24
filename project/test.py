import torch

def test(model, device, data_loader):
    """Evaluate model and return accuracy (percentage)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            # batch can be (images, labels) or (images, labels, probability)
            if len(batch) == 3:
                images, labels, probability = batch
                # Pass all three arguments to match model signature
                images = images.to(device)
                labels = labels.to(device)
                probability = torch.tensor(probability).to(device) if not isinstance(probability, torch.Tensor) else probability.to(device)
                outputs = model(images, percent=None, probability=probability)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        return 0.0
    return 100.0 * correct / total
