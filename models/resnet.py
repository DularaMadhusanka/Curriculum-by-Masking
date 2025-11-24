import torch
import torchvision
import torch.nn as nn
from einops import rearrange

class ResNet18(nn.Module):
    """Wrapper around torchvision's resnet18 with curriculum masking support."""
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet18, self).__init__()
        self.net = torchvision.models.resnet18(pretrained=pretrained)
        # Replace final fc layer
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x, percent=None, probability=None):
        # Apply curriculum masking if both percent and probability provided
        if percent is not None and probability is not None and percent > 0:
            x = self._apply_curriculum_mask(x, percent, probability)
        return self.net(x)
    
    def _apply_curriculum_mask(self, x, percent, probability):
        B, C, H, W = x.shape
        
        # 1. Setup Patches (Divide image into 4x4 patches)
        # Shape: (B, 16, C, patch_size, patch_size)
        x_patches = rearrange(x, 'b c (p1 h) (p2 w) -> b (p1 p2) c h w', p1=4, p2=4)
        
        # 2. Process Probabilities
        if not isinstance(probability, torch.Tensor):
            probability = torch.tensor(probability, device=x.device, dtype=torch.float32)
        
        # Handle dimensions
        if probability.dim() == 1: 
            probability = probability.unsqueeze(0).repeat(B, 1)
        elif probability.dim() == 3:
            probability = probability.mean(dim=-1)
        
        # Ensure probability is (B, 16)
        if probability.shape[-1] != 16:
            probability = probability[:, :16]

        # Normalize so they sum to 1 (for Categorical sampling)
        probability = probability / (probability.sum(dim=1, keepdim=True) + 1e-8)
        
        # 3. Calculate Mask Ratio
        # Since arguments.py passes the exact ratio (e.g., 0.1, 0.2, ... 0.4), 
        # we use it directly.
        mask_ratio = percent 
        
        # Safety clamp
        if isinstance(mask_ratio, torch.Tensor):
             mask_ratio = mask_ratio.item()
        mask_ratio = min(max(mask_ratio, 0.0), 1.0)
        
        # 4. Determine number of patches to mask (0 to 16)
        n_mask = int(16 * mask_ratio)
        
        # 5. Apply Mask
        masked_x_patches = x_patches.clone()
        
        # Optimization: Vectorize this loop if possible, but loop is fine for batch size 64-128
        for b in range(B):
            if n_mask > 0:
                try:
                    # Weighted Lottery: High prob patches get picked to be masked
                    dist = torch.distributions.Categorical(probability[b])
                    patch_indices = dist.sample((n_mask,)) 
                    masked_x_patches[b, patch_indices] = 0
                except:
                    pass # Fallback if sampling fails (very rare)
        
        # 6. Reconstruct image from patches
        x = rearrange(masked_x_patches, 'b (p1 p2) c h w -> b c (p1 h) (p2 w)', p1=4, p2=4)
        
        return x
