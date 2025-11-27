import torch
import numpy as np

class DiceLoss(torch.nn.Module):
    def __init__(self, num_classes, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes
    
    def forward(self, preds, targets):

        assert isinstance(preds, torch.Tensor), "'preds' must be a PyTorch tensor"
        assert isinstance(targets, torch.Tensor), "'targets' must be a PyTorch tensor"

        ## convert raw logits to probabilities
        preds = torch.nn.functional.softmax(preds, dim=1)

        ## extracting classes of targets to convert from shape [B,H,W] to [B,C,H,W]
        temp_onehot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
        
        # Move class dimension to channel position: [..., C] → [N, C, ...]
        dims = list(range(temp_onehot.ndim))
        dims = [0, temp_onehot.ndim - 1] + dims[1:-1]
        targets = temp_onehot.permute(*dims).float()
        del temp_onehot

        assert preds.shape == targets.shape, f"Predictions({preds.shape}) and targets({targets.shape}) must have the same shape"

        intersection = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)
        mean_dice = dice.mean()
        return (1 - mean_dice)
