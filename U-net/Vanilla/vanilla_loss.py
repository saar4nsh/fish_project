import torch
import torch.nn as nn
import torch.nn.functional as F
from dice import DiceLoss

class VanillaLoss(nn.Module):
    def __init__(self, num_class, epsilon=1e-6):
        super(VanillaLoss, self).__init__()
        self.dice = DiceLoss(epsilon=epsilon, num_classes=num_class)
        self.CE = nn.CrossEntropyLoss()
    
    def forward(self, preds, targets):
        # preds: [N, C, H, W], raw logits
        # targets: [N, H, W], class indices (LongTensor)
        assert preds.dim() == 4, "Predictions should be of shape [N, C, H, W]"
        assert targets.dim() == 3, "Targets should be of shape [N, H, W]"
        
        ce_loss = self.CE(preds, targets)
        dice_loss = self.dice(preds, targets)
        
        return (dice_loss + ce_loss, dice_loss, ce_loss)
