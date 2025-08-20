from abc import ABC, abstractmethod
import torch

class Metrics(ABC):
    """Abstract base class for evaluation metrics."""

    def __init__(self, y_true, y_pred):
        """Initialize with ground truth and predicted values."""

        self.y_true = y_true
        self.y_pred = y_pred

    @abstractmethod
    def calculate(self):
        pass
        """
        Abstract method to calculate the metric.
        Must be implemented by concrete classes.
        
        Returns:
            float: The calculated metric value
    """

class Dice(Metrics):
    """ 
        Args:
            smooth (float): Smoothing factor to avoid division by zero
            
        Returns:
            float: Dice coefficient value between 0 and 1
        """
    def calculate(self, smooth=1e-6):
        y_true_f = torch.tensor(self.y_true, dtype=torch.float32).flatten()
        y_pred_f = torch.tensor(self.y_pred, dtype=torch.float32).flatten()
        intersection = torch.sum(y_true_f * y_pred_f)
        total = torch.sum(y_true_f) + torch.sum(y_pred_f)
        dice_score = (2. * intersection + smooth) / (total + smooth)
        return dice_score
    
class IOU(Metrics):
    """
        Args:
            smooth (float): Smoothing factor to avoid division by zero
            
        Returns:
            float: IoU value between 0 and 1
        """
    def calculate(self, smooth=1e-6):
        y_true_f = torch.tensor(self.y_true, dtype=torch.float32).flatten()
        y_pred_f = torch.tensor(self.y_pred, dtype=torch.float32).flatten()
        intersection = torch.sum(y_true_f * y_pred_f)
        union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection
        iou_score = (intersection + smooth) / (union + smooth)
        return iou_score 