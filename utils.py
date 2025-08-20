from abc import ABC, abstractmethod
import numpy as np

class Metrics(ABC):
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    @abstractmethod
    def calculate(self):
        pass

class Dice(Metrics):
    def calculate(self, smooth=1e-6):
        y_true_f = np.array(self.y_true).flatten()
        y_pred_f = np.array(self.y_pred).flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        total = np.sum(y_true_f) + np.sum(y_pred_f)
        dice_score = (2. * intersection + smooth) / (total + smooth)
        return dice_score
    
class IOU(Metrics):
    def calculate(self, smooth=1e-6):
        y_true_f = np.array(self.y_true).flatten()
        y_pred_f = np.array(self.y_pred).flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
        iou_score = (intersection + smooth) / (union + smooth)
        return iou_score 