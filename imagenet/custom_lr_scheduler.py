import warnings
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CosLR(_LRScheduler):
    def __init__(self, optimizer, base_learning_rate, lr_warm_epochs, num_train_epochs, last_epoch=-1, verbose=False):
        self.base_learning_rate = base_learning_rate
        self.lr_warm_epochs = lr_warm_epochs
        self.num_train_epochs = num_train_epochs
        self.cos_decay_epochs = num_train_epochs - lr_warm_epochs
        super(CosLR, self).__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        epoch = self.last_epoch
        if epoch < self.lr_warm_epochs:
            lr_multiplier = epoch / self.lr_warm_epochs
        else:
            lr_multiplier = 0.5 + 0.5 * math.cos(math.pi * (epoch - self.lr_warm_epochs) / self.cos_decay_epochs) 
            # lr_multiplier = 0.5 * (1 + math.cos(math.pi * (epoch - self.lr_warm_epochs) / self.cos_decay_epochs))
        
        return [base_lr * lr_multiplier for base_lr in self.base_lrs]

class FixedLR(_LRScheduler):
    def __init__(self, optimizer, base_learning_rate, lr_warm_epochs, last_epoch=-1, verbose=False):
        self.base_learning_rate = base_learning_rate
        self.lr_warm_epochs = lr_warm_epochs
        super(FixedLR, self).__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        epoch = self.last_epoch
        if epoch < self.lr_warm_epochs:
            lr_multiplier = epoch / self.lr_warm_epochs
        else:
            lr_multiplier = 1.0
        
        return [base_lr * lr_multiplier for base_lr in self.base_lrs]