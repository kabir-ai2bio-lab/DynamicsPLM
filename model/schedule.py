import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from config import config


class WarmupDecaySchedule(LambdaLR):
    """Linear warmup → linear decay → fixed floor at 10% of peak_lr."""

    def __init__(self, optimizer: Optimizer, model_select):
        self.warmup_steps = config['optimizer']['warmup_step']
        self.peak_lr      = config['optimizer']['peak_lr']
        self.total_steps  = config[model_select]['total_update']
        self.decay_steps  = int(self.total_steps * config['optimizer']['decay_step_percent'])
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            decay_factor = float(step) / float(self.warmup_steps)
        elif step < self.decay_steps:
            decay_factor = 1 - 0.9 * (
                float(step - self.warmup_steps) /
                float(self.decay_steps - self.warmup_steps)
            )
        else:
            decay_factor = 0.1
        return decay_factor * self.peak_lr
