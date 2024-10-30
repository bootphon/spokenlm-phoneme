import math

import torch
from torch.optim.optimizer import Optimizer


# From torchaudio: https://github.com/pytorch/audio/blob/b7b7b5d79a4630840ed32fef86f6d6e2d906f2b0/examples/hubert/lightning_modules.py#L53
# BSD 2-Clause License. Copyright (c) 2017 Facebook Inc. (Soumith Chintala). All rights reserved.
class TriStageLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear learning rate scheduler with warmup, hold, and decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        hold_updates: int,
        decay_updates: int,
        init_lr_scale: float = 0.01,
        final_lr_scale: float = 0.05,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_updates = warmup_updates
        self.hold_updates = hold_updates
        self.decay_updates = decay_updates
        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale

        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            return [
                base_lr * (self.init_lr_scale + self._step_count / self.warmup_updates * (1 - self.init_lr_scale))
                for base_lr in self.base_lrs
            ]
        elif self.warmup_updates < self._step_count <= (self.warmup_updates + self.hold_updates):
            return list(self.base_lrs)
        elif self._step_count <= (self.warmup_updates + self.hold_updates + self.decay_updates):
            return [
                base_lr
                * math.exp(
                    math.log(self.final_lr_scale)
                    * (self._step_count - self.warmup_updates - self.hold_updates)
                    / self.decay_updates
                )
                for base_lr in self.base_lrs
            ]
        else:
            return [base_lr * self.final_lr_scale for base_lr in self.base_lrs]
