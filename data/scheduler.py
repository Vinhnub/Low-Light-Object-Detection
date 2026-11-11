import math
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler


def one_cycle(y1: float = 0.0, y2: float = 1.0, steps: int = 100):
    """
    Lambda function for sinusoidal ramp from y1 to y2 (YOLOv5 OneCycle cosine schedule).
    Usage:
        lf = one_cycle(1, lrf, epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    """
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up (increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater than or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.total_epoch <= 0 or self.last_epoch >= self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch + 1) / float(self.total_epoch + 1))
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * (self.last_epoch + 1) / float(self.total_epoch + 1) + 1.0)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / max(1, self.total_epoch) + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list."""
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i
    return len(cumulative_period) - 1


class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """Cosine annealing with cyclic restarts learning rate scheme."""

    def __init__(
        self,
        optimizer,
        periods,
        restart_weights=(1,),
        eta_mins=(0,),
        last_epoch=-1,
    ):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        assert len(self.periods) == len(
            self.restart_weights
        ), "periods and restart_weights should have the same length."
        self.cumulative_period = [
            sum(self.periods[0 : i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        return [
            eta_min
            + current_weight
            * 0.5
            * (base_lr - eta_min)
            * (
                1
                + math.cos(
                    math.pi * ((self.last_epoch - nearest_restart) / current_period)
                )
            )
            for base_lr in self.base_lrs
        ]


class CosineAnnealingRestartLR(_LRScheduler):
    """Cosine annealing with restarts learning rate scheme."""

    def __init__(
        self,
        optimizer,
        periods,
        restart_weights=(1,),
        eta_min=0,
        last_epoch=-1,
    ):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert len(self.periods) == len(
            self.restart_weights
        ), "periods and restart_weights should have the same length."
        self.cumulative_period = [
            sum(self.periods[0 : i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min
            + current_weight
            * 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi * ((self.last_epoch - nearest_restart) / current_period)
                )
            )
            for base_lr in self.base_lrs
        ]
