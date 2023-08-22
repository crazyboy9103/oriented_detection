from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmUpMultiStepDecay(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_iters=5, warmup_start_lr=0, last_epoch=-1, verbose=False):
        # milestones: list of epoch indices to decay LR
        # gamma: decay factor
        # warmup_iters: number of epochs for warmup
        # warmup_start_lr: initial learning rate for warmup
        
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_iters = warmup_iters
        self.warmup_start_lr = warmup_start_lr

        super(LinearWarmUpMultiStepDecay, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_iters
            scale = (1 - alpha) * self.warmup_start_lr + alpha
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Multi-step decay
            return [base_lr * self.gamma ** sum(epoch < self.last_epoch for epoch in self.milestones) for base_lr in self.base_lrs]
