from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import StepLR

class WarmupStepLRSchedule(LambdaLR):
    """ Linear warmup and step decay.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.step_scheduler = StepLR(optimizer, step_size=200, gamma=0.5) 
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            return self.step_scheduler.get_lr()[0]
        
        super(WarmupStepLRSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)


# scheduler = StepLR(optimizer, step_size=200, gamma=0.5)


# lr_config = dict(
# policy='step',
# warmup='linear',
# warmup_iters=500,
# warmup_ratio=1.0 / 3,
# step=[8, 11])

