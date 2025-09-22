
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

import math
from torch.optim.lr_scheduler import _LRScheduler


def get_trainer(config, device):
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=config.base.save_path, name=config.base.exp_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=tb_logger.log_dir,
        filename="{step:06d}",
        save_top_k=-1,         
        every_n_train_steps=config.experiment.save_steps, 
    )

    if config.use_peft:
        trainer = Trainer(
            devices=config.base.world_size,
            accelerator=device,
            logger=tb_logger,
            default_root_dir=config.base.save_path,
            callbacks=[pl.callbacks.ModelSummary(max_depth=2), checkpoint_callback], 
            strategy=DDPStrategy(                                          
                find_unused_parameters=False # LoRA issue
            ),          
            log_every_n_steps=config.experiment.log_steps,
            gradient_clip_val=config.experiment.gradient_clip_val, 
            enable_checkpointing=config.experiment.enable_checkpointing,
            accumulate_grad_batches=config.experiment.gradient_accumulation_steps,
            precision="bf16" if config.experiment.precision is None or config.experiment.precision == "auto" else config.experiment.precision, #config.precision, 
            max_steps=config.experiment.max_training_steps, # or max_epochs   
            # check_val_every_n_epoch=None, # no validation
            val_check_interval=config.experiment.val_steps * config.experiment.gradient_accumulation_steps, 
            num_sanity_val_steps=0, # no val sanity check
        )
        
    else:
        trainer = Trainer(
            devices=config.base.world_size,
            accelerator=device,
            logger=tb_logger,
            default_root_dir=config.base.save_path,
            callbacks=[pl.callbacks.ModelSummary(max_depth=2), checkpoint_callback], 
            strategy=DDPStrategy(                                          
                find_unused_parameters=False 
            ),          
            log_every_n_steps=config.experiment.log_steps,
            gradient_clip_val=config.experiment.gradient_clip_val, 
            enable_checkpointing=config.experiment.enable_checkpointing,
            accumulate_grad_batches=config.experiment.gradient_accumulation_steps,
            precision="bf16" if config.experiment.precision is None or config.experiment.precision == "auto" else config.experiment.precision, #config.precision, 
            max_steps=config.experiment.max_training_steps, # or max_epochs   
            # check_val_every_n_epoch=None, # no validation
            val_check_interval=config.experiment.val_steps * config.experiment.gradient_accumulation_steps,        
        )

    return trainer


# https://gaussian37.github.io/dl-pytorch-lr_scheduler/#cosineannealingwarmrestarts-1
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineDecayWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, warmup_iter, max_iter, eta_min=0, eta_max=1.5e-4, last_epoch=-1):
        self.warmup_iter = warmup_iter
        self.max_iter = max_iter
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.iteration = 0

        super(CosineDecayWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.iteration < self.warmup_iter:
            lr = self.eta_max * self.iteration / self.warmup_iter
        elif self.iteration > self.max_iter:
            lr = self.eta_min
        else:
            decay_ratio = (self.iteration - self.warmup_iter) / (self.max_iter - self.warmup_iter)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.eta_min + (self.eta_max - self.eta_min) * coeff
        return lr
                
    def step(self, epoch=None):
        self.iteration += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = param_group["lr_scale"] * lr
            else:
                param_group['lr'] = lr
