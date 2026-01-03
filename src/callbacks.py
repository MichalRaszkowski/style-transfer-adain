import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from torchvision.utils import make_grid

class ImageLogger(Callback):
    def __init__(self, num_samples=4):
        super().__init__()
        self.num_samples = num_samples
        self.fixed_batch = None 

    def on_validation_start(self, trainer, pl_module):
        if self.fixed_batch is None:
            device = pl_module.device
            val_loader = trainer.datamodule.val_dataloader()
            batch = next(iter(val_loader))
            c, s = batch
            self.fixed_batch = (c[:self.num_samples].to(device), s[:self.num_samples].to(device))

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.fixed_batch is None: return
        
        c, s = self.fixed_batch
        
        with torch.no_grad():
            gen, _ = pl_module(c, s)

        imgs = []
        for i in range(len(c)):
            imgs.append(c[i].cpu())
            imgs.append(s[i].cpu())
            imgs.append(gen[i].cpu())
        
        imgs_stack = torch.stack(imgs)
        
        imgs_stack = torch.clamp(imgs_stack, 0, 1)

        from torchvision.utils import make_grid
        grid = make_grid(imgs_stack, nrow=3, padding=2)

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log({
                "val/comparison": [
                    wandb.Image(grid, caption=f"Epoch: {trainer.current_epoch} Content | Style | Generated")
                ]
            })