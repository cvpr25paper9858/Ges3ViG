from lightning.pytorch.callbacks import Callback
from math import cos, pi


class LrDecayImputedCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # cosine learning rate decay
        current_step = (trainer.current_epoch+1)*pl_module.hparams.cfg.data.dataloader.batch_size
        start_step = pl_module.hparams.cfg.model.lr_decay.start_epoch*pl_module.hparams.cfg.data.dataloader.batch_size
        if current_step < start_step:
            return
        end_step = pl_module.hparams.cfg.trainer.max_epochs * pl_module.hparams.cfg.data.dataloader.batch_size
        print(f"LrDecayImputedCallback:\n    current_step: {current_step},\n    start_step: {start_step},\n    end_step: {end_step}")
        clip = 1e-6
        for param_group in trainer.optimizers[0].param_groups:
            param_group['lr'] = clip + 0.5 * (pl_module.hparams.cfg.model.optimizer.lr - clip) * \
                                (1 + cos(pi * ((current_step - start_step) / (end_step - start_step))))
