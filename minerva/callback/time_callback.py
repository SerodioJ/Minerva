from lightning import Callback, LightningModule, Trainer
import time


class ElapsedTimeCallback(Callback):
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        self.train_start_time = time.time()

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):
        pl_module.log(
            "elapsed_time",
            time.time() - self.train_start_time,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
