from abc import abstractmethod
from typing import Optional

import torch
from lightning import LightningModule
from torchmetrics import Metric


class _SSLTechnique(LightningModule):
    def __init__(
        self,
        backbone: Optional[torch.nn.Module] = None,
        learning_rate: float = 1e-4,
        train_metrics: Optional[Metric] = None,
        val_metrics: Optional[Metric] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.learning_rate = learning_rate
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def _loss_step(self, batch):
        pass

    @abstractmethod
    def training_step(self, batch, idx):
        pass

    @abstractmethod
    def validation_step(self, batch, idx):
        pass

    def technique_callbacks(self):
        return None

    def default_train_strategy(self):
        return None

    def default_technique_transforms(self):
        return None

    def default_technique_collate_fn(self):
        return None

    def configure_optimizers(self):
        return None

    def _compute_metrics(self, batch, batch_idx):
        return None

    # Method to enable per epoch dataset seed
    def on_train_epoch_start(self):
        if hasattr(self.trainer.datamodule, "train_dataset"):
            dataset = self.trainer.datamodule.train_dataset
        elif hasattr(self.trainer.datamodule, "dataset"):
            dataset = self.trainer.datamodule.dataset

        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.current_epoch)
