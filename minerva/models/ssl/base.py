from abc import abstractmethod
from typing import Optional

import torch
from lightning import LightningModule
from torchmetrics import Metric


class _SSLTechnique(LightningModule):
    """
    Abstract base class for Self-Supervised Learning (SSL) techniques implemented in PyTorch Lightning.

    Parameters
    ----------
    backbone : torch.nn.Module, optional
        Underlying neural network feature extractor.
    learning_rate : float, default 1e-4
        Base learning rate for training.
    train_metrics : Metric, optional
        TorchMetrics collection or metric to monitor during training.
    val_metrics : Metric, optional
        TorchMetrics collection or metric to monitor during validation.
    """

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
    def forward(self, x, **kwargs):
        """
        Forward pass of the SSL model.

        Parameters
        ----------
        x : Any
            Input data batch.

        Returns
        -------
        Any
            Model predictions, representations, or loss dictionary.
        """
        pass

    @abstractmethod
    def _loss_step(self, batch):
        """
        Compute SSL loss on a batch.

        Parameters
        ----------
        batch : Any
            Input data batch.

        Returns
        -------
        torch.Tensor or dict
            Computed loss or dictionary of losses.
        """
        pass

    @abstractmethod
    def training_step(self, batch, idx):
        """
        Execute a single training step.

        Parameters
        ----------
        batch : Any
            Input data batch from the training dataloader.
        idx : int
            Index of the current training batch.

        Returns
        -------
        torch.Tensor or dict
            Step loss or output dict.
        """
        pass

    @abstractmethod
    def validation_step(self, batch, idx):
        """
        Execute a single validation step.

        Parameters
        ----------
        batch : Any
            Input data batch from the validation dataloader.
        idx : int
            Index of the current validation batch.

        Returns
        -------
        torch.Tensor or dict
            Step loss or output dict.
        """
        pass

    def technique_callbacks(self):
        """
        Return technique-specific Lightning callbacks.

        Returns
        -------
        list of Callback or None
            List of custom callbacks, or None if none required.
        """
        return None

    def default_train_strategy(self):
        """
        Return default distributed training strategy for this SSL technique.

        Returns
        -------
        Strategy or None
            Lightning training strategy, or None.
        """
        return None

    def default_technique_transforms(self):
        """
        Return default data transformation pipeline for this technique.

        Returns
        -------
        callable or None
            Technique-specific transform callable, or None.
        """
        return None

    def default_technique_collate_fn(self):
        """
        Return default dataloader collate function for this technique.

        Returns
        -------
        callable or None
            Collate function, or None.
        """
        return None

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns
        -------
        Optimizer, tuple, or None
            Optimizer configuration for Lightning.
        """
        return None

    def _compute_metrics(self, batch, batch_idx):
        """
        Compute and log metrics for the batch.

        Parameters
        ----------
        batch : Any
            Current batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        dict or None
            Computed metric values.
        """
        return None

    # Method to enable per epoch dataset seed
    def on_train_epoch_start(self):
        """Notify dataset of current epoch index to allow epoch-based deterministic seeding."""
        dataset = None
        if hasattr(self.trainer.datamodule, "train_dataset"):
            dataset = self.trainer.datamodule.train_dataset
        elif hasattr(self.trainer.datamodule, "dataset"):
            dataset = self.trainer.datamodule.dataset

        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.current_epoch)
