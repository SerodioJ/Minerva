from lightning import Callback, LightningModule, Trainer
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from typing import Optional, List, Union, Dict
from pathlib import Path


class SpecificCheckpointCallback(Callback):
    def __init__(
        self,
        specific_epochs: Optional[List[Union[Dict, int]]] = None,
        specific_steps: Optional[List[Union[Dict, int]]] = None,
        epoch_var_name: Optional[str] = None,
        step_var_name: Optional[str] = None,
    ):
        """
        Callback to save model checkpoints at specific epochs and/or steps.

        Parameters
        ----------
        specific_epochs : list of dict or int, optional
            A list specifying the epoch indices at which to save the checkpoints.
            Each item can be an integer epoch index (starting at 0) or a dictionary
            defining a range of epoch indexes. If -1 is included, the model initial
            random weights will be saved.
        specific_steps : list of dict or int, optional
            A list specifying the step indices at which to save the checkpoints.
            Each item can be an integer step index (starting at 1) or a dictionary
            defining a range of step indexes.
        epoch_var_name : string, optional
            The name of the trainer attribute that holds the current epoch, by default
            None. If None, 'current_epoch' is used.
        step_var_name : string, optional
            The name of the trainer attribute that holds the current step, by default
            None. If None, 'global_step' is used.
        """
        super().__init__()
        self.checkpoint_path = None
        self.specific_epochs = specific_epochs or []
        self.specific_steps = specific_steps or []
        self.epoch_var_name = epoch_var_name or "current_epoch"
        self.step_var_name = step_var_name or "global_step"
        epochs_expanded = []
        for value in self.specific_epochs:
            if type(value) is int:
                epochs_expanded += [value]
            elif type(value) is dict:
                epochs_expanded += list(
                    range(value["start"], value["stop"], value["step"])
                )
        self.specific_epochs = epochs_expanded
        steps_expanded = []
        for value in self.specific_steps:
            if type(value) is int:
                steps_expanded += [value]
            elif type(value) is dict:
                steps_expanded += list(
                    range(value["start"], value["stop"], value["step"])
                )
        self.specific_steps = steps_expanded

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        """
        It creates the checkpoints folder at the start of the training. If required,
        it also saves the model initial random weights.
        """
        self.checkpoint_path = Path(trainer.log_dir) / "checkpoints"
        self.checkpoint_path.mkdir(exist_ok=True)
        if -1 in self.specific_epochs:
            filename = "epoch=-1.ckpt"
            trainer.save_checkpoint(self.checkpoint_path / filename)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """
        Checks the epoch index and saves the checkpoint if specified.
        """
        current_epoch = None
        if hasattr(pl_module, self.epoch_var_name):
            current_epoch = getattr(pl_module, self.epoch_var_name)
        elif hasattr(trainer, self.epoch_var_name):
            current_epoch = getattr(trainer, self.epoch_var_name)
        if current_epoch is not None and current_epoch in self.specific_epochs:
            filename = f"epoch={current_epoch}.ckpt"
            trainer.save_checkpoint(self.checkpoint_path / filename)

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx
    ):
        """
        Checks the step index and saves the checkpoint if specified.
        """
        global_step = None
        if hasattr(pl_module, self.step_var_name):
            global_step = getattr(pl_module, self.step_var_name)
        elif hasattr(trainer, self.step_var_name):
            global_step = getattr(trainer, self.step_var_name)
        if global_step is not None and global_step in self.specific_steps:
            filename = f"step={global_step}.ckpt"
            trainer.save_checkpoint(self.checkpoint_path / filename)


class FilterWeights(Callback):
    def __init__(self, filter_values: Union[str, List[str]]):
        super().__init__()
        self.filter_values = (
            filter_values if isinstance(filter_values, list) else [filter_values]
        )

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        state_dict = checkpoint["state_dict"]
        for value in self.filter_values:
            # Find keys that needs to be filtered
            keys_to_remove = [k for k in state_dict.keys() if value in k]

            # Remove them from the dictionary before it saves
            for k in keys_to_remove:
                del state_dict[k]


class EvalCheckpointCallback(Callback):
    def __init__(self, period: int, name: str, model_attr: str = "model_ema"):
        super().__init__()
        self.period = period
        self.name = name
        self.model_attr = model_attr

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if self.period > 0 and (step + 1) % self.period == 0:
            root = trainer.log_dir or trainer.default_root_dir
            ckpt_dir = Path(root) / "eval" / f"training_{step}" / self.name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model = getattr(pl_module, self.model_attr) 
            state_dict = get_model_state_dict(model)
            dcp.save(state_dict=state_dict, checkpoint_id=str(ckpt_dir))

            trainer.strategy.barrier("async_checkpoint_save")
