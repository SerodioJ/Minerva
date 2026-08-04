from typing import Optional, Callable

import yaml

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from minerva.data.data_modules.datamodule_descriptor import DataModuleDescriptor


class MinervaDataModule(LightningDataModule):
    def __init__(
        self,
        # Datasets
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_split: Optional[str] = "test",
        # DataLoader
        dataloader_cls: type = DataLoader,
        batch_size: int = 1,
        num_workers: int = 0,
        drop_last: bool = False,
        # Dataloader overrides (batch_size, num_workers, shuffle_train)
        additional_train_dataloader_kwargs: Optional[dict] = None,
        additional_val_dataloader_kwargs: Optional[dict] = None,
        additional_test_dataloader_kwargs: Optional[dict] = None,
        shuffle_train: bool = True,
        # Sampler
        sampler_cls: Optional[Callable] = None,
        # Metadata
        name: str = "",
        additional_describes: Optional[DataModuleDescriptor] = None,
    ):
        """A fully-featured data module for PyTorch Lightning with support for
        acessing train, val, test, and predict datasets and dataloaders. This
        class is a generalization of the LightningDataModule class and is
        designed to be used with the Minerva framework.

        Parameters
        ----------
        train_dataset : Optional[Dataset], optional
            The training dataset, by default None
        val_dataset : Optional[Dataset], optional
            The validation dataset, by default None
        test_dataset : Optional[Dataset], optional
            The test dataset, by default None
        predict_split : Optional[str], optional
            Set the split to predict on (using the predict_dataloader method),
            by default "test"
        dataloader_cls : type, optional
            The dataloader class to use. The datasets will be wrapped in this
            class when creating the dataloaders, by default DataLoader
        batch_size : int, optional
            Default batch_size for all dataloaders, by default 1
        num_workers : int, optional
            Default num_workers for all dataloaders, by default 0
        drop_last : bool, optional
            Default drop_last for all dataloaders, by default False
        additional_train_dataloader_kwargs : Optional[dict], optional
            Override the default train dataloader kwargs, by default None
        additional_val_dataloader_kwargs : Optional[dict], optional
            Override the default val dataloader kwargs, by default None
        additional_test_dataloader_kwargs : Optional[dict], optional
            Override the default test dataloader kwargs, by default None
        shuffle_train : bool, optional
            If True, shuffle the training dataset. If False, do not shuffle the
            training dataset, by default True. By default, only the training
            dataloader is shuffled.
        sampler_cls: callable, optional
            Sampler class to use, if not set Torch's default will be used.
        name : str, optional
            Name of the data module, by default ""
        additional_describes : DataModuleDescriptor, optional
            A class that you can apply custom check when load MinervaDataModule.
        """
        super().__init__()

        self._name = name
        self.additional_describes = additional_describes
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset
        self._predict_split = predict_split
        self._sampler_cls = sampler_cls

        if predict_split == "train":
            self._predict_dataset = train_dataset
        elif predict_split == "val":
            self._predict_dataset = val_dataset
        elif predict_split == "test":
            self._predict_dataset = test_dataset
        elif predict_split is None:
            self._predict_dataset = None
        else:
            raise ValueError(
                f"predict_split must be one of 'train', 'val', 'test', or None. Got {predict_split}."
            )

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._shuffle_train = shuffle_train
        self._dataloader_cls = dataloader_cls

        self._train_dataloader_kwargs = self.__update_dataloader_kwargs(
            additional_train_dataloader_kwargs,
            batch_size,
            num_workers,
            drop_last,
            shuffle=shuffle_train,
        )
        self._val_dataloader_kwargs = self.__update_dataloader_kwargs(
            additional_val_dataloader_kwargs,
            batch_size,
            num_workers,
            drop_last,
            shuffle=False,
        )
        self._test_dataloader_kwargs = self.__update_dataloader_kwargs(
            additional_test_dataloader_kwargs,
            batch_size,
            num_workers,
            drop_last,
            shuffle=False,
        )
        if predict_split == "train":
            self._predict_dataloader_kwargs = self._train_dataloader_kwargs
        elif predict_split == "val":
            self._predict_dataloader_kwargs = self._val_dataloader_kwargs
        elif predict_split == "test":
            self._predict_dataloader_kwargs = self._test_dataloader_kwargs
        else:
            self._predict_dataloader_kwargs = {}

        # Monkey patch the dataloaders if the datasets are not provided
        # It is applyed at instance level to avoid breaking the class signature
        if not self._train_dataset:
            self.train_dataloader = None  # type: ignore
        if not self._val_dataset:
            self.val_dataloader = None  # type: ignore
        if not self._test_dataset:
            self.test_dataloader = None  # type: ignore
        if not self._predict_dataset:
            self.predict_dataloader = None  # type: ignore

    @property
    def dataset_name(self):
        return self._name

    @staticmethod
    def __update_dataloader_kwargs(
        additional_kwargs, batch_size, num_workers, drop_last, shuffle
    ):
        kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": shuffle,
            "drop_last": drop_last,
        }

        if additional_kwargs:
            kwargs.update(additional_kwargs)
        return kwargs

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def predict_dataset(self):
        return self._predict_dataset

    def setup_sampler(self, sample_count, kwargs):
        if self._sampler_cls is not None:
            world_size = self.trainer.world_size if self.trainer else 1
            global_rank = self.trainer.global_rank if self.trainer else 0
            sampler = self._sampler_cls(
                sample_count=sample_count,
                shuffle=kwargs.get("shuffle"),
                rank=global_rank,
                world_size=world_size,
                advance=self.trainer.global_step if self.trainer else 0
            )
            kwargs["shuffle"] = False  # avoids override with Torch's default RandomSampler
            kwargs["sampler"] = sampler
        return kwargs

    def train_dataloader(self):
        sample_count = len(self.train_dataset)
        kwargs = self.setup_sampler(sample_count, self._train_dataloader_kwargs)
        return self._dataloader_cls(self.train_dataset, **kwargs)

    def val_dataloader(self):
        sample_count = len(self.val_dataset)
        kwargs = self.setup_sampler(sample_count, self._val_dataloader_kwargs)
        return self._dataloader_cls(self.val_dataset, **kwargs)

    def test_dataloader(self):
        sample_count = len(self.test_dataset)
        kwargs = self.setup_sampler(sample_count, self._test_dataloader_kwargs)
        return self._dataloader_cls(self.test_dataset, **kwargs)

    def predict_dataloader(self):
        sample_count = len(self.predict_dataset)
        kwargs = self.setup_sampler(sample_count, self._predict_dataloader_kwargs)
        return self._dataloader_cls(self.predict_dataset, **kwargs)

    def __str__(self) -> str:
        def indent_text(text, spaces=6, add_line_breaks=True):
            """Indent each line of a string by a given number of spaces."""
            if not text:
                return "No data."

            return "\n".join(
                (
                    " " * int(spaces // 2) + "│"
                    if add_line_breaks
                    else " " * int(spaces // 2) + " "
                )
                + (" " * spaces + line if line.strip() else line)
                for line in text.split("\n")
            )

        def pretty_yaml(d, indent=6):
            """Pretty-print dictionary in YAML format with '├──' for each key."""
            if not d:
                return "No data."
            yaml_str = yaml.dump(
                d, default_flow_style=False, sort_keys=False
            ).strip()  # Remove trailing newlines
            return "\n".join(
                f"{' ' * indent}├── {line}"
                for line in yaml_str.split("\n")
                if line.strip()
            )

        return (
            f"{'=' * 50}\n"
            f"{' ' * ((50 - len(self._name)) // 2)}🆔 {self._name}\n"
            f"{'=' * 50}\n"
            f"└── Predict Split: {self._predict_split}\n"
            f"📂 Datasets:\n"
            f"{'' if self.additional_describes is None else self.additional_describes()}"
            f"   ├── Train Dataset:\n{indent_text(str(self.train_dataset))}\n"
            f"   ├── Val Dataset:\n{indent_text(str(self.val_dataset))}\n"
            f"   └── Test Dataset:\n{indent_text(str(self.test_dataset), add_line_breaks=False)}\n"
            f"\n🛠 **Dataloader Configurations:**\n"
            f"   ├── Dataloader class: {self._dataloader_cls}\n"
            f"   ├── Train Dataloader Kwargs:\n{pretty_yaml(self._train_dataloader_kwargs, indent=9)}\n"
            f"   ├── Val Dataloader Kwargs:\n{pretty_yaml(self._val_dataloader_kwargs, indent=9)}\n"
            f"   └── Test Dataloader Kwargs:\n{pretty_yaml(self._test_dataloader_kwargs, indent=9)}\n"
            f"{'=' * 50}"
        )

    def __repr__(self) -> str:
        return self.__str__()
