# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# References:
#   https://github.com/facebookresearch/dinov3/blob/main/dinov3/data/datasets/image_net.py

import os
from enum import Enum
from io import BytesIO
from typing import Callable, Optional, Tuple, Union, Literal

import numpy as np
from PIL import Image

from minerva.data.datasets.supervised_dataset import SupervisedReconstructionDataset
from minerva.data.readers.reader import _Reader


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 1_281_167,
            _Split.VAL: 50_000,
            _Split.TEST: 100_000,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(self, actual_index: int, class_id: Optional[str] = None) -> str:
        if self == _Split.TRAIN:
            dirname = self.get_dirname(class_id)
            basename = f"{class_id}_{actual_index}"
        else:  # self in (_Split.VAL, _Split.TEST):
            dirname = self.value
            basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
        return os.path.join(dirname, basename + ".JPEG")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        assert self != _Split.TEST
        dirname, filename = os.path.split(image_relpath)
        class_id = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        actual_index = int(basename.split("_")[-1])
        return class_id, actual_index


class ImageNet(SupervisedReconstructionDataset):
    Split = Union[_Split]

    def __init__(
        self,
        split: Literal["train", "test", "val"],
        root: str,
        extra: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        split = ImageNet.Split(split)
        super().__init__(
            readers=[
                _ImageNetDataReader(root, extra, split),
                _ImageNetTargetReader(root, extra, split)
            ],
            transforms=[
                transform,
                target_transform
            ],
        )
        self.root = root
        self.extra = extra
        self.split = split


class _ImageNetBaseReader(_Reader):
    def __init__(
            self,
            root: str,
            extra: str,
            split: "ImageNet.Split"
    ):
        self.root = root
        self._extra_root = extra
        self._split = split

        self._entries = None
        self._class_ids = None
        self._class_names = None

    @property
    def split(self) -> "ImageNet.Split":
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def get_class_id(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_id = entries[index]["class_id"]
        return None if self.split == _Split.TEST else str(class_id)

    def __len__(self) -> int:
        entries = self._get_entries()
        assert len(entries) == self.split.length
        return len(entries)


class _ImageNetDataReader(_ImageNetBaseReader):
    def __getitem__(self, index: int):
        try:
            entries = self._get_entries()
            actual_index = entries[index]["actual_index"]

            class_id = self.get_class_id(index)

            image_relpath = self.split.get_image_relpath(actual_index, class_id)
            image_full_path = os.path.join(self.root, image_relpath)
            with open(image_full_path, mode="rb") as f:
                image_data = f.read()
            f = BytesIO(image_data)
            image = Image.open(f).convert(mode="RGB")
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        return image


class _ImageNetTargetReader(_ImageNetBaseReader):
    def __getitem__(self, index: int):
        entries = self._get_entries()
        class_index = entries[index]["class_index"]
        return None if self.split == _Split.TEST else int(class_index)
