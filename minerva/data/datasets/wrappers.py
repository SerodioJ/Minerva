import wrapt
from typing import Any, Callable, Optional, Union, Tuple

from minerva.data.datasets.base import SimpleDataset
from torch.utils.data import Dataset


class SSLDatasetWrapper(wrapt.ObjectProxy):
    def __init__(
        self,
        dataset: Union[Dataset, SimpleDataset],
        technique_transforms: Optional[Callable],
    ) -> None:
        super().__init__(dataset)
        self.__wrapped__ = dataset
        self.__transforms__ = technique_transforms

    def __getitem__(self, idx: int) -> Union[Any, Tuple[Any, ...]]:
        item = self.__wrapped__[idx]

        if self.__transforms__ is not None:
            aux = [i for i in item]
            aux[0] = self.__transforms__(
                aux[0]
            )  # TODO check techniques that need to apply transform in more then one item
            item = tuple(aux)

        return item
