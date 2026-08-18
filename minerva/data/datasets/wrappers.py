import wrapt
from typing import Any, Callable, Optional, Union, Tuple

from minerva.data.datasets.base import SimpleDataset
from torch.utils.data import Dataset


# TODO: update wrapper to follow new transforms specification
class SSLDatasetWrapper(wrapt.ObjectProxy):
    """
    Wrapper around a dataset to apply self-supervised learning technique transforms.

    Parameters
    ----------
    dataset : Dataset or SimpleDataset
        The underlying PyTorch or Minerva dataset to wrap.
    technique_transforms : callable, optional
        Transformation function or callable (such as multi-crop augmentation) to apply
        to sample inputs returned by the dataset.
    """

    def __init__(
        self,
        dataset: Union[Dataset, SimpleDataset],
        technique_transforms: Optional[Callable],
    ) -> None:
        super().__init__(dataset)
        self.__wrapped__ = dataset
        self.__transforms__ = technique_transforms

    def __getitem__(self, idx: int) -> Union[Any, Tuple[Any, ...]]:
        """
        Get sample by index with SSL technique transforms applied.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        Any or tuple of Any
            Sample with technique transforms applied to the primary input element.
        """
        item = self.__wrapped__[idx]

        if self.__transforms__ is not None:
            if isinstance(item, (tuple, list)):
                aux = [i for i in item]
                aux[0] = self.__transforms__(
                    aux[0]
                )  # TODO check techniques that need to apply transform in more then one item
                item = tuple(aux)
            else:
                item = self.__transforms__(item)

        return item
