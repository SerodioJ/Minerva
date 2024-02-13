from typing import List, Tuple

import numpy as np
from base import SimpleDataset

from sslt.data.readers.reader import _Reader
from sslt.transforms.transform import _Transform


class SupervisedReconstructionDataset(SimpleDataset):
    """
    A dataset class for supervised semantic segmentation tasks.
    """

    def __init__(self, readers: List[_Reader], transforms: _Transform | None = None):
        """
        Initializes a SupervisedSemanticSegmentationDataset object.

        Parameters
        ----------
            readers: List[_Reader]
                List of data readers. It must contain exactly 2 readers.
                The first reader for the input data and the second reader for the target data.
            transforms: TransformPipeline | None
                Optional data transformation pipeline.

        Raises:
            AssertionError: If the number of readers is not exactly 2.
        """
        super().__init__(readers, transforms)

        assert (
            len(self.readers) == 2
        ), "SupervisedSemanticSegmentationDataset requires exactly 2 readers"

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.

        """
        return len(self.readers[0])

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from sources and apply specified transforms.

        Parameters
        ----------
        index : int
            The index of the sample to load.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing two numpy arrays representing the data.

        """
        data = super().__getitem__(index)

        return (data[0], data[1])
