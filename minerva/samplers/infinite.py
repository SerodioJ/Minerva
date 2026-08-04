# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# References:
#   https://github.com/facebookresearch/dinov3/blob/main/dinov3/data/samplers.py

import itertools
from typing import Any
from torch.utils.data.sampler import Sampler

import torch
import numpy as np
import warnings


class InfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        advance: int = 0,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._rank = rank
        self._world_size = world_size
        self._advance = advance

    def __iter__(self):
        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._rank, None, self._world_size)

    def _shuffled_iterator(self):
        assert self._shuffle

        # Instantiate a generator here (rather than in the ctor) to keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator().manual_seed(self._seed)

        while True:
            iterable = _generate_randperm_indices(
                size=self._sample_count, generator=generator
            )
            yield from itertools.islice(iterable, self._rank, None, self._world_size)


class ShardedInfiniteSampler(Sampler):
    def __init__(
        self,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 2,
        rank: int = 0,
        world_size: int = 1,
        advance: int = 0,
        use_new_shuffle_tensor_slice: bool = False,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._rank = rank
        self._world_size = world_size
        self._advance = advance
        self._iter_count = 0
        self._shuffle_tensor_slice_fn = (
            _new_shuffle_tensor_slice
            if use_new_shuffle_tensor_slice
            else _shuffle_tensor_slice
        )

    def __iter__(self):
        iter_count = self._advance // self._sample_count
        if iter_count > 0:
            self._advance -= iter_count * self._sample_count
            self._iter_count += iter_count

        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._rank, None, self._world_size)

    def _shuffled_iterator(self):
        assert self._shuffle

        # Instantiate a generator here (rather than in the ctor) to be keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator()

        # Always shuffle everything first
        generator.manual_seed(self._seed)
        dtype = _get_torch_dtype(self._sample_count)
        perm = torch.randperm(self._sample_count, dtype=dtype, generator=generator)

        while True:
            # Re-seed on each iteration to allow skipping whole permutations
            seed = _make_seed(self._seed, self._rank, self._iter_count)
            generator.manual_seed(seed)

            iterable = self._shuffle_tensor_slice_fn(
                tensor=perm,
                start=self._rank,
                step=self._world_size,
                generator=generator,
            )
            yield from iterable
            self._iter_count += 1


def _get_numpy_dtype(size: int) -> Any:
    return np.int32 if size <= 2**31 else np.int64


def _get_torch_dtype(size: int) -> Any:
    return torch.int32 if size <= 2**31 else torch.int64


def _generate_randperm_indices(*, size: int, generator: torch.Generator):
    """Generate the indices of a random permutation."""
    dtype = _get_torch_dtype(size)
    # This is actually matching PyTorch's CPU implementation, see: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorFactories.cpp#L900-L921
    perm = torch.arange(size, dtype=dtype)
    for i in range(size):
        j = torch.randint(i, size, size=(1,), generator=generator).item()

        # Always swap even if no-op
        value = perm[j].item()
        perm[j] = perm[i].item()
        perm[i] = value
        yield value


# The following function is somewhat equivalent to _new_shuffle_tensor_slice below,
# but avoids a full in-place random permutation generation.
def _shuffle_tensor_slice(
    *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    stop = len(tensor)
    count = stop // step
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}", stacklevel=1)

    dtype = _get_numpy_dtype(stop)
    result = np.empty(count, dtype=dtype)

    for i in range(count):
        j = (
            torch.randint(0, i + 1, size=(1,), generator=generator).item()
            if i > 0
            else 0
        )

        result[i] = result[j]
        result[j] = tensor[start + i * step].item()

    return result


def _new_shuffle_tensor_slice(
    *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    stop = len(tensor)
    count = stop // step
    dtype = torch.int64  # Needed for using randperm result as indices
    count = stop // step
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}", stacklevel=1)
    indices = torch.randperm(count, dtype=dtype, generator=generator)
    return tensor[start::step][indices].numpy()


def _make_seed(seed: int, start: int, iter_count: int) -> int:
    # NOTE: Tried a few variants (including iter_count << 32), this one worked best.
    return seed + start + (iter_count << 24)
