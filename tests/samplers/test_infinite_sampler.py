import itertools
import numpy as np
import pytest
import torch

from minerva.samplers.infinite import (
    InfiniteSampler,
    ShardedInfiniteSampler,
    _generate_randperm_indices,
    _get_numpy_dtype,
    _get_torch_dtype,
    _make_seed,
    _new_shuffle_tensor_slice,
    _shuffle_tensor_slice,
)


def test_infinite_sampler_unshuffled():
    sampler = InfiniteSampler(sample_count=5, shuffle=False)
    # Take first 12 items
    items = list(itertools.islice(iter(sampler), 12))
    assert items == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]


def test_infinite_sampler_shuffled_deterministic():
    sampler1 = InfiniteSampler(sample_count=10, shuffle=True, seed=42)
    sampler2 = InfiniteSampler(sample_count=10, shuffle=True, seed=42)
    items1 = list(itertools.islice(iter(sampler1), 25))
    items2 = list(itertools.islice(iter(sampler2), 25))
    assert items1 == items2
    # Ensure all elements from 0 to 9 are generated in first 10
    assert sorted(items1[:10]) == list(range(10))


def test_infinite_sampler_advance():
    sampler_no_advance = InfiniteSampler(sample_count=10, shuffle=False, advance=0)
    sampler_advance = InfiniteSampler(sample_count=10, shuffle=False, advance=4)

    items_all = list(itertools.islice(iter(sampler_no_advance), 20))
    items_advanced = list(itertools.islice(iter(sampler_advance), 16))

    assert items_all[4:20] == items_advanced


def test_infinite_sampler_world_size_sharding():
    world_size = 3
    sample_count = 12
    samplers = [
        InfiniteSampler(
            sample_count=sample_count,
            shuffle=False,
            rank=r,
            world_size=world_size,
        )
        for r in range(world_size)
    ]

    # For one epoch (sample_count items in total, sample_count // world_size per rank)
    epoch_items = [
        list(itertools.islice(iter(samplers[r]), sample_count // world_size))
        for r in range(world_size)
    ]

    assert epoch_items[0] == [0, 3, 6, 9]
    assert epoch_items[1] == [1, 4, 7, 10]
    assert epoch_items[2] == [2, 5, 8, 11]

    # Combined indices should cover range(12)
    combined = sorted([idx for rank_list in epoch_items for idx in rank_list])
    assert combined == list(range(sample_count))


def test_sharded_infinite_sampler_unshuffled():
    sampler = ShardedInfiniteSampler(sample_count=6, shuffle=False)
    items = list(itertools.islice(iter(sampler), 15))
    assert items == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2]


def test_sharded_infinite_sampler_shuffled_default_slice():
    sampler = ShardedInfiniteSampler(
        sample_count=10,
        shuffle=True,
        seed=123,
        use_new_shuffle_tensor_slice=False,
    )
    items = list(itertools.islice(iter(sampler), 30))
    assert len(items) == 30
    assert sorted(items[:10]) == list(range(10))
    assert sorted(items[10:20]) == list(range(10))


def test_sharded_infinite_sampler_shuffled_new_slice():
    sampler = ShardedInfiniteSampler(
        sample_count=10,
        shuffle=True,
        seed=123,
        use_new_shuffle_tensor_slice=True,
    )
    items = list(itertools.islice(iter(sampler), 30))
    assert len(items) == 30
    assert sorted(items[:10]) == list(range(10))
    assert sorted(items[10:20]) == list(range(10))


def test_sharded_infinite_sampler_advance():
    sample_count = 8
    advance = 10  # 1 full iteration + 2 samples
    sampler_no_advance = ShardedInfiniteSampler(
        sample_count=sample_count, shuffle=False, advance=0
    )
    sampler_advance = ShardedInfiniteSampler(
        sample_count=sample_count, shuffle=False, advance=advance
    )

    items_all = list(itertools.islice(iter(sampler_no_advance), 25))
    items_adv = list(itertools.islice(iter(sampler_advance), 15))
    assert items_all[10:25] == items_adv


def test_sharded_infinite_sampler_world_size():
    world_size = 2
    sample_count = 10
    sampler_r0 = ShardedInfiniteSampler(
        sample_count=sample_count,
        shuffle=False,
        rank=0,
        world_size=world_size,
    )
    sampler_r1 = ShardedInfiniteSampler(
        sample_count=sample_count,
        shuffle=False,
        rank=1,
        world_size=world_size,
    )

    r0_items = list(itertools.islice(iter(sampler_r0), 5))
    r1_items = list(itertools.islice(iter(sampler_r1), 5))

    assert r0_items == [0, 2, 4, 6, 8]
    assert r1_items == [1, 3, 5, 7, 9]


def test_helper_dtypes():
    assert _get_numpy_dtype(100) == np.int32
    assert _get_numpy_dtype(2**32) == np.int64
    assert _get_torch_dtype(100) == torch.int32
    assert _get_torch_dtype(2**32) == torch.int64


def test_generate_randperm_indices():
    generator = torch.Generator().manual_seed(42)
    indices = list(_generate_randperm_indices(size=10, generator=generator))
    assert len(indices) == 10
    assert sorted(indices) == list(range(10))


def test_shuffle_tensor_slice_functions():
    generator = torch.Generator().manual_seed(42)
    tensor = torch.arange(20)

    slice1 = _shuffle_tensor_slice(tensor=tensor, start=0, step=2, generator=generator)
    assert len(slice1) == 10
    assert sorted(slice1.tolist()) == list(range(0, 20, 2))

    generator = torch.Generator().manual_seed(42)
    slice2 = _new_shuffle_tensor_slice(
        tensor=tensor, start=1, step=2, generator=generator
    )
    assert len(slice2) == 10
    assert sorted(slice2.tolist()) == list(range(1, 20, 2))


def test_make_seed():
    s1 = _make_seed(seed=10, start=0, iter_count=0)
    s2 = _make_seed(seed=10, start=1, iter_count=0)
    s3 = _make_seed(seed=10, start=0, iter_count=1)
    assert s1 != s2
    assert s1 != s3
