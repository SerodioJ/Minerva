import pytest
import torch
import torch.nn as nn

from minerva.models.nets.image.dino import (
    DINOHead,
    _build_mlp,
    get_activation_checkpoint_wrapper,
    wrap_compile_block,
)


def test_dino_head_v3_forward():
    in_dim = 128
    out_dim = 64
    bottleneck_dim = 32
    head = DINOHead(
        in_dim=in_dim,
        out_dim=out_dim,
        nlayers=2,
        hidden_dim=128,
        bottleneck_dim=bottleneck_dim,
        dino_version=3,
    )
    head.init_weights()

    x = torch.randn(4, in_dim)
    out = head(x)
    assert out.shape == (4, out_dim)

    # Test no_last_layer
    out_no_last = head(x, no_last_layer=True)
    assert out_no_last.shape == (4, bottleneck_dim)

    # Test only_last_layer
    x_bottleneck = torch.randn(4, bottleneck_dim)
    out_only_last = head(x_bottleneck, only_last_layer=True)
    assert out_only_last.shape == (4, out_dim)


def test_dino_head_v2_forward():
    in_dim = 64
    out_dim = 32
    bottleneck_dim = 16
    head = DINOHead(
        in_dim=in_dim,
        out_dim=out_dim,
        nlayers=3,
        hidden_dim=64,
        bottleneck_dim=bottleneck_dim,
        dino_version=2,
    )
    assert hasattr(head, "weight_g")
    head.init_weights()
    assert torch.all(head.weight_g == 1.0)

    x = torch.randn(2, in_dim)
    out = head(x)
    assert out.shape == (2, out_dim)


def test_dino_head_single_layer():
    head = DINOHead(
        in_dim=32,
        out_dim=16,
        nlayers=1,
        bottleneck_dim=16,
        dino_version=3,
    )
    x = torch.randn(3, 32)
    out = head(x)
    assert out.shape == (3, 16)


def test_dino_head_with_bn():
    head = DINOHead(
        in_dim=32,
        out_dim=16,
        use_bn=True,
        nlayers=3,
        hidden_dim=32,
        bottleneck_dim=16,
        dino_version=3,
    )
    x = torch.randn(4, 32)
    out = head(x)
    assert out.shape == (4, 16)


def test_dino_head_invalid_version():
    with pytest.raises(ValueError):
        DINOHead(in_dim=32, out_dim=16, dino_version=1)


def test_build_mlp():
    mlp1 = _build_mlp(nlayers=1, in_dim=16, bottleneck_dim=8)
    assert isinstance(mlp1, nn.Linear)

    mlp3 = _build_mlp(
        nlayers=3, in_dim=16, bottleneck_dim=8, hidden_dim=32, use_bn=True
    )
    assert isinstance(mlp3, nn.Sequential)
    x = torch.randn(4, 16)
    out = mlp3(x)
    assert out.shape == (4, 8)


def test_wrap_compile_block():
    layer = nn.Linear(8, 8)
    compiled = wrap_compile_block(layer, use_cuda_graphs=False, is_backbone_block=False)
    assert compiled is not None


def test_activation_checkpoint_wrapper():
    wrapper_full = get_activation_checkpoint_wrapper(checkpointing_full=True)
    assert callable(wrapper_full)
    wrapper_selective = get_activation_checkpoint_wrapper(checkpointing_full=False)
    assert callable(wrapper_selective)
